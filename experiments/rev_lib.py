"""Shared library for the revision analyses.

Everything the revision adds is evaluated offline on stored posteriors.

* Seed 42 (the original runs) lives in experiments/probs, seeds 1-4 in experiments/probs_seeds/seed{S}.
  All seeds share the same folds and validation splits, so only fine-tuning randomness differs.
* Every fusion rule is *fitted* on validation posteriors and labels only and returns a callable that
  maps any posterior stack (test fold or transfer corpus) to (predictions, P(positive)). The same
  fitted rule can therefore be scored in domain and under corpus shift without touching test labels.
* Shrinkage estimators share one paired bootstrap of the validation split per fit, so all variants
  see the same resamples.
"""
import os, math, json, time
from collections import OrderedDict
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import binomtest, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MO = ["indobert-p1", "indobert-cahya", "indoroberta-cahya", "xlmr-base", "mbert-cased"]
MLAB = {"indobert-p1": "IndoBERT-p1", "indobert-cahya": "IndoBERT-1.5G", "indoroberta-cahya": "IndoRoBERTa-1.5G",
        "xlmr-base": "XLM-R", "mbert-cased": "mBERT"}
DS_LAB = {"evprimary-formal": "EV-Norm", "evprimary-informal": "EV-Raw", "evpublic": "EV-Public",
          "evtwitter": "EV-Twitter", "evyoutube": "EV-YouTube"}
DS_ID = {"evprimary-formal": 0, "evpublic": 1, "evprimary-informal": 2, "evtwitter": 3, "evyoutube": 4}
IN_DOMAIN = ["evprimary-formal", "evpublic", "evprimary-informal"]
SEEDS = [42, 1, 2, 3, 4]
N_FOLDS = 5
EPS = 1e-6
BETAS = [0, 2, 5, 10, 20, 50, 100]
N_BOOT = 1000


# ============================================================================ data
def _p(*a):
    return os.path.join(ROOT, *a)


def seed_dir(seed):
    return _p("experiments", "probs") if seed == 42 else _p("experiments", "probs_seeds", f"seed{seed}")


def fold_complete(ds, fold, seed):
    """all five posterior files and their training histories exist (run_seeds.py writes the history
    right after the posteriors, so requiring both avoids reading a seed that is still being finished)"""
    d = os.path.join(seed_dir(seed), ds, f"fold{fold}")
    return all(os.path.exists(os.path.join(d, f"{m}.npz")) and os.path.exists(os.path.join(d, f"{m}_history.json"))
               for m in MO)


def complete_seeds(ds, seeds=SEEDS):
    return [s for s in seeds if all(fold_complete(ds, f, s) for f in range(N_FOLDS))]


_CACHE = {}


def load_fold(ds, fold, seed):
    """Posteriors of the five learners for one (corpus, fold, seed)."""
    key = (ds, fold, seed)
    if key in _CACHE:
        return _CACHE[key]
    d = os.path.join(seed_dir(seed), ds, f"fold{fold}")
    Pv, Pt, X, hist, cost = [], [], {}, {}, {}
    yv = yt = vi = ti = None
    for m in MO:
        z = np.load(os.path.join(d, f"{m}.npz"))
        Pv.append(z["val_probs"].astype(np.float64)); Pt.append(z["test_probs"].astype(np.float64))
        yv, yt, vi, ti = z["y_val"].astype(int), z["y_test"].astype(int), z["val_idx"], z["test_idx"]
        for k in z.files:
            if k.startswith("xfer_"):
                X.setdefault(k[5:], []).append(z[k].astype(np.float64))
            elif k == "transfer_probs":                       # seed-42 runs: EV-Norm -> EV-Public only
                X.setdefault("evpublic", []).append(z[k].astype(np.float64))
            elif k.startswith("cost_"):
                cost.setdefault(m, {})[k[5:]] = float(z[k])
        hp = os.path.join(d, f"{m}_history.json")
        try:
            hist[m] = json.load(open(hp)) if os.path.exists(hp) else None
        except json.JSONDecodeError as e:                 # history still being written by run_seeds.py
            raise RuntimeError(f"{hp} is incomplete; rerun once training has finished") from e
    out = dict(Pv=np.stack(Pv), yv=yv, Pt=np.stack(Pt), yt=yt, val_idx=vi, test_idx=ti,
               xfer={k: np.stack(v) for k, v in X.items() if len(v) == len(MO)}, hist=hist, cost=cost)
    _CACHE[key] = out
    return out


def target_labels(name):
    p = _p("experiments", "probs_seeds", f"labels_{name}.npz")
    if os.path.exists(p):
        y = np.load(p)["y"].astype(int)
        if name == "evpublic":
            y42 = np.load(_p("experiments", "probs", "transfer_labels.npz"))["y"].astype(int)
            assert np.array_equal(y, y42), "EV-Public labels differ between the seed-42 and the new runs"
        return y
    if name == "evpublic":
        return np.load(_p("experiments", "probs", "transfer_labels.npz"))["y"].astype(int)
    raise FileNotFoundError(p)


def fold_rng(seed, ds, fold, rep=0, salt=0):
    return np.random.default_rng([int(seed), DS_ID.get(ds, 9), int(fold), int(rep), int(salt)])


# ========================================================================= metrics
def macro_f1(y, p):
    f = 0.0
    for c in (0, 1):
        tp = np.count_nonzero((p == c) & (y == c))
        fp = np.count_nonzero((p == c) & (y != c))
        fn = np.count_nonzero((p != c) & (y == c))
        d = 2 * tp + fp + fn
        f += (2.0 * tp / d) if d else 0.0
    return f / 2.0


def full_metrics(y, p):
    pr, rc = [], []
    for c in (0, 1):
        tp = np.count_nonzero((p == c) & (y == c))
        fp = np.count_nonzero((p == c) & (y != c))
        fn = np.count_nonzero((p != c) & (y == c))
        pr.append(tp / (tp + fp) if tp + fp else 0.0)
        rc.append(tp / (tp + fn) if tp + fn else 0.0)
    return {"acc": float(np.mean(y == p)), "prec": float(np.mean(pr)), "rec": float(np.mean(rc)),
            "f1": macro_f1(y, p), "rec_neg": float(rc[0]), "rec_pos": float(rc[1])}


def _chunks(M, B, n, budget=2 * 10 ** 7):
    step = max(1, budget // max(M * n, 1))
    return [slice(i, min(i + step, B)) for i in range(0, B, step)]


def macro_f1_batch(y, preds, idx=None):
    """preds (M, n). Returns macro-F1 per learner (M,), or per learner and resample (M, B) given idx (B, n)."""
    single = idx is None
    if single:
        idx = np.arange(len(y))[None]
    y8, p8 = np.asarray(y).astype(np.int8), np.asarray(preds).astype(np.int8)
    M, B = p8.shape[0], idx.shape[0]
    f = np.zeros((M, B))
    for sl in _chunks(M, B, idx.shape[1]):
        yb, pb = y8[idx[sl]], p8[:, idx[sl]]
        for c in (0, 1):
            pc, yc = pb == c, (yb == c)[None]
            tp = (pc & yc).sum(-1); fp = (pc & ~yc).sum(-1); fn = (~pc & yc).sum(-1)
            d = 2 * tp + fp + fn
            f[:, sl] += np.where(d > 0, 2.0 * tp / np.maximum(d, 1), 0.0)
    f /= 2.0
    return f[:, 0] if single else f


def accuracy_batch(y, preds, idx=None):
    single = idx is None
    if single:
        idx = np.arange(len(y))[None]
    y8, p8 = np.asarray(y).astype(np.int8), np.asarray(preds).astype(np.int8)
    M, B = p8.shape[0], idx.shape[0]
    a = np.zeros((M, B))
    for sl in _chunks(M, B, idx.shape[1]):
        a[:, sl] = (p8[:, idx[sl]] == y8[idx[sl]][None]).mean(-1)
    return a[:, 0] if single else a


def nll(y, P):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def brier(y, P):
    """binary Brier score, mean squared error of P(positive)"""
    return float(np.mean((P[:, 1] - y) ** 2))


def ece(y, P, n_bins=15, adaptive=False):
    """top-label expected calibration error; equal-width bins, or equal-mass bins if adaptive"""
    conf, pred = P.max(-1), P.argmax(-1)
    ok = (pred == y).astype(float)
    n = len(y)
    if adaptive:
        order = np.argsort(conf, kind="stable")
        groups = np.array_split(order, n_bins)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        b = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
        groups = [np.flatnonzero(b == i) for i in range(n_bins)]
    return float(sum(len(g) / n * abs(ok[g].mean() - conf[g].mean()) for g in groups if len(g)))


def auroc(score, label):
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney)"""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ===================================================================== components
def conf_entropy(P):
    Pc = np.clip(P, EPS, 1.0)
    return 1.0 - (-(Pc * np.log(Pc)).sum(-1)) / math.log(P.shape[-1])


def conf_maxprob(P):
    C = P.shape[-1]
    return (P.max(-1) - 1.0 / C) / (1.0 - 1.0 / C)


def logodds(f):
    f = np.clip(np.asarray(f, float), 0.5 + EPS, 1 - EPS)
    return np.log(f / (1 - f))


def softmax_w(f, beta):
    z = beta * np.asarray(f, float); z -= z.max(); e = np.exp(z); return e / e.sum()


def _softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def fuse(P, w=None, c=None):
    """weighted soft vote; per-instance weights w_m * c_m(x), renormalised over learners"""
    M, n, _ = P.shape
    W = np.ones((M, n)) if w is None else np.repeat(np.asarray(w, float)[:, None], n, 1)
    if c is not None:
        W = W * np.clip(c, EPS, 1.0)
    W = W / np.clip(W.sum(0, keepdims=True), EPS, None)
    return (W[:, :, None] * P).sum(0)


def majority(P, tiebreak):
    v = P.argmax(-1)
    counts = np.stack([(v == c).sum(0) for c in range(P.shape[-1])], -1)
    pred = counts.argmax(-1)
    tie = counts.max(-1) == np.sort(counts, -1)[:, -2]
    pred[tie] = tiebreak.argmax(-1)[tie]
    return pred


def fit_temperature(P, y):
    """temperature minimising validation NLL; posteriors are stored, so log P serves as the logits"""
    z = np.log(np.clip(P, 1e-12, 1.0))
    idx = np.arange(len(y))

    def f(logT):
        s = z / math.exp(logT)
        s = s - s.max(1, keepdims=True)
        return float(-(s[idx, y] - np.log(np.exp(s).sum(1))).mean())

    r = minimize_scalar(f, bounds=(math.log(0.05), math.log(20.0)), method="bounded", options={"xatol": 1e-4})
    return float(math.exp(r.x))


def apply_temperature(P, T):
    """P (..., C); T scalar or broadcastable over the leading (learner) axis"""
    T = np.asarray(T, float)
    if T.ndim == 1:
        T = T.reshape((-1,) + (1,) * (P.ndim - 1))
    return _softmax(np.log(np.clip(P, 1e-12, 1.0)) / T)


# ====================================================================== shrinkage
def _js_from_noise(F, noise_trace):
    """positive-part James-Stein shrinkage of F toward its mean.

    noise_trace estimates E||d||^2 under 'all learners equally good', d = F - mean(F). For M
    independent estimates with common variance s^2 it equals (M-1) s^2, and the factor below is
    then exactly the classical 1 - (M-3) s^2 / ||d||^2.
    """
    F = np.asarray(F, float); M = len(F); Fbar = F.mean()
    ss = float(((F - Fbar) ** 2).sum())
    lam = 0.0 if (ss <= 0 or M <= 3) else max(0.0, 1.0 - ((M - 3) / (M - 1)) * noise_trace / ss)
    return Fbar + lam * (F - Fbar), {"lam": lam, "noise": float(noise_trace), "ss": ss}


def shrink_js_binomial(F, n):
    """the original SA-WSV estimator: binomial variance F(1-F)/n, independence across learners"""
    F = np.asarray(F, float); M = len(F)
    var = float(np.mean(F * (1.0 - F)) / max(n, 1))
    return _js_from_noise(F, (M - 1) * var)


def shrink_js_boot_marginal(F, Fb):
    """bootstrap variance of each learner's macro-F1, still treating the learners as independent"""
    M = len(F)
    return _js_from_noise(F, (M - 1) * float(Fb.var(1, ddof=1).mean()))


def shrink_js_boot_paired(F, Fb):
    """paired bootstrap of the *centred* vector d = F - mean(F): all learners are resampled on the same
    validation instances, so the noise in their differences, including the correlation induced by
    the shared split, is estimated directly. This is the estimator used by SA-WSV in the revision."""
    D = Fb - Fb.mean(0, keepdims=True)
    return _js_from_noise(F, float(D.var(1, ddof=1).sum()))


def shrink_bock(F, Fb):
    """Bock's (1975) generalisation of James-Stein to a non-spherical covariance, applied to d."""
    F = np.asarray(F, float); Fbar = F.mean(); d = F - Fbar
    S = np.cov(Fb - Fb.mean(0, keepdims=True))
    ev = np.linalg.eigvalsh(S)
    tr, lmax = float(ev.sum()), float(ev.max())
    q, eff = 0.0, (tr / lmax if lmax > 0 else 0.0)
    lam = 0.0
    if lmax > 0 and np.any(d):
        q = float(d @ np.linalg.pinv(S, rcond=1e-10, hermitian=True) @ d)
        lam = 0.0 if q <= 0 else max(0.0, 1.0 - max(0.0, eff - 2.0) / q)
    return Fbar + lam * d, {"lam": lam, "noise": tr, "eff_dim": eff, "q": q, "ss": float((d ** 2).sum())}


def shrink_eb(F, Fb):
    """normal-normal empirical Bayes with learner-specific bootstrap variances (method-of-moments tau^2)"""
    F = np.asarray(F, float); M = len(F)
    s2 = np.maximum(Fb.var(1, ddof=1), 1e-12)
    w = 1.0 / s2
    mu_fe = float((w * F).sum() / w.sum())
    Q = float((w * (F - mu_fe) ** 2).sum())
    tau2 = max(0.0, (Q - (M - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1.0 / (s2 + tau2)
    mu = float((wr * F).sum() / wr.sum())
    B = tau2 / (tau2 + s2)
    return mu + B * (F - mu), {"lam": float(B.mean()), "tau2": tau2, "B": B.tolist(),
                               "noise": float((M - 1) * s2.mean()), "ss": float(((F - F.mean()) ** 2).sum())}


def weights_bagged(Fb):
    """bootstrap-aggregated log-odds weights (no explicit shrinkage)"""
    W = logodds(Fb)
    W = W / np.clip(W.sum(0, keepdims=True), EPS, None)
    return W.mean(1)


# ========================================================================== rules
class Ctx:
    """Validation-side quantities shared by the rules; the expensive ones are computed lazily."""

    def __init__(self, Pv, yv, rng, hist=None, n_boot=N_BOOT, models=None):
        self.Pv, self.yv, self.rng, self.hist, self.n_boot = Pv, yv, rng, hist, n_boot
        self.models = models if models is not None else MO
        self.M, self.n = Pv.shape[0], Pv.shape[1]
        self.pred_v = Pv.argmax(-1)
        self.F = macro_f1_batch(yv, self.pred_v)
        self.A = accuracy_batch(yv, self.pred_v)
        self._idx = self._Fb = self._Ab = self._T = None
        self.meta = {}

    @property
    def idx(self):
        if self._idx is None:
            self._idx = self.rng.integers(0, self.n, size=(self.n_boot, self.n))
        return self._idx

    @property
    def Fb(self):
        if self._Fb is None:
            self._Fb = macro_f1_batch(self.yv, self.pred_v, self.idx)
        return self._Fb

    @property
    def Ab(self):
        if self._Ab is None:
            self._Ab = accuracy_batch(self.yv, self.pred_v, self.idx)
        return self._Ab

    @property
    def T(self):
        if self._T is None:
            self._T = np.array([fit_temperature(self.Pv[m], self.yv) for m in range(self.M)])
        return self._T


def _out(P):
    return P.argmax(-1), P[:, 1]


def _stack_features(P):
    p = np.clip(P[:, :, 1], 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).T


def r_best_single(c):
    m = int(np.argmax(c.F)); c.meta["best_single"] = m
    return lambda P: _out(P[m])


def r_majority(c):
    def f(P):
        pred = majority(P, fuse(P))
        return pred, (P.argmax(-1) == 1).mean(0)
    return f


def r_uniform(c):
    return lambda P: _out(fuse(P))


def r_linear(c):
    w = c.F.copy()
    return lambda P: _out(fuse(P, w))


def r_softmax(c):
    b = max(BETAS, key=lambda bb: macro_f1(c.yv, fuse(c.Pv, softmax_w(c.F, bb)).argmax(-1)))
    w = softmax_w(c.F, b); c.meta["beta"] = b
    return lambda P: _out(fuse(P, w))


def r_logodds(c):
    w = logodds(c.F)
    return lambda P: _out(fuse(P, w))


def r_maxprob(c):
    return lambda P: _out(fuse(P, None, conf_maxprob(P)))


def r_entropy(c):
    return lambda P: _out(fuse(P, None, conf_entropy(P)))


def r_lin_mp(c):
    w = c.F.copy()
    return lambda P: _out(fuse(P, w, conf_maxprob(P)))


def r_lin_ent(c):
    w = c.F.copy()
    return lambda P: _out(fuse(P, w, conf_entropy(P)))


def r_lo_mp(c):
    w = logodds(c.F)
    return lambda P: _out(fuse(P, w, conf_maxprob(P)))


def r_lo_ent(c):
    w = logodds(c.F)
    return lambda P: _out(fuse(P, w, conf_entropy(P)))


def _sawsv(c, key, Fs, info):
    c.meta[key] = dict(info, w=(logodds(Fs) / logodds(Fs).sum()).tolist())
    w = logodds(Fs)
    return lambda P: _out(fuse(P, w, conf_entropy(P)))


def r_sawsv_binom(c):
    Fs, info = shrink_js_binomial(c.F, c.n)
    return _sawsv(c, "sawsv_binom", Fs, info)


def r_sawsv_boot(c):
    Fs, info = shrink_js_boot_paired(c.F, c.Fb)
    return _sawsv(c, "sawsv_boot", Fs, info)


def r_sawsv_bootm(c):
    Fs, info = shrink_js_boot_marginal(c.F, c.Fb)
    return _sawsv(c, "sawsv_bootm", Fs, info)


def r_sawsv_bock(c):
    Fs, info = shrink_bock(c.F, c.Fb)
    return _sawsv(c, "sawsv_bock", Fs, info)


def r_sawsv_eb(c):
    Fs, info = shrink_eb(c.F, c.Fb)
    return _sawsv(c, "sawsv_eb", Fs, info)


def r_sawsv_bag(c):
    w = weights_bagged(c.Fb); c.meta["sawsv_bag"] = {"w": (w / w.sum()).tolist()}
    return lambda P: _out(fuse(P, w, conf_entropy(P)))


def r_sawsv_noconf(c):
    Fs, _ = shrink_js_boot_paired(c.F, c.Fb)
    w = logodds(Fs)
    return lambda P: _out(fuse(P, w))


def r_sawsv_ts(c):
    """SA-WSV on temperature-scaled posteriors: TS changes both p_m and c_m, never the argmax"""
    Fs, _ = shrink_js_boot_paired(c.F, c.Fb)
    w, T = logodds(Fs), c.T.copy()

    def f(P):
        Q = apply_temperature(P, T)
        return _out(fuse(Q, w, conf_entropy(Q)))
    return f


def r_sawsv_rank(c):
    """SA-WSV with each learner's confidence replaced by its percentile among its own validation
    confidences, which removes learner-specific confidence scales"""
    Fs, _ = shrink_js_boot_paired(c.F, c.Fb)
    w = logodds(Fs)
    cv = np.sort(conf_entropy(c.Pv), axis=1)

    def f(P):
        ct = conf_entropy(P)
        cr = np.stack([(np.searchsorted(cv[m], ct[m], side="right") + 1.0) / (cv.shape[1] + 2.0)
                       for m in range(len(cv))])
        return _out(fuse(P, w, cr))
    return f


def r_uniform_ts(c):
    T = c.T.copy(); c.meta["T"] = T.tolist()
    return lambda P: _out(fuse(apply_temperature(P, T)))


def r_product(c):
    def f(P):
        Q = _softmax(np.log(np.clip(P, 1e-12, 1.0)).mean(0))
        return _out(Q)
    return f


def r_late_acc(c):
    """Wu et al. (2026): weights = mean validation accuracy over the final training epochs; with three
    epochs here, the mean over all epochs. Needs the per-epoch history of the full validation split."""
    if c.hist is None or any(c.hist.get(m) is None for m in c.models):
        return None
    w = np.array([np.mean([h["acc"] for h in c.hist[m]]) for m in c.models])
    return lambda P: _out(fuse(P, w))


def r_stacking(c):
    X, y = _stack_features(c.Pv), c.yv
    Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
    k = int(min(5, np.bincount(y, minlength=2).min()))
    C = 1.0
    if k >= 2:
        skf = StratifiedKFold(k, shuffle=True, random_state=0)
        losses = []
        for Cc in Cs:
            tot = 0.0
            for tr, te in skf.split(X, y):
                lr = LogisticRegression(C=Cc, max_iter=2000).fit(X[tr], y[tr])
                pp = np.clip(lr.predict_proba(X[te])[:, 1], 1e-12, 1 - 1e-12)
                tot += float(-(y[te] * np.log(pp) + (1 - y[te]) * np.log(1 - pp)).sum())
            losses.append(tot)
        C = Cs[int(np.argmin(losses))]
    if np.bincount(y, minlength=2).min() == 0:
        return r_uniform(c)
    lr = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    c.meta["stack_C"] = C

    def f(P):
        pp = lr.predict_proba(_stack_features(P))[:, 1]
        return (pp > 0.5).astype(int), pp
    return f


def convex_nll_weights(Py):
    """weights on the simplex minimising -mean log(w @ Py), Py (M, n) = probability of the true class.
    SLSQP from the uniform start; the result is guarded by the uniform point and the vertices, so a
    solver stall can never return a worse convex combination than those."""
    M = Py.shape[0]

    def f(w):
        mix = np.clip(w @ Py, 1e-12, None)
        return float(-np.log(mix).mean()), -(Py / mix).mean(1)

    r = minimize(f, np.full(M, 1.0 / M), jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * M,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
                 options={"ftol": 1e-12, "maxiter": 500})
    w = np.clip(r.x, 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(M, 1.0 / M)
    return min([w, np.full(M, 1.0 / M)] + list(np.eye(M)), key=lambda v: f(v)[0])


def r_convex_nll(c):
    """convex weights minimising validation log-loss of the averaged posterior (Super-Learner style)"""
    w = convex_nll_weights(c.Pv[:, np.arange(c.n), c.yv]); c.meta["convex_w"] = w.tolist()
    return lambda P: _out(fuse(P, w))


def r_ens_sel(c, K=20):
    """Caruana et al. ensemble selection: greedy forward selection with replacement on validation log-loss"""
    Py = c.Pv[:, np.arange(c.n), c.yv]
    counts, cur = np.zeros(c.M), np.zeros(c.n)
    for k in range(K):
        cand = (cur[None] * k + Py) / (k + 1)
        m = int(np.argmin(-np.log(np.clip(cand, 1e-12, None)).mean(1)))
        counts[m] += 1; cur = cand[m]
    w = counts / K; c.meta["ens_sel_w"] = w.tolist()
    return lambda P: _out(fuse(P, w))


def r_abl(stat, transform, shrink):
    """reliability-transform ablation: statistic x transform x shrinkage, always with entropy confidence"""
    def fit(c):
        R, Rb = (c.F, (lambda: c.Fb)) if stat == "f1" else (c.A, (lambda: c.Ab))
        Rs = shrink_js_boot_paired(R, Rb())[0] if shrink else R
        w = logodds(Rs) if transform == "logodds" else np.asarray(Rs, float)
        return lambda P: _out(fuse(P, w, conf_entropy(P)))
    return fit


ORIGINAL_RULES = OrderedDict([
    ("Best single (val-selected)", r_best_single),
    ("Majority voting", r_majority),
    ("Uniform soft voting", r_uniform),
    ("Linear-F1 weighting", r_linear),
    ("Softmax-F1 weighting", r_softmax),
    ("Log-odds weighting", r_logodds),
    ("Max-prob confidence", r_maxprob),
    ("Entropy confidence", r_entropy),
    ("Linear-F1 x max-prob", r_lin_mp),
    ("Log-odds x max-prob", r_lo_mp),
    ("Log-odds x entropy", r_lo_ent),
    ("SA-WSV (binomial)", r_sawsv_binom),
])
NEW_BASELINES = OrderedDict([
    ("Product rule", r_product),
    ("Late-epoch accuracy weighting", r_late_acc),
    ("Uniform + TS", r_uniform_ts),
    ("Stacking (LR)", r_stacking),
    ("Convex NLL weights", r_convex_nll),
    ("Ensemble selection", r_ens_sel),
])
SAWSV_VARIANTS = OrderedDict([
    ("SA-WSV (bootstrap)", r_sawsv_boot),
    ("SA-WSV (bootstrap, marginal)", r_sawsv_bootm),
    ("SA-WSV (Bock)", r_sawsv_bock),
    ("SA-WSV (EB)", r_sawsv_eb),
    ("SA-WSV (bagged)", r_sawsv_bag),
    ("SA-WSV + TS", r_sawsv_ts),
    ("SA-WSV, rank confidence", r_sawsv_rank),
    ("SA-WSV w/o confidence", r_sawsv_noconf),
])
ABLATION_RULES = OrderedDict([
    ("Abl: linear-F1, raw", r_lin_ent),
    ("Abl: linear-F1, shrunk", r_abl("f1", "linear", True)),
    ("Abl: log-odds-F1, raw", r_lo_ent),
    ("Abl: log-odds-F1, shrunk", r_sawsv_boot),
    ("Abl: linear-acc, raw", r_abl("acc", "linear", False)),
    ("Abl: linear-acc, shrunk", r_abl("acc", "linear", True)),
    ("Abl: log-odds-acc, raw", r_abl("acc", "logodds", False)),
    ("Abl: log-odds-acc, shrunk", r_abl("acc", "logodds", True)),
])
ALL_RULES = OrderedDict(list(ORIGINAL_RULES.items()) + list(NEW_BASELINES.items()) + list(SAWSV_VARIANTS.items()))
PRIMARY = "SA-WSV (bootstrap)"
BASELINE = "Uniform soft voting"


def fit_rules(Pv, yv, rng, hist=None, rules=None, n_boot=N_BOOT, models=None, timings=None):
    """Fit every rule on validation data. Returns (name -> callable(P) -> (pred, p_pos), ctx).
    Rules that cannot be fitted in this setting (e.g. no training history) are omitted."""
    rules = ALL_RULES if rules is None else rules
    c = Ctx(Pv, yv, rng, hist=hist, n_boot=n_boot, models=models)
    out = OrderedDict()
    for name, fit in rules.items():
        t0 = time.perf_counter()
        f = fit(c)
        if timings is not None:
            timings[name] = time.perf_counter() - t0
        if f is not None:
            out[name] = f
    return out, c


# ================================================================== per-seed runs
def pooled_run(ds, seed, rules=None, n_boot=N_BOOT, with_singles=True, transfer=False):
    """Fit every rule per fold on that fold's validation split and score the fold's test portion.

    Returns dict with y (pooled out-of-fold labels, in fold order), fold (fold id per instance),
    pred/prob {rule: array}, per_fold {rule: [metrics per fold]}, meta [ctx.meta per fold], and,
    if transfer=True, xfer {target: {rule: [pred per fold]}} using the same fitted rules."""
    ys, folds, preds, probs, per_fold, metas = [], [], {}, {}, {}, []
    xfer = {}
    for fold in range(N_FOLDS):
        d = load_fold(ds, fold, seed)
        fitted, ctx = fit_rules(d["Pv"], d["yv"], fold_rng(seed, ds, fold), hist=d["hist"], rules=rules, n_boot=n_boot)
        res = OrderedDict()
        if with_singles:
            for m, name in enumerate(MO):
                res[MLAB[name]] = (d["Pt"][m].argmax(-1), d["Pt"][m][:, 1])
        for name, f in fitted.items():
            res[name] = f(d["Pt"])
        for name, (p, q) in res.items():
            preds.setdefault(name, []).append(p); probs.setdefault(name, []).append(q)
            per_fold.setdefault(name, []).append(full_metrics(d["yt"], p))
        ys.append(d["yt"]); folds.append(np.full(len(d["yt"]), fold)); metas.append(ctx.meta)
        if transfer:
            for tgt, Px in d["xfer"].items():
                r = xfer.setdefault(tgt, OrderedDict())
                if with_singles:
                    for m, name in enumerate(MO):
                        r.setdefault(MLAB[name], []).append(Px[m].argmax(-1))
                for name, f in fitted.items():
                    r.setdefault(name, []).append(f(Px)[0])
    return {"y": np.concatenate(ys), "fold": np.concatenate(folds),
            "pred": {k: np.concatenate(v) for k, v in preds.items()},
            "prob": {k: np.concatenate(v) for k, v in probs.items()},
            "per_fold": per_fold, "meta": metas, "xfer": xfer}


def summarise_folds(per_fold, metric="f1"):
    v = np.array([r[metric] for r in per_fold])
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "folds": v.tolist()}


# ==================================================================== statistics
def mcnemar(y, p1, p2):
    r1, r2 = p1 == y, p2 == y
    b, c = int((r1 & ~r2).sum()), int((~r1 & r2).sum())
    return b, c, (1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue))


def holm(pv):
    it = sorted(pv.items(), key=lambda kv: kv[1])
    m, adj, prev = len(it), {}, 0.0
    for i, (k, p) in enumerate(it):
        v = min(1.0, max(prev, (m - i) * p)); adj[k] = v; prev = v
    return adj


def stratified_boot_index(strata, B, rng):
    """(B, n) bootstrap indices that resample within each stratum (fold), preserving fold sizes"""
    strata = np.asarray(strata)
    idx = np.empty((B, len(strata)), dtype=np.int32)
    pos = 0
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        idx[:, pos:pos + len(members)] = members[rng.integers(0, len(members), size=(B, len(members)))]
        pos += len(members)
    return idx


def two_stage_bootstrap(y, A, Bp, strata, n_boot=5000, seed=0):
    """Difference in macro-F1 between rule A and rule B, averaged over fine-tuning seeds.

    A, Bp: (S, n) pooled out-of-fold predictions for S seeds on the same n instances. Each replicate
    resamples seeds with replacement (fine-tuning variance) and instances within folds (test-sample
    variance) jointly for all seeds, which keeps the pairing between rules and between seeds.
    Returns point estimate, percentile CIs (95% and 90%) and a two-sided bootstrap p-value."""
    rng = np.random.default_rng(seed)
    S = A.shape[0]
    point = float(np.mean([macro_f1(y, A[s]) - macro_f1(y, Bp[s]) for s in range(S)]))
    idx = stratified_boot_index(strata, n_boot, rng)
    sidx = rng.integers(0, S, size=(n_boot, S))
    fa = macro_f1_batch(y, A, idx)          # (S, B)
    fb = macro_f1_batch(y, Bp, idx)
    diff = fa - fb
    d = diff[sidx, np.arange(n_boot)[:, None]].mean(1)   # (B,)
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"diff": point, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "ci90": [float(np.percentile(d, 5)), float(np.percentile(d, 95))], "p_boot": p}


def two_stage_bootstrap_all(y, preds, strata, n_boot=5000, seed=0):
    """Two-stage bootstrap for many rules at once with common random numbers.

    preds: {rule: (S, n)} pooled out-of-fold predictions for S seeds. Returns ({rule: point estimate
    of the seed-averaged macro-F1}, {rule: (n_boot,) replicates}). All rules share the same instance
    and seed resamples, so reps[a] - reps[b] is a valid replicate of the difference a - b."""
    rng = np.random.default_rng(seed)
    names = list(preds)
    S = preds[names[0]].shape[0]
    idx = stratified_boot_index(strata, n_boot, rng)
    sidx = rng.integers(0, S, size=(n_boot, S))
    stacked = np.concatenate([np.asarray(preds[k]) for k in names])
    Fb = macro_f1_batch(y, stacked, idx).reshape(len(names), S, n_boot)
    point = {k: float(np.mean([macro_f1(y, preds[k][s]) for s in range(S)])) for k in names}
    reps = {k: Fb[i][sidx, np.arange(n_boot)[:, None]].mean(1) for i, k in enumerate(names)}
    return point, reps


def compare_from_reps(point, reps, a, b):
    """difference a - b with percentile CIs and a two-sided bootstrap p-value"""
    d = reps[a] - reps[b]
    return {"diff": point[a] - point[b],
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "ci90": [float(np.percentile(d, 5)), float(np.percentile(d, 95))],
            "p_boot": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def q_statistic(ok):
    """mean pairwise Q-statistic and disagreement over correctness indicators ok (M, n)"""
    Q, D = [], []
    M = ok.shape[0]
    for i in range(M):
        for j in range(i + 1, M):
            n11 = np.count_nonzero(ok[i] & ok[j]); n00 = np.count_nonzero(~ok[i] & ~ok[j])
            n10 = np.count_nonzero(ok[i] & ~ok[j]); n01 = np.count_nonzero(~ok[i] & ok[j])
            Q.append((n11 * n00 - n01 * n10) / max(n11 * n00 + n01 * n10, 1))
            D.append(np.mean(ok[i] != ok[j]))
    return float(np.mean(Q)), float(np.mean(D))


def effective_voters(ok):
    """error-correlation matrix R of the correctness indicators and the participation ratio
    (tr R)^2 / tr(R^2): 1 for perfectly correlated errors, M for independent ones"""
    X = ok.astype(float)
    sd = X.std(1)
    keep = sd > 0
    R = np.corrcoef(X[keep])
    m_eff = float(np.trace(R) ** 2 / np.sum(R * R))
    off = R[~np.eye(len(R), dtype=bool)]
    return {"rho_mean": float(off.mean()), "m_eff": m_eff, "M": int(keep.sum())}


if __name__ == "__main__":
    # regression check: the original twelve rules on seed 42 must reproduce final.json exactly
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    F = json.load(open(_p("experiments", "results", "final.json")))["per_dataset"]
    bad = 0
    for ds in IN_DOMAIN:
        R = pooled_run(ds, 42, rules=ORIGINAL_RULES)
        for name in ORIGINAL_RULES:
            key = "SA-WSV (proposed)" if name == "SA-WSV (binomial)" else name
            mine = summarise_folds(R["per_fold"][name])
            ref = F[ds]["summary"][key]["f1"]
            ok = abs(mine["mean"] - ref["mean"]) < 1e-12 and abs(mine["std"] - ref["std"]) < 1e-12
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {DS_LAB[ds]:10s} {name:28s} {100*mine['mean']:.4f} vs {100*ref['mean']:.4f}")
        lam = [m["sawsv_binom"]["lam"] for m in R["meta"]]
        ref_lam = [i["lambda"] for i in F[ds]["info"]]
        assert np.allclose(lam, ref_lam), (lam, ref_lam)
    print("REGRESSION", "PASSED" if bad == 0 else f"FAILED ({bad})")
