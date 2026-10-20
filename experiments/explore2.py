"""Exploration round 2: calibration-aware, robust, and stacked fusion.

Motivation from round 1: the five learners are highly correlated (Q = 0.86-0.94) and
differ mainly in how *confidently* they are wrong. Probability averaging therefore gives
an over-confident, miscalibrated learner an unfairly loud vote. Round 2 tests rules that
attack that directly. Every hyper-parameter is fitted on validation posteriors only.
"""
import os, glob, math, json, itertools
import numpy as np
from scipy.optimize import minimize_scalar

MODEL_ORDER = ["indobert-p1", "indobert-cahya", "indoroberta-cahya", "xlmr-base", "mbert-cased"]
EPS = 1e-9


def macro_f1(y, p):
    f = 0.0
    for c in (0, 1):
        tp = np.count_nonzero((p == c) & (y == c))
        fp = np.count_nonzero((p == c) & (y != c))
        fn = np.count_nonzero((p != c) & (y == c))
        d = 2 * tp + fp + fn
        f += (2.0 * tp / d) if d else 0.0
    return f / 2.0


def nll(y, P):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0)).mean())


def ece(y, P, bins=10):
    conf, pred = P.max(-1), P.argmax(-1)
    ok = (pred == y).astype(float)
    e, n = 0.0, len(y)
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def to_logit(P):
    return np.log(np.clip(P, EPS, 1.0))


def temp_scale(P, T):
    z = to_logit(P) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_T(Pv, yv):
    """temperature minimizing validation NLL (Guo et al. 2017)"""
    r = minimize_scalar(lambda t: nll(yv, temp_scale(Pv, max(t, 1e-2))), bounds=(0.05, 20.0), method="bounded")
    return float(max(r.x, 1e-2))


def conf_entropy(P):
    Pc = np.clip(P, 1e-6, 1.0)
    return 1.0 - (-(Pc * np.log(Pc)).sum(-1)) / math.log(P.shape[-1])


def fuse(P, w=None, c=None, gamma=0.0):
    M, n, _ = P.shape
    W = np.ones((M, n)) if w is None else np.repeat(np.asarray(w, float)[:, None], n, 1)
    if gamma > 0:
        W = W * np.power(np.clip(c, 1e-9, 1.0), gamma)
    W = W / np.clip(W.sum(0, keepdims=True), 1e-9, None)
    return (W[:, :, None] * P).sum(0)


def fuse_logit(P, w=None):
    """geometric (log-opinion-pool) fusion"""
    M, n, _ = P.shape
    w = np.ones(M) if w is None else np.asarray(w, float)
    w = w / w.sum()
    z = (w[:, None, None] * to_logit(P)).sum(0)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def softmax_w(f, beta):
    z = beta * np.asarray(f, float); z -= z.max(); e = np.exp(z); return e / e.sum()


def logodds(f):
    f = np.clip(np.asarray(f, float), 0.5 + 1e-6, 1 - 1e-6); return np.log(f / (1 - f))


def load(root, ds):
    out = []
    for d in sorted(glob.glob(os.path.join(root, ds, "fold*"))):
        Pv, Pt, ok = [], [], True
        yv = yt = None
        for m in MODEL_ORDER:
            f = os.path.join(d, f"{m}.npz")
            if not os.path.exists(f):
                ok = False; break
            z = np.load(f)
            Pv.append(z["val_probs"]); Pt.append(z["test_probs"]); yv, yt = z["y_val"], z["y_test"]
        if ok:
            out.append((np.stack(Pv), yv, np.stack(Pt), yt))
    return out


BETAS = [0, 2, 5, 10, 20, 50, 100]
GAMMAS = [0, 0.5, 1, 2, 3, 5, 8]


def evaluate(folds):
    res, pooled, diag = {}, {}, []

    def add(name, yt, pred):
        res.setdefault(name, []).append(macro_f1(yt, pred))
        pooled.setdefault(name, {"y": [], "p": []})
        pooled[name]["y"].append(yt); pooled[name]["p"].append(pred)

    for Pv, yv, Pt, yt in folds:
        M = len(MODEL_ORDER)
        fv = np.array([macro_f1(yv, Pv[m].argmax(-1)) for m in range(M)])
        # ---- per-model temperature fitted on validation
        T = np.array([fit_T(Pv[m], yv) for m in range(M)])
        Qv = np.stack([temp_scale(Pv[m], T[m]) for m in range(M)])
        Qt = np.stack([temp_scale(Pt[m], T[m]) for m in range(M)])
        diag.append({"T": T.tolist(),
                     "ece_raw": [ece(yt, Pt[m]) for m in range(M)],
                     "ece_cal": [ece(yt, Qt[m]) for m in range(M)],
                     "nll_raw": [nll(yt, Pt[m]) for m in range(M)],
                     "nll_cal": [nll(yt, Qt[m]) for m in range(M)],
                     "val_f1": fv.tolist()})
        cvr, ctr = conf_entropy(Pv), conf_entropy(Pt)
        cvc, ctc = conf_entropy(Qv), conf_entropy(Qt)

        add("uniform-soft (raw)", yt, fuse(Pt).argmax(-1))
        add("linearF1 (raw)", yt, fuse(Pt, fv).argmax(-1))
        add("CAL uniform-soft", yt, fuse(Qt).argmax(-1))
        add("CAL linearF1", yt, fuse(Qt, fv).argmax(-1))
        add("CAL logodds", yt, fuse(Qt, logodds(fv)).argmax(-1))
        add("CAL logodds x conf^1", yt, fuse(Qt, logodds(fv), ctc, 1).argmax(-1))

        g = max(GAMMAS, key=lambda gg: macro_f1(yv, fuse(Qv, logodds(fv), cvc, gg).argmax(-1)))
        add("CAL logodds x conf^gamma*", yt, fuse(Qt, logodds(fv), ctc, g).argmax(-1))
        b, g2 = max(itertools.product(BETAS, GAMMAS),
                    key=lambda t: (macro_f1(yv, fuse(Qv, softmax_w(fv, t[0]), cvc, t[1]).argmax(-1)), -t[0], -t[1]))
        add("CAL softmaxF1(b*) x conf^g*", yt, fuse(Qt, softmax_w(fv, b), ctc, g2).argmax(-1))

        add("logit-pool uniform (raw)", yt, fuse_logit(Pt).argmax(-1))
        add("CAL logit-pool uniform", yt, fuse_logit(Qt).argmax(-1))
        add("CAL logit-pool linearF1", yt, fuse_logit(Qt, fv).argmax(-1))

        # robust fusion: drop the single most extreme posterior per instance, average the rest
        s = np.sort(Pt[:, :, 1], axis=0)
        add("trimmed-mean (raw)", yt, (s[1:-1].mean(0) > 0.5).astype(int))
        s2 = np.sort(Qt[:, :, 1], axis=0)
        add("CAL trimmed-mean", yt, (s2[1:-1].mean(0) > 0.5).astype(int))
        add("median (raw)", yt, (np.median(Pt[:, :, 1], axis=0) > 0.5).astype(int))
        add("CAL median", yt, (np.median(Qt[:, :, 1], axis=0) > 0.5).astype(int))

        # rank averaging (scale-free)
        rk = np.stack([np.argsort(np.argsort(Pt[m, :, 1])) / (Pt.shape[1] - 1) for m in range(M)])
        add("rank-average", yt, (rk.mean(0) > 0.5).astype(int))

        # stacking: logistic regression on validation posteriors
        from sklearn.linear_model import LogisticRegression
        Xv = np.concatenate([Pv[m][:, 1:2] for m in range(M)], 1)
        Xt = np.concatenate([Pt[m][:, 1:2] for m in range(M)], 1)
        for C in [1.0]:
            lr = LogisticRegression(C=C, max_iter=2000).fit(Xv, yv)
            add("stacking-LR", yt, lr.predict(Xt))
        Xv2 = np.concatenate([to_logit(Pv[m])[:, 1:2] - to_logit(Pv[m])[:, 0:1] for m in range(M)], 1)
        Xt2 = np.concatenate([to_logit(Pt[m])[:, 1:2] - to_logit(Pt[m])[:, 0:1] for m in range(M)], 1)
        lr2 = LogisticRegression(C=1.0, max_iter=2000).fit(Xv2, yv)
        add("stacking-LR (logit feats)", yt, lr2.predict(Xt2))

        # oracle references
        bo, bos = None, -1
        for r in range(1, M + 1):
            for sub in itertools.combinations(range(M), r):
                sc = macro_f1(yt, fuse(Pt[list(sub)]).argmax(-1))
                if sc > bos: bos, bo = sc, sub
        add("ORACLE subset (TEST)", yt, fuse(Pt[list(bo)]).argmax(-1))

    final = {}
    for k, v in res.items():
        y = np.concatenate(pooled[k]["y"]); p = np.concatenate(pooled[k]["p"])
        final[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "pooled": macro_f1(y, p)}
    return final, diag


if __name__ == "__main__":
    out = {}
    for ds in ["evprimary-formal", "evpublic", "evprimary-informal"]:
        folds = load("experiments/probs", ds)
        if not folds:
            continue
        r, diag = evaluate(folds)
        out[ds] = {"rules": r, "diag": diag}
        base = r["uniform-soft (raw)"]["pooled"]
        print(f"\n===== {ds} =====")
        for k, v in sorted(r.items(), key=lambda kv: -kv[1]["pooled"]):
            print(f"{k:32s} mean={100 * v['mean']:6.2f}+-{100 * v['std']:4.2f} pooled={100 * v['pooled']:6.2f} vs-uniform={100 * (v['pooled'] - base):+5.2f}")
        T = np.array([d["T"] for d in diag])
        print("temperatures (mean per model):", dict(zip(MODEL_ORDER, np.round(T.mean(0), 3))))
        print("ECE raw ->cal :", np.round(np.mean([d["ece_raw"] for d in diag], 0), 4), "->", np.round(np.mean([d["ece_cal"] for d in diag], 0), 4))
        print("NLL raw ->cal :", np.round(np.mean([d["nll_raw"] for d in diag], 0), 4), "->", np.round(np.mean([d["nll_cal"] for d in diag], 0), 4))
    json.dump(out, open("experiments/results/explore2.json", "w"), indent=1)
