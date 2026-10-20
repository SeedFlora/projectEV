"""Calibration and confidence analysis for the revision (reviewer 2, third comment).

SA-WSV multiplies each learner's reliability weight by its entropy confidence
c_m(x) = 1 - H(p_m(x))/ln 2, i.e. it compares softmax confidences directly across five heterogeneous
PLMs that may be miscalibrated on different scales. This script measures that miscalibration, what
temperature scaling (TS) does and does not change, and what calibrated inputs change in the fusion,
in domain and under corpus shift.

Design decisions
* Units are (corpus, fine-tuning seed, fold) for every seed complete when the script starts
  (L.complete_seeds); the seeds used are recorded in the JSON, so the script needs no edits once all
  five seeds exist. Per-unit summaries are mean and SD (ddof=1) over seeds x folds; the "mean of
  learners" averages the learners within each unit first, so its SD is over units as well (the SD
  between the learner means is stored as sd_between_learners).
* One temperature per learner and unit, fitted with L.fit_temperature (validation NLL) on that fold's
  validation split only, and scored on the fold's test portion or on a transfer corpus. The only
  quantities fitted on the scored labels are the "*_oracle" references (T refitted to the NLL of the
  scored instances), which show how much of the NLL-optimal correction the validation-fitted T
  recovers; they are not ECE lower bounds and feed nothing else. The validation split also selected
  the checkpoint (best validation F1 of three epochs), so validation accuracy is optimistic and the
  validation T tends to under-correct; T_oracle / T and the post-TS gap measure this.
* Scale differences between learners: max/min of the learner-mean temperatures, with a noise-only
  reference from shuffling the learner labels within each unit. The mean within-unit max/min
  ("T_ratio") is kept but includes the estimation noise of T on 111/288 validation instances; the
  range of the learner-mean c_m is likewise stored next to the mean within-unit range.
* Calibration is top-label: ECE with 15 equal-width bins, adaptive ECE with 15 equal-mass bins, the
  Brier score of P(positive), NLL, and the signed gap mean confidence - accuracy (> 0: overconfident).
  The plug-in ECE is biased upward, the more so the smaller n, so ECEs are compared only at equal n:
  per test fold (n = 277/719) within the per-unit table only; pooled out of fold per seed (n =
  1382/3592, the n of the fused posteriors) for every in-domain comparison (raw vs TS, learner vs
  fusion); in domain vs corpus shift at n = min(pooled source n, target n), the larger side averaged
  over N_SUB subsamples without replacement ("*_m" fields). Between corpora the pooled n still differs
  (1382 vs 3592), so there the floors, the debiased ECE, NLL and Brier carry the comparison. Next to
  these ECEs: the floor, the mean ECE of a perfectly calibrated predictor with the same confidences and
  n (correctness drawn as Bernoulli(confidence)), and in domain the debiased L2-ECE of Kumar et al.
  (2019), which removes the binomial variance of each bin: nearly n-free once the miscalibration
  clearly exceeds the noise, but near zero the clipping and square root leave a small upward bias (in a
  simulation of a calibrated predictor about 0.009 at n = 1382 and 0.005 at n = 3592, where the
  plug-in ECE was 0.020 and 0.012).
* Inference on calibration uses NLL and Brier differences (smooth proper scores). ECE differences are
  descriptive: bootstrap replicates of the plug-in ECE are inflated more for the better calibrated
  rule, which pulls percentile CIs of a difference toward zero (replicate bias and basic CI stored).
* For C = 2, TS maps the logit margin log(p1/p0) to margin/T: it keeps every argmax and the ranking of
  a learner's confidences, so it cannot change which predictions are correct or any within-learner
  AUROC; it only rescales confidence across learners, which is what the entropy term compares. Both
  facts are checked numerically (the AUROC of the TS-invariant margin is reported alongside).
* Contested instances are those on which the learners' predicted labels differ, found without labels.
  The stored posteriors come from bf16 logits and are quantised, so learners can tie exactly for the
  highest confidence; ties get fractional credit (expected accuracy under uniform tie-breaking).
* Fusion rules are fitted on validation only, by L.pooled_run, and scored on pooled out-of-fold
  predictions; the fused posterior is [1 - P(pos), P(pos)]. Paired differences use the two-stage
  bootstrap of rev_lib (seeds and within-fold instances resampled jointly, common random numbers
  across rules); the ECE, NLL and Brier replicates reuse exactly the same resamples. Primary family,
  per corpus, on pooled macro-F1: SA-WSV + TS, SA-WSV rank confidence and SA-WSV w/o confidence, each
  vs SA-WSV (bootstrap), Holm over the three (and over all corpora, p_holm_all_corpora). The first and
  the last also get TOST equivalence at +/-0.5 and +/-1.0 points (90% CI inside the margin; a claim
  for every corpus is an intersection-union test and needs no correction). The other F1 tests are
  exploratory (Holm within corpus), the NLL and Brier tests are Holm within corpus and metric.
* Under corpus shift unit() refits the same rules with pooled_run's generator (its test-fold outputs
  are asserted equal to pooled_run's) and scores every transfer corpus of a unit; units there share
  the target instances, so their SD reflects fine-tuning and fold, not sampling.
* Extra rules defined here: entropy confidence on TS inputs, and post-hoc TS of the fused posterior
  ("output TS", T fitted on the fused validation posterior, in-sample for rules that also fit weights
  there). Output TS is a monotone map of P(pos) that fixes 0.5, so it never changes a decision.
* All randomness is seeded (subsamples and floors by name, independent of the worker schedule), so two
  runs on the same seeds give identical JSON.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one BLAS thread per worker: bounded CPU use, reproducible sums
import sys, json, time, zlib, argparse
from collections import OrderedDict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rev_lib as L

OUT_JSON = os.path.join(L.ROOT, "experiments", "results", "rev", "rev_calibration.json")
OUT_FIG = os.path.join(L.ROOT, "paper", "fig_reliability.pdf")
N_BINS = 15
N_BOOT = 5000
N_SUB = 200               # subsamples (size-matched ECE) and Bernoulli draws (ECE floor)
N_PERM = 5000             # within-unit shuffles of the learner labels (noise reference of the temperature ratio)
SALT = 31                 # generators of the subsamples and floors (the rule fits use L.fold_rng)
DELTAS = (0.005, 0.010)   # TOST equivalence margins in macro-F1: 0.5 and 1.0 points
MAX_WORKERS = 4
MIN_BIN_FIG = 20          # reliability-diagram bins with fewer pooled predictions are not drawn
REF = L.PRIMARY           # "SA-WSV (bootstrap)"
FUSION = ["Uniform soft voting", "Uniform + TS", "Entropy confidence", "SA-WSV (bootstrap)", "SA-WSV + TS",
          "SA-WSV, rank confidence", "SA-WSV w/o confidence", "SA-WSV (binomial)"]
PAIRS = [("Uniform + TS", "Uniform soft voting"), ("Entropy confidence + TS", "Entropy confidence"),
         ("SA-WSV + TS", "SA-WSV (bootstrap)"), ("SA-WSV, rank confidence", "SA-WSV (bootstrap)"),
         ("SA-WSV w/o confidence", "SA-WSV (bootstrap)"), ("SA-WSV (binomial)", "SA-WSV (bootstrap)"),
         ("SA-WSV (bootstrap)", "Uniform soft voting"), ("Entropy confidence", "Uniform soft voting"),
         ("Uniform soft voting, output TS", "Uniform soft voting"),
         ("SA-WSV (bootstrap), output TS", "SA-WSV (bootstrap)")]
PRIMARY_FAMILY = [("SA-WSV + TS", REF), ("SA-WSV, rank confidence", REF), ("SA-WSV w/o confidence", REF)]
EQUIV_PAIRS = [("SA-WSV + TS", REF), ("SA-WSV w/o confidence", REF)]
SHORT = {"indobert-p1": "IB-p1", "indobert-cahya": "IB-1.5G", "indoroberta-cahya": "IR-1.5G",
         "xlmr-base": "XLM-R", "mbert-cased": "mBERT"}
COL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]     # Okabe-Ito
MK = ["o", "s", "^", "D", "v"]
LS = ["-", "-", "-", "--", "--"]                                    # monolingual solid, multilingual dashed
STYLE = {"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
         "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]}
LEARNER_FIELDS = ["T", "T_oracle", "acc", "ece_raw", "ece_ts", "ece_ts_oracle", "aece_raw", "aece_ts", "brier_raw",
                  "brier_ts", "nll_raw", "nll_ts", "gap_raw", "gap_ts", "auroc_raw", "auroc_ts", "auroc_margin",
                  "conf_raw", "conf_ts", "conf_ok_raw", "conf_wrong_raw", "conf_ok_ts", "conf_wrong_ts",
                  "share_raw", "share_ts", "acc_contested"]
CONTESTED_FIELDS = ["frac_contested", "T_ratio", "conf_range_raw", "conf_range_ts", "count_frac", "correct_mass_raw",
                    "correct_mass_ts", "most_conf_acc_raw", "most_conf_acc_ts", "frac_top_tied_raw", "frac_top_tied_ts",
                    "frac_top_changed", "acc_uniform_raw", "acc_uniform_ts", "acc_entropy_raw", "acc_entropy_ts",
                    "acc_majority"]
XFER_FIELDS = ["acc", "ece_raw", "ece_ts", "ece_ts_target_oracle", "aece_raw", "aece_ts", "nll_raw", "nll_ts",
               "brier_raw", "brier_ts", "gap_raw", "gap_ts", "auroc_raw", "conf_raw", "conf_ts", "T", "T_target_oracle",
               "ece_raw_m", "ece_ts_m", "ece_oracle_m", "floor_raw_m", "floor_ts_m"]
IN_FIELDS = ["acc", "ece_raw", "ece_ts", "aece_raw", "aece_ts", "nll_raw", "nll_ts", "gap_raw", "gap_ts", "auroc_raw"]
IN_M_FIELDS = ["ece_raw_m", "ece_ts_m", "floor_raw_m", "floor_ts_m"]
POOLED_FIELDS = ["acc", "ece_raw", "ece_ts", "floor_raw", "floor_ts", "dece_raw", "dece_ts", "aece_raw", "aece_ts",
                 "nll_raw", "nll_ts", "brier_raw", "brier_ts", "gap_raw", "gap_ts"]
CORPUS_FIELDS = [f for f in LEARNER_FIELDS if not f.startswith("share") and f != "acc_contested"]
FUSED_FIELDS = ["f1", "ece", "nll", "brier", "gap"]
XFER_SHOW = ["Uniform soft voting", "Uniform + TS", "Entropy confidence", "Entropy confidence + TS", "SA-WSV (bootstrap)",
             "SA-WSV + TS", "SA-WSV, rank confidence", "SA-WSV w/o confidence", "SA-WSV (bootstrap), output TS"]


# ======================================================================= helpers
def two(p):
    return np.stack([1.0 - p, p], -1)


def margin(P):
    """|log p1 - log p0|; TS divides it by T, so its ranking is TS-invariant by construction"""
    z = np.log(np.clip(P, 1e-12, 1.0))
    return np.abs(z[..., 1] - z[..., 0])


def calib(y, P, tag):
    return {f"ece_{tag}": L.ece(y, P, N_BINS), f"aece_{tag}": L.ece(y, P, N_BINS, adaptive=True),
            f"brier_{tag}": L.brier(y, P), f"nll_{tag}": L.nll(y, P),
            f"gap_{tag}": float(P.max(-1).mean() - (P.argmax(-1) == y).mean())}


def bins_of(conf):
    edges = np.linspace(0.0, 1.0, N_BINS + 1)     # same binning as L.ece
    return np.clip(np.digitize(conf, edges[1:-1], right=True), 0, N_BINS - 1)


def named_rng(*key):
    """generator seeded by a name, so a quantity gets the same draws wherever (and in whichever worker) it is computed"""
    return np.random.default_rng([zlib.crc32("|".join(map(str, key)).encode()), SALT])


def ece_rows(conf, ok):
    """plug-in top-label ECE (15 equal-width bins, as L.ece) of every row of conf, ok (R, m)"""
    R, m = conf.shape
    key = bins_of(conf) + N_BINS * np.arange(R)[:, None]
    return np.abs(np.bincount(key.ravel(), (ok - conf).ravel(), R * N_BINS).reshape(R, N_BINS)).sum(1) / m


def sub_index(n, m, rng):
    """N_SUB subsamples of size m without replacement, or None (the whole sample) if m >= n"""
    return None if m >= n else np.argsort(rng.random((N_SUB, n)), axis=1)[:, :m]


def ece_at(conf, ok, idx, rng):
    """(ECE, floor) on the subsamples idx (None: the whole sample). The floor is the same ECE with the
    correctness redrawn as Bernoulli(confidence): what a perfectly calibrated predictor with these
    confidences scores at this n, i.e. the part of an ECE that is sampling noise."""
    ok = np.asarray(ok, float)
    if idx is None:
        c, e = np.broadcast_to(conf, (N_SUB, len(conf))), float(ece_rows(conf[None], ok[None])[0])
    else:
        c, e = conf[idx], float(ece_rows(conf[idx], ok[idx]).mean())
    return e, float(ece_rows(c, (rng.random(c.shape) < c).astype(float)).mean())


def dece(y, P):
    """debiased top-label L2-ECE (Kumar et al., 2019): sqrt(max(0, sum_b w_b [(acc_b - conf_b)^2 -
    acc_b (1 - acc_b) / (n_b - 1)])), 15 equal-width bins; bins with a single instance contribute 0"""
    conf = P.max(-1)
    ok = (P.argmax(-1) == y).astype(float)
    b = bins_of(conf)
    nb = np.bincount(b, minlength=N_BINS).astype(float)
    k = nb > 1
    a, c = np.bincount(b, ok, N_BINS)[k] / nb[k], np.bincount(b, conf, N_BINS)[k] / nb[k]
    return float(np.sqrt(max(float((nb[k] / len(y) * ((a - c) ** 2 - a * (1 - a) / (nb[k] - 1))).sum()), 0.0)))


def calib_full(y, P, tag, rng):
    """calib() plus the ECE floor at this n and the debiased L2-ECE"""
    out = calib(y, P, tag)
    out[f"floor_{tag}"] = ece_at(P.max(-1), P.argmax(-1) == y, None, rng)[1]
    out[f"dece_{tag}"] = dece(y, P)
    return out


def top_set(c):
    return c == c.max(0, keepdims=True)


def most_confident_acc(c, ok):
    top = top_set(c)
    return float(((top & ok).sum(0) / top.sum(0)).mean())


def msd(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "n": int(len(v))}


def summarise(rows, by, fields):
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(tuple(r[b] for b in by), []).append(r)
    return OrderedDict((k, OrderedDict((f, msd([r[f] for r in g])) for f in fields)) for k, g in groups.items())


def unit_means(rows, fields, by=("ds", "seed", "fold")):
    """average the learners within each unit (or seed), so that an SD over these is over units, not learner-units"""
    g = OrderedDict()
    for r in rows:
        g.setdefault(tuple(r[b] for b in by), []).append(r)
    return [OrderedDict(list(zip(by, k)) + [(f, float(np.mean([r[f] for r in v]))) for f in fields]) for k, v in g.items()]


def with_between(corpus, learners, fields):
    """add the SD between the learner means to every field of a corpus summary"""
    for f in fields:
        corpus[f]["sd_between_learners"] = float(np.std([v[f]["mean"] for v in learners.values()], ddof=1))
    return corpus


def t_ratio(Tm, rng):
    """max/min of the learner-mean temperatures, Tm (units, learners), and its distribution when the learner
    labels are shuffled within every unit (no learner effect: estimation noise and unit effects only)"""
    def ratio(M):
        mu = M.mean(-2)
        return mu.max(-1) / mu.min(-1)
    return float(ratio(Tm)), ratio(rng.permuted(np.tile(Tm, (N_PERM, 1, 1)), axis=2))


def tost(d, ci90):
    """TOST at 5% per side from bootstrap replicates d of a difference: equivalent within +/-delta iff the
    90% percentile CI lies inside (-delta, +delta); p is the larger of the two one-sided bootstrap p-values"""
    return OrderedDict((f"{dl:.3f}", OrderedDict(equivalent=bool(-dl < ci90[0] and ci90[1] < dl),
                                                p=float(max((d <= -dl).mean(), (d >= dl).mean()))))
                       for dl in DELTAS)


def nest(flat):
    """{(a, b): v} -> {a: {b: v}}"""
    out = OrderedDict()
    for (a, b), v in flat.items():
        out.setdefault(a, OrderedDict())[b] = v
    return out


def _py(o):
    if isinstance(o, dict):
        return OrderedDict((str(k), _py(v)) for k, v in o.items())
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


_LAB = {}


def labels(name):
    if name not in _LAB:
        _LAB[name] = L.target_labels(name)
    return _LAB[name]


# ================================================================= extra rules
def _out(P):
    return P.argmax(-1), P[:, 1]


def r_entropy_ts(c):
    T = c.T.copy()

    def f(P):
        Q = L.apply_temperature(P, T)
        return _out(L.fuse(Q, None, L.conf_entropy(Q)))
    return f


def output_ts(base):
    """post-hoc TS of the fused posterior, T fitted on the fused validation posterior"""
    def fit(c):
        f = base(c)
        T = L.fit_temperature(two(f(c.Pv)[1]), c.yv)

        def g(P):
            pred, p = f(P)
            return pred, L.apply_temperature(two(p), T)[:, 1]
        return g
    return fit


RULES = OrderedDict([(k, L.ALL_RULES[k]) for k in FUSION] + [
    ("Entropy confidence + TS", r_entropy_ts),
    ("Uniform soft voting, output TS", output_ts(L.r_uniform)),
    ("SA-WSV (bootstrap), output TS", output_ts(L.r_sawsv_boot))])
EXTRA_RULES = list(RULES)[len(FUSION):]


# ======================================================== (a)-(c), (e): per unit
def unit(ds, seed, fold, n_pool):
    d = L.load_fold(ds, fold, seed)
    Pv, yv, Pt, yt = d["Pv"], d["yv"], d["Pt"], d["yt"]
    T = np.array([L.fit_temperature(Pv[m], yv) for m in range(len(L.MO))])
    fitted, _ = L.fit_rules(Pv, yv, L.fold_rng(seed, ds, fold), hist=d["hist"], rules=RULES)   # = pooled_run's fits
    test_out = OrderedDict((k, f(Pt)) for k, f in fitted.items())                              # checked in fusion()
    Q = L.apply_temperature(Pt, T)
    pred = Pt.argmax(-1)
    ok = pred == yt[None]
    c, cq = L.conf_entropy(Pt), L.conf_entropy(Q)
    K = (pred != pred[:1]).any(0)                       # contested: the learners' labels differ
    w, wq = np.clip(c[:, K], L.EPS, 1.0), np.clip(cq[:, K], L.EPS, 1.0)   # the weights L.fuse gives
    sh, shq = w / w.sum(0), wq / wq.sum(0)
    key = OrderedDict(ds=ds, seed=seed, fold=fold)
    rows = []
    for m, name in enumerate(L.MO):
        To = L.fit_temperature(Pt[m], yt)
        r = OrderedDict(key, learner=L.MLAB[name], T=T[m], T_oracle=To, T_at_bound=bool(T[m] < 0.0501 or T[m] > 19.99),
                        argmax_changed=int((Q[m].argmax(-1) != pred[m]).sum()), acc=ok[m].mean())
        r.update(calib(yt, Pt[m], "raw")); r.update(calib(yt, Q[m], "ts"))
        r["ece_ts_oracle"] = L.ece(yt, L.apply_temperature(Pt[m], To), N_BINS)
        r.update(auroc_raw=L.auroc(c[m], ok[m]), auroc_ts=L.auroc(cq[m], ok[m]), auroc_margin=L.auroc(margin(Pt[m]), ok[m]),
                 conf_raw=c[m].mean(), conf_ts=cq[m].mean(),
                 conf_ok_raw=c[m][ok[m]].mean(), conf_wrong_raw=c[m][~ok[m]].mean(),
                 conf_ok_ts=cq[m][ok[m]].mean(), conf_wrong_ts=cq[m][~ok[m]].mean(),
                 share_raw=sh[m].mean(), share_ts=shq[m].mean(), acc_contested=ok[m][K].mean())
        rows.append(r)
    yk, okK = yt[K], ok[:, K]
    top, topq = top_set(c[:, K]), top_set(cq[:, K])

    def acc_k(F):
        return float((F.argmax(-1)[K] == yk).mean())
    con = OrderedDict(key, n_test=len(yt), n_contested=int(K.sum()), frac_contested=K.mean(), T_ratio=T.max() / T.min(),
                      conf_range_raw=np.ptp(c.mean(1)), conf_range_ts=np.ptp(cq.mean(1)),
                      count_frac=okK.mean(), correct_mass_raw=(sh * okK).sum(0).mean(),
                      correct_mass_ts=(shq * okK).sum(0).mean(),
                      most_conf_acc_raw=most_confident_acc(c[:, K], okK), most_conf_acc_ts=most_confident_acc(cq[:, K], okK),
                      frac_top_tied_raw=(top.sum(0) > 1).mean(), frac_top_tied_ts=(topq.sum(0) > 1).mean(),
                      frac_top_changed=(top != topq).any(0).mean(),
                      acc_uniform_raw=acc_k(L.fuse(Pt)), acc_uniform_ts=acc_k(L.fuse(Q)),
                      acc_entropy_raw=acc_k(L.fuse(Pt, None, c)), acc_entropy_ts=acc_k(L.fuse(Q, None, cq)),
                      acc_majority=float((L.majority(Pt, L.fuse(Pt))[K] == yk).mean()))
    xrows, frows = [], []
    for tgt, Px in d["xfer"].items():
        yx = labels(tgt)
        assert Px.shape[1] == len(yx), (ds, tgt, Px.shape, len(yx))
        Qx = L.apply_temperature(Px, T)
        okx = Px.argmax(-1) == yx[None]
        nm = min(n_pool, len(yx))                   # ECE at the smaller of pooled source n and target n
        rng = named_rng("sub", ds, seed, fold, tgt)
        idx = sub_index(len(yx), nm, rng)
        for m, name in enumerate(L.MO):
            To = L.fit_temperature(Px[m], yx)
            Po = L.apply_temperature(Px[m], To)
            r = OrderedDict(key, tgt=tgt, learner=L.MLAB[name], T=T[m], T_target_oracle=To, acc=okx[m].mean(),
                            n_tgt=len(yx), n_match=nm)
            r.update(calib(yx, Px[m], "raw")); r.update(calib(yx, Qx[m], "ts"))
            r.update(ece_ts_target_oracle=L.ece(yx, Po, N_BINS),
                     auroc_raw=L.auroc(L.conf_entropy(Px[m]), okx[m]),
                     conf_raw=L.conf_entropy(Px[m]).mean(), conf_ts=L.conf_entropy(Qx[m]).mean())
            for tag, PP in (("raw", Px[m]), ("ts", Qx[m]), ("oracle", Po)):
                r[f"ece_{tag}_m"], r[f"floor_{tag}_m"] = ece_at(PP.max(-1), PP.argmax(-1) == yx, idx, rng)
            xrows.append(r)
        for rule, f in fitted.items():
            pr, p = f(Px)
            P2 = two(p)
            frows.append(OrderedDict(key, tgt=tgt, rule=rule, f1=L.macro_f1(yx, pr), ece=L.ece(yx, P2, N_BINS),
                                     nll=L.nll(yx, P2), brier=L.brier(yx, P2),
                                     gap=float(P2.max(-1).mean() - (P2.argmax(-1) == yx).mean())))
    rel = (Pt.max(-1), Q.max(-1), ok)
    arrays = dict(y=yt, p_raw=Pt[:, :, 1], p_ts=Q[:, :, 1])
    return rows, con, xrows, frows, rel, arrays, test_out


def unit_job(a):
    return unit(*a)


# ====================================== (a2) per learner on pooled out-of-fold predictions
def pooled_learners(ds, seed, parts, targets):
    """per learner on the pooled out-of-fold test predictions of one seed (the n of the fused posteriors),
    raw and with each fold's validation T; and, per transfer target, the in-domain ECEs at
    n = min(pooled n, target n) that the corpus-shift comparison uses"""
    y = np.concatenate([p["y"] for p in parts])
    n = len(y)
    P = {t: [two(np.concatenate([p[f"p_{t}"][m] for p in parts])) for m in range(len(L.MO))] for t in ("raw", "ts")}
    rows = []
    for m, name in enumerate(L.MO):
        r = OrderedDict(ds=ds, seed=seed, learner=L.MLAB[name], n=n, acc=float((P["raw"][m].argmax(-1) == y).mean()))
        for t in ("raw", "ts"):
            r.update(calib_full(y, P[t][m], t, named_rng("floor", ds, seed, L.MLAB[name], t)))
        rows.append(r)
    by_n = OrderedDict()
    for nm in sorted({min(n, v) for v in targets.values()}):
        if nm >= n:                                  # the whole pooled sample: the values above
            by_n[nm] = [OrderedDict((f"{f}_{t}_m", r[f"{f}_{t}"]) for t in ("raw", "ts") for f in ("ece", "floor"))
                        for r in rows]
            continue
        rng = named_rng("sub", ds, seed, "pooled", nm)
        idx = sub_index(n, nm, rng)
        by_n[nm] = []
        for m in range(len(L.MO)):
            v = OrderedDict()
            for t in ("raw", "ts"):
                v[f"ece_{t}_m"], v[f"floor_{t}_m"] = ece_at(P[t][m].max(-1), P[t][m].argmax(-1) == y, idx, rng)
            by_n[nm].append(v)
    return rows, OrderedDict((tgt, by_n[min(n, nt)]) for tgt, nt in targets.items())


def pooled_job(a):
    return pooled_learners(*a)


# ============================================================= (d) fusion rules
def boot_resamples(strata, S, n_boot, seed=0):
    """the resamples of L.two_stage_bootstrap_all (same generator, same call order)"""
    rng = np.random.default_rng(seed)
    idx = L.stratified_boot_index(strata, n_boot, rng)
    return idx, rng.integers(0, S, size=(n_boot, S))


def calib_reps(y, probs, idx, sidx, chunk=500):
    """seed-averaged ECE, NLL and Brier replicates for P(pos) probs (S, n) on given resamples.
    ECE = sum_b |sum_{i in b} (correct_i - conf_i)| / n, so a resample needs one weighted bincount."""
    S, n = probs.shape
    B = idx.shape[0]
    E, N, R = np.zeros((S, B)), np.zeros((S, B)), np.zeros((S, B))
    for s in range(S):
        P2 = two(probs[s])
        conf = P2.max(-1)
        g = (P2.argmax(-1) == y) - conf
        b = bins_of(conf)
        loss = -np.log(np.clip(P2[np.arange(n), y], 1e-12, 1.0))
        sq = (probs[s] - y) ** 2                       # = L.brier per instance
        for i in range(0, B, chunk):
            ii = idx[i:i + chunk]
            nb = len(ii)
            key = b[ii] + N_BINS * np.arange(nb)[:, None]
            E[s, i:i + nb] = np.abs(np.bincount(key.ravel(), g[ii].ravel(), nb * N_BINS).reshape(nb, N_BINS)).sum(1) / n
            N[s, i:i + nb] = loss[ii].mean(1)
            R[s, i:i + nb] = sq[ii].mean(1)
    col = np.arange(B)[:, None]
    return E[sidx, col].mean(1), N[sidx, col].mean(1), R[sidx, col].mean(1)


def fusion(ds, seeds, n_boot, test_out):
    runs = [L.pooled_run(ds, s, rules=RULES) for s in seeds]
    y, strata = runs[0]["y"], runs[0]["fold"]
    for s, R in zip(seeds, runs):
        assert np.array_equal(R["y"], y) and np.array_equal(R["fold"], strata), "pooled order differs between seeds"
        for f in range(L.N_FOLDS):                       # unit() refitted the rules to score the transfer corpora
            sel = strata == f
            for k, (pr, pp) in test_out[(s, f)].items():
                assert np.array_equal(R["pred"][k][sel], pr) and np.array_equal(R["prob"][k][sel], pp), \
                    f"unit() and pooled_run fits differ: {ds} seed {s} fold {f} {k}"
    table = OrderedDict()
    for k in runs[0]["prob"]:
        per = []
        for s, R in zip(seeds, runs):
            P2 = two(R["prob"][k])
            per.append(OrderedDict(f1_cv=L.summarise_folds(R["per_fold"][k])["mean"], f1_pooled=L.macro_f1(y, R["pred"][k]),
                                   ece=L.ece(y, P2, N_BINS), aece=L.ece(y, P2, N_BINS, adaptive=True),
                                   nll=L.nll(y, P2), brier=L.brier(y, P2),
                                   gap=float(P2.max(-1).mean() - (P2.argmax(-1) == y).mean()),
                                   ece_floor=ece_at(P2.max(-1), P2.argmax(-1) == y, None, named_rng("floor", ds, s, k, "raw"))[1],
                                   dece=dece(y, P2)))
        table[k] = OrderedDict((f, dict(msd([p[f] for p in per]), per_seed=[p[f] for p in per])) for f in per[0])
    preds = OrderedDict((k, np.stack([R["pred"][k] for R in runs])) for k in RULES)
    f1_pt, f1_reps = L.two_stage_bootstrap_all(y, preds, strata, n_boot=n_boot, seed=0)
    idx, sidx = boot_resamples(strata, len(seeds), n_boot)
    chk = L.macro_f1_batch(y, preds[L.PRIMARY], idx)[sidx, np.arange(n_boot)[:, None]].mean(1)
    assert np.allclose(chk, f1_reps[L.PRIMARY]), "bootstrap resamples differ from rev_lib's"
    pt, reps = OrderedDict(), OrderedDict()
    for k in RULES:
        probs = np.stack([R["prob"][k] for R in runs])
        for met, rep in zip(("ece", "nll", "brier"), calib_reps(y, probs, idx, sidx)):
            pt.setdefault(met, {})[k] = table[k][met]["mean"]
            reps.setdefault(met, {})[k] = rep
        table[k]["ece"]["boot_rep_bias"] = float(reps["ece"][k].mean() - pt["ece"][k])   # mean replicate - point
    tests = OrderedDict()
    for a, b in PAIRS:
        f1 = L.compare_from_reps(f1_pt, f1_reps, a, b)
        if (a, b) in EQUIV_PAIRS:
            f1["tost"] = tost(f1_reps[a] - f1_reps[b], f1["ci90"])
        e = L.compare_from_reps(pt["ece"], reps["ece"], a, b)
        de = reps["ece"][a] - reps["ece"][b]
        e.update(descriptive_only=True, rep_bias=float(de.mean() - e["diff"]),
                 ci95_basic=[float(2 * e["diff"] - np.percentile(de, 97.5)), float(2 * e["diff"] - np.percentile(de, 2.5))])
        tests[f"{a} vs {b}"] = OrderedDict(f1_pooled=f1, ece=e, nll=L.compare_from_reps(pt["nll"], reps["nll"], a, b),
                                           brier=L.compare_from_reps(pt["brier"], reps["brier"], a, b))
    fam = OrderedDict(primary=[f"{a} vs {b}" for a, b in PRIMARY_FAMILY],
                      exploratory=[f"{a} vs {b}" for a, b in PAIRS
                                   if (a, b) not in PRIMARY_FAMILY and not a.endswith(", output TS")])
    for name, keys in fam.items():
        for k, p in L.holm({k: tests[k]["f1_pooled"]["p_boot"] for k in keys}).items():
            tests[k]["f1_pooled"].update(family=name, p_holm=p)
    for t in tests.values():
        if "family" not in t["f1_pooled"]:              # output TS: identical decisions by construction, no test
            t["f1_pooled"].update(family="none (output TS never changes a decision)", p_holm=None)
    for met in ("nll", "brier"):
        for k, p in L.holm({k: t[met]["p_boot"] for k, t in tests.items()}).items():
            tests[k][met].update(family="calibration", p_holm=p)
    lam = [m["sawsv_boot"]["lam"] for R in runs for m in R["meta"]]
    return OrderedDict(seeds=seeds, n=len(y), rules=table, tests=tests,
                       primary=OrderedDict((k, tests[k]["f1_pooled"]) for k in fam["primary"]),
                       families=fam, sawsv_boot_lambda=msd(lam))


def fusion_job(a):
    return fusion(*a)


# ============================================================== (f) the figure
def reliability(conf, ok):
    b = bins_of(conf)
    n = np.bincount(b, minlength=N_BINS)
    s_conf = np.bincount(b, conf, N_BINS)
    s_ok = np.bincount(b, ok.astype(float), N_BINS)
    keep = n > 0
    return OrderedDict(count=n[keep], conf=s_conf[keep] / n[keep], acc=s_ok[keep] / n[keep],
                       ece=float(np.abs(s_ok - s_conf).sum() / len(conf)), share_top_bin=float(n[-1] / len(conf)))


def fig_reliability(rel, path):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.0), sharex=True, sharey=True)
    for ax, kind, lab in zip(axes, ("raw", "ts"), ("raw", "temperature-scaled")):
        ax.plot([0.5, 1.0], [0.5, 1.0], color="0.35", lw=0.8, ls=":", zorder=1)
        for m, name in enumerate(L.MO):
            r = rel[L.MLAB[name]][kind]
            keep = r["count"] >= MIN_BIN_FIG
            ax.plot(r["conf"][keep], r["acc"][keep], ls=LS[m], marker=MK[m], color=COL[m], ms=2.8, lw=0.9,
                    mew=0.5, label=SHORT[name], zorder=2)
        ax.text(0.04, 0.97, lab, transform=ax.transAxes, va="top", ha="left", fontsize=7)
        ax.set_xlim(0.48, 1.01); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xticklabels(["0.5", "", "0.7", "", "0.9", ""])
        ax.set_xlabel("top-label probability")
    axes[0].set_ylabel("accuracy")
    axes[1].legend(frameon=False, fontsize=6, loc="lower right", handlelength=2.2, borderaxespad=0.2, labelspacing=0.25)
    fig.tight_layout(pad=0.25, w_pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ===================================================================== report
def fmt(s, d=3):
    return "  -  " if s is None or s["mean"] is None else f"{s['mean']:.{d}f}"


def fsd(s, d=2):
    return "  -  " if s["sd"] is None else f"{s['sd']:.{d}f}"


def fp(p):
    return "  -   " if p is None else f"{p:.4f}"


def shown(path):
    try:
        return os.path.relpath(path, L.ROOT)
    except ValueError:                                  # another drive
        return path


def report(out):
    dss = list(out["fusion"])
    print("\n(a,b) per learner on the test folds (n = 277/719 per fold), mean over seeds x folds (SD over units); TS uses the "
          "validation-fitted T; AUROC of c_m for 'learner correct' (identical after TS, see check). Per-fold ECE is biased "
          "upward by the small n: compare it only within this table, (a2) has the estimator for comparisons across settings")
    print(f"{'corpus':10s}{'learner':18s}{'T (SD)':>12s}{'ECE raw->TS [oracle]':>23s}{'NLL raw->TS':>14s}{'gap raw->TS':>16s}"
          f"{'AUROC':>7s}{'mean c raw->TS':>16s}{'c|ok/c|err raw':>16s}{'TS':>10s}")
    for ds, g in out["per_learner"].items():
        for lab, s in list(g.items()) + [("mean of learners", out["per_corpus"][ds])]:
            print(f"{L.DS_LAB[ds]:10s}{lab:18s}{fmt(s['T'], 2):>6s} ({fsd(s['T'])})"
                  f"{fmt(s['ece_raw']):>9s}->{fmt(s['ece_ts'])} [{fmt(s['ece_ts_oracle'])}]"
                  f"{fmt(s['nll_raw']):>7s}->{fmt(s['nll_ts'])}{s['gap_raw']['mean']:+8.3f}->{s['gap_ts']['mean']:+.3f}"
                  f"{fmt(s['auroc_raw']):>7s}{fmt(s['conf_raw']):>9s}->{fmt(s['conf_ts'])}"
                  f"{fmt(s['conf_ok_raw']):>10s}/{fmt(s['conf_wrong_raw'])}{fmt(s['conf_ok_ts']):>7s}/{fmt(s['conf_wrong_ts'])}")
    inv = out["auroc_invariance"]
    print(f"TS invariance over {inv['n_learner_units']} learner-units: argmax changes {inv['argmax_changed']}, "
          f"max |AUROC_TS - AUROC_raw| = {inv['max_abs_diff_ts_vs_raw']:.1e}, max |AUROC_margin - AUROC_raw| = "
          f"{inv['max_abs_diff_margin_vs_raw']:.1e}; T > 1 in {100 * inv['frac_T_gt_1']:.0f}% of learner-units; "
          f"T at a search bound: {inv['T_at_bound']}")
    print("\ntemperatures: max/min over learners of the learner-mean T [noise-only reference with the learner labels shuffled "
          "within units: mean, 95th percentile; permutation p]; mean within-fold max/min (inflated by the estimation noise of "
          "T); T refitted on the test fold (oracle) vs the validation T")
    for ds, t in out["temperature"].items():
        print(f"{L.DS_LAB[ds]:10s} learner-mean T " + "/".join(f"{v:.2f}" for v in t["T_learner_means"].values()) +
              f": ratio {t['ratio_learner_means']:.2f} [noise {t['noise_ref_mean']:.2f}, 95% {t['noise_ref_q95']:.2f}; "
              f"p {t['perm_p']:.4f}] | within-fold ratio {t['ratio_within_unit_mean']:.2f} (noisy) | T_oracle > T in "
              f"{100 * t['frac_T_oracle_gt_T']:.0f}% of learner-units, geometric mean T_oracle/T {t['geo_mean_T_oracle_over_T']:.2f}")
    print("\n(a2) per learner on the pooled out-of-fold predictions of each seed (the n of the fused posteriors), mean over "
          "seeds: ECE raw->TS [floor raw/TS], debiased L2-ECE raw->TS, NLL raw->TS, Brier raw->TS, gap raw->TS")
    for ds, g in out["per_learner_pooled"].items():
        pc = out["per_corpus_pooled"][ds]
        for lab, s in list(g.items()) + [("mean of learners", pc)]:
            print(f"{L.DS_LAB[ds]:10s}{lab:18s}{fmt(s['ece_raw'])}->{fmt(s['ece_ts'])} [{fmt(s['floor_raw'])}/"
                  f"{fmt(s['floor_ts'])}]{fmt(s['dece_raw']):>8s}->{fmt(s['dece_ts'])}{fmt(s['nll_raw']):>8s}->"
                  f"{fmt(s['nll_ts'])}{fmt(s['brier_raw'], 4):>8s}->{fmt(s['brier_ts'], 4)}"
                  f"{s['gap_raw']['mean']:+8.3f}->{s['gap_ts']['mean']:+.3f}")
        rr = pc["ts_relative_reduction"]
        npos = sum(v["gap_ts"]["mean"] > 0 for v in g.values())
        print(f"{'':10s} n = {out['fusion'][ds]['n']}; TS lowers the learner-mean ECE by {100 * rr['ece']:.0f}%, dECE by "
              f"{100 * rr['dece']:.0f}%, NLL by {100 * rr['nll']:.1f}%, Brier by {100 * rr['brier']:.1f}%; "
              f"post-TS gap > 0 for {npos}/{len(g)} learners")
    print("\n(c) cross-learner confidence scale and contested instances (mean over seeds x folds)")
    for ds, s in out["contested"].items():
        print(f"{L.DS_LAB[ds]:10s} contested {100 * s['frac_contested']['mean']:.1f}% | range of the learner-mean c_m raw "
              f"{s['conf_range_learner_means_raw']:.3f} -> TS {s['conf_range_learner_means_ts']:.3f} (mean within-unit range "
              f"{fmt(s['conf_range_raw'])} -> {fmt(s['conf_range_ts'])}) | confidence mass on correct learners raw "
              f"{fmt(s['correct_mass_raw'])}, TS {fmt(s['correct_mass_ts'])} vs uniform {fmt(s['count_frac'])}")
        print(f"{'':10s} most confident learner correct: raw {fmt(s['most_conf_acc_raw'])}, TS {fmt(s['most_conf_acc_ts'])} "
              f"(changed by TS on {100 * s['frac_top_changed']['mean']:.1f}%) | accuracy on contested: uniform "
              f"{fmt(s['acc_uniform_raw'])}/{fmt(s['acc_uniform_ts'])} (raw/TS), entropy {fmt(s['acc_entropy_raw'])}/"
              f"{fmt(s['acc_entropy_ts'])}, majority {fmt(s['acc_majority'])}")
        print(f"{'':10s} share raw/TS/accuracy: " + "; ".join(
            f"{k} {v['share_raw']:.3f}/{v['share_ts']:.3f}/{v['acc']:.3f}" for k, v in s["learners"].items())
              + f" | Pearson(share, acc) raw {s['pearson_share_acc_raw']:+.2f}, TS {s['pearson_share_acc_ts']:+.2f}")
    print("\n(d) fusion on pooled out-of-fold predictions, mean over seeds: macro-F1 (%, CV mean) | ECE [floor] | debiased "
          "L2-ECE | NLL | Brier | gap")
    for ds in dss:
        F = out["fusion"][ds]
        print(f"{L.DS_LAB[ds]} (n = {F['n']}, seeds {F['seeds']})")
        for k, r in F["rules"].items():
            print(f"  {k:32s}{100 * r['f1_cv']['mean']:6.2f} | {r['ece']['mean']:.3f} [{r['ece_floor']['mean']:.3f}] | "
                  f"{r['dece']['mean']:.3f} | {r['nll']['mean']:.3f} | {r['brier']['mean']:.4f} | {r['gap']['mean']:+.3f}")
    print("\n(d) primary family: pooled macro-F1 difference to SA-WSV (bootstrap), points, [95% CI] (90% CI); p_boot; Holm "
          "over the 3 per corpus [over all corpora]; TOST: 90% CI inside +/-0.5 or +/-1.0 points")
    for ds in dss:
        for k, t in out["fusion"][ds]["primary"].items():
            line = (f"{L.DS_LAB[ds]:10s}{k:47s}{100 * t['diff']:+6.2f} [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}] "
                    f"({100 * t['ci90'][0]:+.2f},{100 * t['ci90'][1]:+.2f}) p {fp(t['p_boot'])} Holm {fp(t['p_holm'])} "
                    f"[{fp(t['p_holm_all_corpora'])}]")
            if "tost" in t:
                line += " | TOST " + ", ".join(f"+/-{100 * float(dl):.1f}: {'equivalent' if v['equivalent'] else 'not shown'}"
                                               f" (p {v['p']:.4f})" for dl, v in t["tost"].items())
            print(line)
    print("\n(d) paired two-stage bootstrap, a - b [95% CI] p_Holm: F1 in points (primary or exploratory family), NLL and Brier "
          "(calibration inference, Holm over the 10 pairs); ECE descriptive only (basic CI; percentile CIs are pulled to 0)")
    for pair in out["fusion"][dss[0]]["tests"]:
        cells = []
        for ds in dss:
            t = out["fusion"][ds]["tests"][pair]
            f, e, nl, br = t["f1_pooled"], t["ece"], t["nll"], t["brier"]
            cells.append(f"{L.DS_LAB[ds]:9s} F1 {100 * f['diff']:+.2f} [{100 * f['ci95'][0]:+.2f},{100 * f['ci95'][1]:+.2f}] "
                         f"{fp(f['p_holm'])} | NLL {nl['diff']:+.4f} [{nl['ci95'][0]:+.4f},{nl['ci95'][1]:+.4f}] "
                         f"{fp(nl['p_holm'])} | Brier {br['diff']:+.4f} [{br['ci95'][0]:+.4f},{br['ci95'][1]:+.4f}] "
                         f"{fp(br['p_holm'])} | ECE {e['diff']:+.3f} (basic [{e['ci95_basic'][0]:+.3f},"
                         f"{e['ci95_basic'][1]:+.3f}])")
        print(f"  {pair} ({out['fusion'][dss[0]]['tests'][pair]['f1_pooled']['family']})\n    " + "\n    ".join(cells))
    print("\n(e) corpus shift, mean over learners (T fitted on the source validation split). ECE at a common n per pair: "
          "pooled in-domain predictions of the same seeds vs the target, the larger side subsampled; (floor)")
    for k, s in out["transfer"].items():
        a = s["mean_over_learners"]
        print(f"{k:27s} units {s['n_units']:2d} | n {s['n_match']} | acc in {a['in_acc']:.3f} -> xfer {a['xfer_acc']:.3f} | "
              f"ECE in {a['in_ece_raw_m']:.3f} ({a['in_floor_raw_m']:.3f}) -> TS {a['in_ece_ts_m']:.3f} "
              f"({a['in_floor_ts_m']:.3f}); xfer {a['xfer_ece_raw_m']:.3f} ({a['xfer_floor_raw_m']:.3f}) -> TS "
              f"{a['xfer_ece_ts_m']:.3f} ({a['xfer_floor_ts_m']:.3f}), target-oracle T {a['xfer_ece_oracle_m']:.3f}")
        print(f"{'':27s} NLL in {a['in_nll_raw']:.3f} -> xfer {a['xfer_nll_raw']:.3f} -> TS {a['xfer_nll_ts']:.3f} | Brier xfer "
              f"{a['xfer_brier_raw']:.4f} -> TS {a['xfer_brier_ts']:.4f} | gap xfer {a['xfer_gap_raw']:+.3f}->{a['xfer_gap_ts']:+.3f}"
              f" | AUROC in {a['in_auroc_raw']:.3f}, xfer {a['xfer_auroc_raw']:.3f}")
    print("\n(e2) fused posterior under corpus shift, mean over units: macro-F1 (%) / ECE (same target, same n per row)")
    ab = ["Uniform", "Uniform+TS", "Entropy", "Entropy+TS", "SA-WSV", "SA-WSV+TS", "SA-WSV rank", "SA-WSV no-c", "SA-WSV outTS"]
    print(f"{'':27s}" + "".join(f"{a:>14s}" for a in ab))
    for k, s in out["transfer"].items():
        print(f"{k:27s}" + "".join(f"{100 * s['fused'][r]['f1']['mean']:8.2f}/{s['fused'][r]['ece']['mean']:.3f}"
                                   for r in XFER_SHOW))


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--seeds", type=int, nargs="+", default=L.SEEDS, help="restrict to these seeds (default: all complete)")
    ap.add_argument("--workers", type=int, default=3, help=f"worker processes (at most {MAX_WORKERS})")
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--fig", default=OUT_FIG)
    args = ap.parse_args()
    t0 = time.perf_counter()
    seeds = OrderedDict((ds, L.complete_seeds(ds, args.seeds)) for ds in L.IN_DOMAIN)   # fixed for the whole run
    n_pool = OrderedDict((ds, sum(len(L.load_fold(ds, f, ss[0])["yt"]) for f in range(L.N_FOLDS))) for ds, ss in seeds.items())
    jobs = [(ds, seed, fold, n_pool[ds]) for ds, ss in seeds.items() for seed in ss for fold in range(L.N_FOLDS)]
    rows, cons, xrows, frows, parts, test_out = [], [], [], [], OrderedDict(), OrderedDict()
    rel_acc = {m: ([], [], []) for m in L.MO}
    with Pool(max(1, min(args.workers, MAX_WORKERS))) as pool:
        for (ds, seed, fold, _), (r, c, x, fr, rel, arr, to) in zip(jobs, pool.map(unit_job, jobs)):
            rows += r; cons.append(c); xrows += x; frows += fr
            parts.setdefault((ds, seed), []).append(arr)
            test_out.setdefault(ds, OrderedDict())[(seed, fold)] = to
            if ds == "evprimary-formal":
                for m, name in enumerate(L.MO):
                    for lst, v in zip(rel_acc[name], (rel[0][m], rel[1][m], rel[2][m])):
                        lst.append(v)
        targets = OrderedDict((u, OrderedDict(sorted({(r["tgt"], r["n_tgt"]) for r in xrows if (r["ds"], r["seed"]) == u})))
                              for u in parts)
        pres = pool.map(pooled_job, [(ds, s, parts[(ds, s)], targets[(ds, s)]) for ds, s in parts])
        fus = OrderedDict(zip(seeds, pool.map(fusion_job, [(ds, ss, args.n_boot, test_out[ds]) for ds, ss in seeds.items()])))
    print(f"seeds used: {dict((L.DS_LAB[k], v) for k, v in seeds.items())}; units: {len(cons)}")
    prows, matched = [], OrderedDict()
    for u, (pr, mt) in zip(parts, pres):
        prows += pr; matched[u] = mt
    for r in prows:                                    # the fusion table scores the single learners on the same pool
        tab, i = fus[r["ds"]]["rules"][r["learner"]], fus[r["ds"]]["seeds"].index(r["seed"])
        assert abs(r["ece_raw"] - tab["ece"]["per_seed"][i]) < 1e-12 and r["floor_raw"] == tab["ece_floor"]["per_seed"][i]
    allp = {(ds, k): v["p_boot"] for ds, F in fus.items() for k, v in F["primary"].items()}
    for (ds, k), p in L.holm(allp).items():
        fus[ds]["primary"][k]["p_holm_all_corpora"] = p
        fus[ds]["tests"][k]["f1_pooled"]["p_holm_all_corpora"] = p

    per_learner = nest(summarise(rows, ("ds", "learner"), LEARNER_FIELDS))
    per_corpus = OrderedDict((k[0], with_between(v, per_learner[k[0]], CORPUS_FIELDS))
                             for k, v in summarise(unit_means(rows, CORPUS_FIELDS), ("ds",), CORPUS_FIELDS).items())
    inv = OrderedDict(n_learner_units=len(rows), argmax_changed=int(sum(r["argmax_changed"] for r in rows)),
                      max_abs_diff_ts_vs_raw=float(max(abs(r["auroc_ts"] - r["auroc_raw"]) for r in rows)),
                      max_abs_diff_margin_vs_raw=float(max(abs(r["auroc_margin"] - r["auroc_raw"]) for r in rows)),
                      max_abs_diff_ts_vs_margin=float(max(abs(r["auroc_ts"] - r["auroc_margin"]) for r in rows)),
                      T_at_bound=int(sum(r["T_at_bound"] for r in rows)),
                      frac_T_gt_1=float(np.mean([r["T"] > 1 for r in rows])))
    contested = OrderedDict()
    for (ds,), s in summarise(cons, ("ds",), CONTESTED_FIELDS).items():
        lr = OrderedDict((lab, OrderedDict(share_raw=v["share_raw"]["mean"], share_ts=v["share_ts"]["mean"],
                                           acc=v["acc_contested"]["mean"])) for lab, v in per_learner[ds].items())
        sr, st, ac = (np.array([v[f] for v in lr.values()]) for f in ("share_raw", "share_ts", "acc"))
        cm = {t: [v[f"conf_{t}"]["mean"] for v in per_learner[ds].values()] for t in ("raw", "ts")}
        contested[ds] = OrderedDict(s, conf_range_learner_means_raw=float(np.ptp(cm["raw"])),
                                    conf_range_learner_means_ts=float(np.ptp(cm["ts"])), learners=lr,
                                    pearson_share_acc_raw=float(np.corrcoef(sr, ac)[0, 1]),
                                    pearson_share_acc_ts=float(np.corrcoef(st, ac)[0, 1]))
    temperature = OrderedDict()
    for ds in seeds:
        rr = [r for r in rows if r["ds"] == ds]
        assert [r["learner"] for r in rr] == [L.MLAB[m] for m in L.MO] * (len(rr) // len(L.MO))
        obs, null = t_ratio(np.array([r["T"] for r in rr]).reshape(-1, len(L.MO)), np.random.default_rng([L.DS_ID[ds], SALT]))
        lo = np.log([r["T_oracle"] / r["T"] for r in rr])
        temperature[ds] = OrderedDict(
            T_learner_means=OrderedDict((lab, v["T"]["mean"]) for lab, v in per_learner[ds].items()),
            ratio_learner_means=obs, noise_ref_mean=float(null.mean()), noise_ref_q95=float(np.percentile(null, 95)),
            perm_p=float((1 + (null >= obs).sum()) / (len(null) + 1)),
            ratio_within_unit_mean=contested[ds]["T_ratio"]["mean"],
            frac_T_oracle_gt_T=float((lo > 0).mean()), geo_mean_T_oracle_over_T=float(np.exp(lo.mean())))

    per_learner_pooled = nest(summarise(prows, ("ds", "learner"), POOLED_FIELDS))
    per_corpus_pooled = OrderedDict()
    for (ds,), v in summarise(unit_means(prows, POOLED_FIELDS, by=("ds", "seed")), ("ds",), POOLED_FIELDS).items():
        v = with_between(v, per_learner_pooled[ds], POOLED_FIELDS)
        v["ts_relative_reduction"] = OrderedDict((f, 1.0 - v[f"{f}_ts"]["mean"] / v[f"{f}_raw"]["mean"])
                                                 for f in ("ece", "dece", "nll", "brier"))
        per_corpus_pooled[ds] = v

    transfer = OrderedDict()
    ind = {(r["ds"], r["seed"], r["fold"], r["learner"]): r for r in rows}
    for src, tgt in OrderedDict.fromkeys((r["ds"], r["tgt"]) for r in xrows):
        xs = [r for r in xrows if r["ds"] == src and r["tgt"] == tgt]
        xseeds = sorted({r["seed"] for r in xs})
        lrn = OrderedDict()
        for i, lab in enumerate(L.MLAB[m] for m in L.MO):
            xl = [r for r in xs if r["learner"] == lab]
            il = [ind[(r["ds"], r["seed"], r["fold"], lab)] for r in xl]      # the same units, scored in domain
            im = [matched[(src, s)][tgt][i] for s in xseeds]                  # pooled in domain, same seeds, matched n
            lrn[lab] = OrderedDict([(f"xfer_{f}", msd([r[f] for r in xl])) for f in XFER_FIELDS] +
                                   [(f"in_{f}", msd([r[f] for r in il])) for f in IN_FIELDS] +
                                   [(f"in_{f}", msd([v[f] for v in im])) for f in IN_M_FIELDS])
        mol = OrderedDict((f, float(np.mean([v[f]["mean"] for v in lrn.values()]))) for f in lrn[lab])
        fr = {(r["seed"], r["fold"], r["rule"]): r for r in frows if r["ds"] == src and r["tgt"] == tgt}
        units = list(OrderedDict.fromkeys((s, f) for s, f, _ in fr))
        fused = OrderedDict((k, OrderedDict((f, msd([fr[u + (k,)][f] for u in units])) for f in FUSED_FIELDS)) for k in RULES)
        fpairs = OrderedDict((f"{a} vs {b}", OrderedDict(
            (f, dict(msd([fr[u + (a,)][f] - fr[u + (b,)][f] for u in units]),
                     n_positive=int(sum(fr[u + (a,)][f] > fr[u + (b,)][f] for u in units)))) for f in FUSED_FIELDS))
            for a, b in PAIRS)
        transfer[f"{L.DS_LAB[src]} -> {L.DS_LAB[tgt]}"] = OrderedDict(
            src=src, tgt=tgt, seeds=xseeds, n_units=len({(r["seed"], r["fold"]) for r in xs}),
            n_tgt=xs[0]["n_tgt"], n_match=xs[0]["n_match"], learners=lrn, mean_over_learners=mol, fused=fused,
            fused_paired_diffs=fpairs)

    rel = OrderedDict()
    for name in L.MO:
        conf_r, conf_t, ok = (np.concatenate(v) for v in rel_acc[name])
        rel[L.MLAB[name]] = OrderedDict(raw=reliability(conf_r, ok), ts=reliability(conf_t, ok))
    os.makedirs(os.path.dirname(os.path.abspath(args.fig)), exist_ok=True)
    fig_reliability(rel, args.fig)

    out = OrderedDict(
        config=OrderedDict(script="experiments/rev_calibration.py", n_bins=N_BINS, n_boot_pairs=args.n_boot,
                           n_boot_shrinkage=L.N_BOOT, min_bin_fig=MIN_BIN_FIG, ece="top-label, 15 equal-width bins",
                           aece="top-label, 15 equal-mass bins", brier="mean squared error of P(positive)",
                           gap="mean top-label confidence minus accuracy", confidence="c_m = 1 - H(p_m)/ln 2 (L.conf_entropy)",
                           temperature="per learner and unit, validation NLL (L.fit_temperature), bounds [0.05, 20]",
                           oracle="*_oracle fields refit T on the scored labels; reference only",
                           T_oracle="T refitted on the test fold's NLL; T_oracle > T means the validation T under-corrects "
                                    "(the validation split also selected the checkpoint by F1)",
                           contested="test instances on which the learners' predicted labels differ",
                           T_ratio="contested.T_ratio is the mean within-unit max/min and includes the estimation noise of T; "
                                   "temperature.ratio_learner_means is max/min of the learner-mean T, with a noise-only "
                                   f"reference from {N_PERM} within-unit shuffles of the learner labels",
                           conf_range="contested.conf_range_raw/ts: mean within-unit range of the learner mean c_m; "
                                      "conf_range_learner_means_raw/ts: range of the learner means",
                           per_corpus_sd="per_corpus and per_corpus_pooled: SD over units (seeds) of the mean of the learners; "
                                         "sd_between_learners: SD of the learner means",
                           ece_comparisons="per_learner: per test fold, compare only within it; per_learner_pooled and fusion: "
                                           "pooled out-of-fold predictions per seed (same n), for comparisons in domain; "
                                           "*_m fields: n_match = min(pooled source n, target n), the larger side averaged "
                                           f"over {N_SUB} subsamples without replacement, for in domain vs corpus shift",
                           ece_floor="mean plug-in ECE of a perfectly calibrated predictor with the same confidences and n "
                                     f"(correctness ~ Bernoulli(confidence), {N_SUB} draws)",
                           monte_carlo="size-matched ECEs and floors are means over N_SUB random draws; their Monte Carlo SD "
                                       "is about 0.0005 per learner and seed (one set of subsamples is shared by the "
                                       "learners of a seed or unit, so learner means do not average it away)",
                           dece="debiased top-label L2-ECE (Kumar et al., 2019), 15 equal-width bins; an L2 quantity, "
                                "not on the scale of the L1 ECE; slightly biased upward near 0 (clipping, square root)",
                           inference="NLL and Brier differences: two-stage bootstrap percentile CI, p_boot, Holm; ECE "
                                     "differences descriptive only (rep_bias = mean replicate - point of the difference; "
                                     "ci95_basic = basic bootstrap CI)",
                           primary_family=[f"{a} vs {b}" for a, b in PRIMARY_FAMILY],
                           multiplicity="pooled macro-F1: primary family Holm within corpus (p_holm) and over corpora "
                                        "(p_holm_all_corpora); other F1 tests exploratory, Holm within corpus; NLL and Brier "
                                        "Holm within corpus and metric over all pairs; p_boot = 0 means below 2/n_boot",
                           equivalence=f"TOST on pooled macro-F1 for {[f'{a} vs {b}' for a, b in EQUIV_PAIRS]} at margins "
                                       f"{list(DELTAS)}: equivalent iff the 90% percentile CI lies inside (-delta, +delta); "
                                       "a claim for every corpus is an intersection-union test, so no correction",
                           fusion_rules=FUSION, extra_rules=EXTRA_RULES, labels=L.DS_LAB),
        seeds_used=seeds, per_learner=per_learner, per_corpus=per_corpus, auroc_invariance=inv, temperature=temperature,
        per_learner_pooled=per_learner_pooled, per_corpus_pooled=per_corpus_pooled,
        contested=contested, fusion=fus, transfer=transfer,
        reliability_ev_norm=OrderedDict(seeds=seeds["evprimary-formal"], learners=rel),
        units=rows, contested_units=cons, pooled_units=prows, transfer_units=xrows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(_py(out), fh, indent=1)
    report(_py(out))
    print(f"\nwrote {shown(args.out)} and {shown(args.fig)}; runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
