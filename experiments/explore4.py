"""Exploration round 4: shrinkage-adaptive reliability weighting (SA-WSV).

Diagnosis from rounds 1-3: validation macro-F1 differences between the learners are of
the same order as their standard error, so any weight derived from them is mostly noise;
that is why every heuristic weighting matched plain uniform soft voting.

Proposal: shrink the reliability estimates toward their mean by a James-Stein factor
computed from the validation size, then weight with the shrunken estimates. When the
observed spread is explained by sampling noise the rule collapses to uniform soft voting;
when the spread is real it approaches F1-based weighting. Nothing is tuned on test data.
"""
import os, math, itertools, json
import numpy as np

MO = ["indobert-p1", "indobert-cahya", "indoroberta-cahya", "xlmr-base", "mbert-cased"]
EPS = 1e-6


def macro_f1(y, p):
    f = 0.0
    for c in (0, 1):
        tp = np.count_nonzero((p == c) & (y == c))
        fp = np.count_nonzero((p == c) & (y != c))
        fn = np.count_nonzero((p != c) & (y == c))
        d = 2 * tp + fp + fn
        f += (2.0 * tp / d) if d else 0.0
    return f / 2.0


def conf_entropy(P):
    Pc = np.clip(P, EPS, 1.0)
    return 1.0 - (-(Pc * np.log(Pc)).sum(-1)) / math.log(P.shape[-1])


def fuse(P, w=None, c=None, gamma=0.0):
    M, n, _ = P.shape
    W = np.ones((M, n)) if w is None else np.repeat(np.asarray(w, float)[:, None], n, 1)
    if gamma > 0:
        W = W * np.power(np.clip(c, EPS, 1.0), gamma)
    W = W / np.clip(W.sum(0, keepdims=True), EPS, None)
    return (W[:, :, None] * P).sum(0)


def logodds(f):
    f = np.clip(np.asarray(f, float), 0.5 + EPS, 1 - EPS)
    return np.log(f / (1 - f))


def softmax_w(f, beta):
    z = beta * np.asarray(f, float); z -= z.max(); e = np.exp(z); return e / e.sum()


def js_shrink(F, n_val):
    """James-Stein shrinkage of per-model accuracy estimates toward their mean.

    F: (M,) validation macro-F1. n_val: validation size.
    Returns shrunken estimates and the shrinkage factor lambda in [0,1]
    (lambda = 1 keeps the raw estimates, lambda = 0 collapses them to the mean).
    """
    F = np.asarray(F, float)
    M = len(F)
    Fbar = F.mean()
    # binomial-style sampling variance of an accuracy-like statistic
    var = float(np.mean(F * (1.0 - F)) / max(n_val, 1))
    ss = float(((F - Fbar) ** 2).sum())
    if ss <= 0 or M <= 3:
        lam = 0.0
    else:
        lam = max(0.0, 1.0 - (M - 3) * var / ss)
    return Fbar + lam * (F - Fbar), lam, math.sqrt(var)


def load(ds, fold):
    Pv, Pt = [], []
    yv = yt = None
    for m in MO:
        z = np.load(f"experiments/probs/{ds}/fold{fold}/{m}.npz")
        Pv.append(z["val_probs"]); Pt.append(z["test_probs"]); yv, yt = z["y_val"], z["y_test"]
    return np.stack(Pv), yv, np.stack(Pt), yt


BETAS = [0, 2, 5, 10, 20, 50, 100]


def rules_for(Pv, yv, Pt, n_val_eff=None, rng=None, sub=None):
    """Apply every rule. `sub` optionally subsamples the validation set to study the
    small-validation regime; reliability is then estimated from that subsample only."""
    M = len(MO)
    if sub is not None:
        idx = rng.choice(len(yv), size=sub, replace=False)
        Pv_r, yv_r = Pv[:, idx], yv[idx]
    else:
        Pv_r, yv_r = Pv, yv
    n_val = len(yv_r)
    F = np.array([macro_f1(yv_r, Pv_r[m].argmax(-1)) for m in range(M)])
    ct = conf_entropy(Pt)
    Fs, lam, se = js_shrink(F, n_val)
    out = {
        "best-single(val)": Pt[int(F.argmax())].argmax(-1),
        "uniform": fuse(Pt).argmax(-1),
        "linearF1": fuse(Pt, F).argmax(-1),
        "logodds": fuse(Pt, logodds(F)).argmax(-1),
        "logodds x conf": fuse(Pt, logodds(F), ct, 1).argmax(-1),
    }
    b = max(BETAS, key=lambda bb: macro_f1(yv_r, fuse(Pv_r, softmax_w(F, bb)).argmax(-1)))
    out["softmaxF1(b*)"] = fuse(Pt, softmax_w(F, b)).argmax(-1)
    # proposed: shrinkage-adaptive
    out["SA linear"] = fuse(Pt, Fs).argmax(-1)
    out["SA logodds"] = fuse(Pt, logodds(Fs)).argmax(-1)
    out["SA softmax(b=50)"] = fuse(Pt, softmax_w(Fs, 50)).argmax(-1)
    out["SA softmax(b*)"] = fuse(Pt, softmax_w(Fs, b)).argmax(-1)
    out["SA logodds x conf"] = fuse(Pt, logodds(Fs), ct, 1).argmax(-1)
    return out, {"lambda": lam, "se": se, "F": F.tolist(), "Fs": Fs.tolist(), "beta": b, "n_val": n_val}


def evaluate(ds, sub=None, reps=1, seed=0):
    rng = np.random.default_rng(seed)
    acc = {}
    meta = []
    for rep in range(reps):
        for fold in range(5):
            Pv, yv, Pt, yt = load(ds, fold)
            preds, info = rules_for(Pv, yv, Pt, rng=rng, sub=sub)
            meta.append(info)
            for k, p in preds.items():
                acc.setdefault(k, {"f1": [], "y": [], "p": []})
                acc[k]["f1"].append(macro_f1(yt, p))
                acc[k]["y"].append(yt); acc[k]["p"].append(p)
    res = {}
    for k, v in acc.items():
        y, p = np.concatenate(v["y"]), np.concatenate(v["p"])
        res[k] = {"mean": float(np.mean(v["f1"])), "std": float(np.std(v["f1"], ddof=1)), "pooled": macro_f1(y, p)}
    return res, meta


if __name__ == "__main__":
    report = {}
    for ds in ["evprimary-formal", "evpublic", "evprimary-informal"]:
        print(f"\n########## {ds} ##########")
        for sub, reps, tag in [(None, 1, "full validation"), (40, 8, "n_val=40"), (20, 8, "n_val=20")]:
            res, meta = evaluate(ds, sub=sub, reps=reps, seed=11)
            base = res["uniform"]["pooled"]
            lam = np.mean([m["lambda"] for m in meta]); se = np.mean([m["se"] for m in meta])
            print(f"\n--- {tag} (mean lambda={lam:.3f}, mean SE(F1)={se:.4f}, n_val={meta[0]['n_val']}) ---")
            for k, v in sorted(res.items(), key=lambda kv: -kv[1]["pooled"]):
                print(f"  {k:20s} f1={100 * v['mean']:6.2f}+-{100 * v['std']:4.2f} pooled={100 * v['pooled']:6.2f} vs-uniform={100 * (v['pooled'] - base):+5.2f}")
            report[f"{ds}|{tag}"] = {"res": res, "lambda": lam, "se": se}
    json.dump(report, open("experiments/results/explore4.json", "w"), indent=1)
