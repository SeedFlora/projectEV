"""Definitive analysis: produces every number, table and figure used in the paper.

Design notes
------------
* All fusion rules are scored on identical, frozen posteriors, so differences come
  only from the fusion rule and never from training variance.
* Hyper-parameters of any rule are fitted on validation posteriors only.
* Significance: exact McNemar on pooled out-of-fold predictions, Holm-corrected across
  the family of comparisons against the proposed rule, plus percentile bootstrap CIs.
"""
import os, math, json, glob, itertools, argparse
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

MO = ["indobert-p1", "indobert-cahya", "indoroberta-cahya", "xlmr-base", "mbert-cased"]
MLAB = {"indobert-p1": "IndoBERT-p1", "indobert-cahya": "IndoBERT-1.5G", "indoroberta-cahya": "IndoRoBERTa-1.5G",
        "xlmr-base": "XLM-R", "mbert-cased": "mBERT"}
DS_LAB = {"evprimary-formal": "EV-Norm", "evprimary-informal": "EV-Raw", "evpublic": "EV-Public"}
PROPOSED = "SA-WSV (proposed)"
EPS = 1e-6
BETAS = [0, 2, 5, 10, 20, 50, 100]


# ------------------------------------------------------------------ metrics
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
    return {"acc": float(np.mean(y == p)), "prec": float(np.mean(pr)), "rec": float(np.mean(rc)), "f1": macro_f1(y, p)}


# --------------------------------------------------------------- components
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


def js_shrink(F, n_val):
    """James-Stein shrinkage toward the mean; returns (shrunken F, lambda, SE)."""
    F = np.asarray(F, float); M = len(F); Fbar = F.mean()
    var = float(np.mean(F * (1.0 - F)) / max(n_val, 1))
    ss = float(((F - Fbar) ** 2).sum())
    lam = 0.0 if (ss <= 0 or M <= 3) else max(0.0, 1.0 - (M - 3) * var / ss)
    return Fbar + lam * (F - Fbar), lam, math.sqrt(var)


def fuse(P, w=None, c=None, gamma=0.0):
    M, n, _ = P.shape
    W = np.ones((M, n)) if w is None else np.repeat(np.asarray(w, float)[:, None], n, 1)
    if gamma > 0:
        W = W * np.power(np.clip(c, EPS, 1.0), gamma)
    W = W / np.clip(W.sum(0, keepdims=True), EPS, None)
    return (W[:, :, None] * P).sum(0)


def majority(P, tiebreak):
    v = P.argmax(-1)
    counts = np.stack([(v == c).sum(0) for c in range(P.shape[-1])], -1)
    pred = counts.argmax(-1)
    tie = counts.max(-1) == np.sort(counts, -1)[:, -2]
    pred[tie] = tiebreak.argmax(-1)[tie]
    return pred


def all_rules(Pv, yv, Pt):
    """name -> test predictions. Hyper-parameters fitted on (Pv, yv) only."""
    M = Pv.shape[0]
    F = np.array([macro_f1(yv, Pv[m].argmax(-1)) for m in range(M)])
    Fs, lam, se = js_shrink(F, len(yv))
    ct, cv = conf_entropy(Pt), conf_entropy(Pv)
    mt = conf_maxprob(Pt)
    uni = fuse(Pt)
    b = max(BETAS, key=lambda bb: macro_f1(yv, fuse(Pv, softmax_w(F, bb)).argmax(-1)))
    r = {
        "Best single (val-selected)": Pt[int(F.argmax())].argmax(-1),
        "Majority voting": majority(Pt, uni),
        "Uniform soft voting": uni.argmax(-1),
        "Linear-F1 weighting": fuse(Pt, F).argmax(-1),
        "Softmax-F1 weighting": fuse(Pt, softmax_w(F, b)).argmax(-1),
        "Log-odds weighting": fuse(Pt, logodds(F)).argmax(-1),
        "Max-prob confidence": fuse(Pt, None, mt, 1).argmax(-1),
        "Entropy confidence": fuse(Pt, None, ct, 1).argmax(-1),
        "Linear-F1 x max-prob": fuse(Pt, F, mt, 1).argmax(-1),
        "Log-odds x max-prob": fuse(Pt, logodds(F), mt, 1).argmax(-1),
        "Log-odds x entropy": fuse(Pt, logodds(F), ct, 1).argmax(-1),
        PROPOSED: fuse(Pt, logodds(Fs), ct, 1).argmax(-1),
    }
    return r, {"F": F.tolist(), "Fs": Fs.tolist(), "lambda": lam, "se": se, "beta": b,
               "w_raw": (logodds(F) / logodds(F).sum()).tolist(),
               "w_shrunk": (logodds(Fs) / logodds(Fs).sum()).tolist()}


# ------------------------------------------------------------ significance
def mcnemar(y, p1, p2):
    r1, r2 = p1 == y, p2 == y
    b, c = int((r1 & ~r2).sum()), int((~r1 & r2).sum())
    return b, c, (1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue))


def boot_ci(y, p1, p2, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    d = np.array([macro_f1(y[i], p1[i]) - macro_f1(y[i], p2[i]) for i in idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(pv):
    it = sorted(pv.items(), key=lambda kv: kv[1])
    m, adj, prev = len(it), {}, 0.0
    for i, (k, p) in enumerate(it):
        v = min(1.0, max(prev, (m - i) * p)); adj[k] = v; prev = v
    return adj


# ------------------------------------------------------------------- data
def load(ds, fold):
    Pv, Pt, Px = [], [], []
    yv = yt = None
    for m in MO:
        z = np.load(f"experiments/probs/{ds}/fold{fold}/{m}.npz")
        Pv.append(z["val_probs"]); Pt.append(z["test_probs"]); yv, yt = z["y_val"], z["y_test"]
        if "transfer_probs" in z:
            Px.append(z["transfer_probs"])
    return np.stack(Pv), yv, np.stack(Pt), yt, (np.stack(Px) if len(Px) == len(MO) else None)


def q_and_disagreement(ds):
    Q, D, n = [], [], 0
    for fold in range(5):
        _, _, Pt, yt, _ = load(ds, fold)
        ok = np.stack([Pt[m].argmax(-1) == yt for m in range(len(MO))])
        for i, j in itertools.combinations(range(len(MO)), 2):
            n11 = np.count_nonzero(ok[i] & ok[j]); n00 = np.count_nonzero(~ok[i] & ~ok[j])
            n10 = np.count_nonzero(ok[i] & ~ok[j]); n01 = np.count_nonzero(~ok[i] & ok[j])
            Q.append((n11 * n00 - n01 * n10) / max(n11 * n00 + n01 * n10, 1))
            D.append(np.mean(ok[i] != ok[j]))
        n += 1
    return float(np.mean(Q)), float(np.mean(D))


def agreement_profile(ds):
    """distribution of how many of the 5 learners classify each instance correctly"""
    hist = np.zeros(len(MO) + 1, dtype=int)
    for fold in range(5):
        _, _, Pt, yt, _ = load(ds, fold)
        k = np.stack([Pt[m].argmax(-1) == yt for m in range(len(MO))]).sum(0)
        hist += np.bincount(k, minlength=len(MO) + 1)
    return (hist / hist.sum()).tolist(), hist.tolist()


# --------------------------------------------------------------- pipeline
def analyse(ds):
    per_fold, pooled, info = [], {}, []
    for fold in range(5):
        Pv, yv, Pt, yt, _ = load(ds, fold)
        preds = {MLAB[MO[m]]: Pt[m].argmax(-1) for m in range(len(MO))}
        rr, meta = all_rules(Pv, yv, Pt)
        preds.update(rr); info.append(meta)
        per_fold.append({k: full_metrics(yt, p) for k, p in preds.items()})
        for k, p in preds.items():
            pooled.setdefault(k, {"y": [], "p": []})
            pooled[k]["y"].append(yt); pooled[k]["p"].append(p)
    names = list(per_fold[0])
    summary = {k: {m: {"mean": float(np.mean([r[k][m] for r in per_fold])),
                       "std": float(np.std([r[k][m] for r in per_fold], ddof=1))}
                   for m in ["acc", "prec", "rec", "f1"]} for k in names}
    pooled_yp = {k: (np.concatenate(v["y"]), np.concatenate(v["p"])) for k, v in pooled.items()}
    pooled_m = {k: full_metrics(*v) for k, v in pooled_yp.items()}
    for k, v in pooled_yp.items():
        pooled_m[k]["cm"] = confusion_matrix(v[0], v[1]).tolist()
    yP, pP = pooled_yp[PROPOSED]
    tests, raw = {}, {}
    for k in names:
        if k == PROPOSED:
            continue
        y, p = pooled_yp[k]
        b, c, pv = mcnemar(y, pP, p)
        lo, hi = boot_ci(y, pP, p)
        tests[k] = {"b": b, "c": c, "p": pv, "d_f1": pooled_m[PROPOSED]["f1"] - pooled_m[k]["f1"],
                    "d_acc": pooled_m[PROPOSED]["acc"] - pooled_m[k]["acc"], "ci": [lo, hi]}
        raw[k] = pv
    for k, v in holm(raw).items():
        tests[k]["p_holm"] = v
    q, dis = q_and_disagreement(ds)
    prof, hist = agreement_profile(ds)
    return {"summary": summary, "pooled": pooled_m, "tests": tests, "info": info, "names": names,
            "Q": q, "disagreement": dis, "agreement_profile": prof, "agreement_hist": hist,
            "pooled_preds": {k: (v[0].tolist(), v[1].tolist()) for k, v in pooled_yp.items()}}


def regret_study(datasets, sizes=(20, 40, None), reps=10, seed=7):
    """Robustness of each rule to the size of the validation split used for weighting."""
    rng = np.random.default_rng(seed)
    table = {}
    for ds in datasets:
        for sz in sizes:
            acc = {}
            for _ in range(reps if sz else 1):
                for fold in range(5):
                    Pv, yv, Pt, yt, _ = load(ds, fold)
                    if sz:
                        sel = rng.choice(len(yv), size=sz, replace=False)
                        Pvr, yvr = Pv[:, sel], yv[sel]
                    else:
                        Pvr, yvr = Pv, yv
                    rr, _ = all_rules(Pvr, yvr, Pt)
                    for k, p in rr.items():
                        acc.setdefault(k, {"y": [], "p": []})
                        acc[k]["y"].append(yt); acc[k]["p"].append(p)
            tag = f"{DS_LAB[ds]}/n_val={sz if sz else 'full'}"
            table[tag] = {k: macro_f1(np.concatenate(v["y"]), np.concatenate(v["p"])) for k, v in acc.items()}
    rules = list(next(iter(table.values())))
    regret = {}
    for r in rules:
        d = [table[t][r] - table[t]["Uniform soft voting"] for t in table]
        regret[r] = {"mean": float(np.mean(d)), "worst": float(np.min(d)), "best": float(np.max(d)),
                     "per_setting": {t: table[t][r] - table[t]["Uniform soft voting"] for t in table}}
    return table, regret


def transfer_study(ds="evprimary-formal"):
    yx = np.load("experiments/probs/transfer_labels.npz")["y"]
    rows = {}
    for fold in range(5):
        Pv, yv, Pt, yt, Px = load(ds, fold)
        if Px is None:
            return None
        preds = {MLAB[MO[m]]: Px[m].argmax(-1) for m in range(len(MO))}
        rr, _ = all_rules(Pv, yv, Px)     # weights from the in-domain validation split
        preds.update(rr)
        for k, p in preds.items():
            rows.setdefault(k, []).append(full_metrics(yx, p))
    return {k: {m: {"mean": float(np.mean([r[m] for r in v])), "std": float(np.std([r[m] for r in v], ddof=1))}
                for m in ["acc", "f1"]} for k, v in rows.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="experiments/results"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    DATASETS = ["evprimary-formal", "evpublic", "evprimary-informal"]
    res = {}
    for ds in DATASETS:
        R = analyse(ds)
        res[ds] = R
        print(f"\n########## {DS_LAB[ds]} ({ds}) ##########")
        print(f"mean Q-statistic={R['Q']:.4f}  mean pairwise disagreement={R['disagreement']:.4f}")
        print("agreement profile (share of instances with k/5 learners correct):",
              " ".join(f"{k}:{100 * v:.1f}%" for k, v in enumerate(R["agreement_profile"])))
        for k in R["names"]:
            s, t = R["summary"][k], R["tests"].get(k)
            ex = "" if t is None else f" dF1={100 * t['d_f1']:+5.2f} CI=[{100 * t['ci'][0]:+5.2f},{100 * t['ci'][1]:+5.2f}] p={t['p']:.3f} pH={t['p_holm']:.3f}"
            print(f"  {k:28s} acc={100 * s['acc']['mean']:6.2f}+-{100 * s['acc']['std']:4.2f} "
                  f"f1={100 * s['f1']['mean']:6.2f}+-{100 * s['f1']['std']:4.2f} pooled={100 * R['pooled'][k]['f1']:6.2f}{ex}")
        lam = [i["lambda"] for i in R["info"]]; se = [i["se"] for i in R["info"]]
        spread = [max(i["F"]) - min(i["F"]) for i in R["info"]]
        print(f"  lambda per fold: {[round(x, 3) for x in lam]}  mean={np.mean(lam):.3f}")
        print(f"  SE(val F1) mean={np.mean(se):.4f}   inter-model val-F1 spread mean={np.mean(spread):.4f}")
        print(f"  raw log-odds weights   (fold0): {[round(x, 3) for x in R['info'][0]['w_raw']]}")
        print(f"  shrunk log-odds weights(fold0): {[round(x, 3) for x in R['info'][0]['w_shrunk']]}")

    print("\n########## validation-size robustness / regret vs uniform soft voting ##########")
    table, regret = regret_study(DATASETS)
    for r, v in sorted(regret.items(), key=lambda kv: -kv[1]["worst"]):
        print(f"  {r:28s} mean={100 * v['mean']:+5.2f}  worst={100 * v['worst']:+5.2f}  best={100 * v['best']:+5.2f}")
    print("\n  per-setting macro-F1:")
    for t, row in table.items():
        print(f"   {t:24s} " + "  ".join(f"{k.split()[0][:9]}={100 * v:.2f}" for k, v in row.items()))

    print("\n########## cross-corpus transfer: trained on EV-Norm, tested on EV-Public ##########")
    tr = transfer_study()
    for k, v in sorted(tr.items(), key=lambda kv: -kv[1]["f1"]["mean"]):
        print(f"  {k:28s} acc={100 * v['acc']['mean']:6.2f}+-{100 * v['acc']['std']:4.2f} f1={100 * v['f1']['mean']:6.2f}+-{100 * v['f1']['std']:4.2f}")

    json.dump({"per_dataset": {k: {kk: vv for kk, vv in v.items() if kk != "pooled_preds"} for k, v in res.items()},
               "regret_table": table, "regret": regret, "transfer": tr},
              open(os.path.join(a.out, "final.json"), "w"), indent=1)
    json.dump({k: v["pooled_preds"] for k, v in res.items()}, open(os.path.join(a.out, "pooled_preds.json"), "w"))
    print("\nwrote", os.path.join(a.out, "final.json"))
