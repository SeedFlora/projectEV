"""Offline ensemble analysis over saved per-fold posteriors.

Evaluates every fusion rule on identical posteriors (no retraining), runs the
reliability x confidence ablation grid, exact McNemar tests, bootstrap CIs on
pooled out-of-fold predictions, calibration evidence, cross-corpus transfer,
and writes LaTeX table bodies.
"""
import os, json, glob, math, argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.stats import binomtest

MODEL_ORDER = ["indobert-p1", "indobert-cahya", "indoroberta-cahya", "xlmr-base", "mbert-cased"]
MODEL_LABEL = {"indobert-p1": "IndoBERT-p1", "indobert-cahya": "IndoBERT-1.5G", "indoroberta-cahya": "IndoRoBERTa-1.5G",
               "xlmr-base": "XLM-R", "mbert-cased": "mBERT"}
PROPOSED = "RC-WSV (proposed)"
EPS = 1e-6

# ---------------------------------------------------------------- metrics


def metrics(y, p):
    acc = accuracy_score(y, p)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"acc": float(acc), "prec": float(pr), "rec": float(rc), "f1": float(f1)}


# ------------------------------------------------------- weighting components

def conf_entropy(P):
    """normalized-entropy confidence, P (M,n,C) -> (M,n)"""
    Pc = np.clip(P, EPS, 1.0)
    return 1.0 - (-(Pc * np.log(Pc)).sum(-1)) / math.log(P.shape[-1])


def conf_maxprob(P):
    """max-softmax confidence rescaled to [0,1]: (max p - 1/C)/(1 - 1/C)"""
    C = P.shape[-1]
    return (P.max(-1) - 1.0 / C) / (1.0 - 1.0 / C)


def rel_linear(f):
    return np.asarray(f, float)


def rel_logodds(f):
    f = np.clip(np.asarray(f, float), 0.5 + EPS, 1 - EPS)
    return np.log(f / (1 - f))


def rel_softmax(f, beta):
    z = beta * np.asarray(f, float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fuse(P, w_static=None, conf=None):
    """P (M,n,C); w_static (M,) or None; conf callable or None -> fused posterior (n,C)"""
    M, n, _ = P.shape
    w = np.ones(M) if w_static is None else np.asarray(w_static, float)
    W = np.repeat(w[:, None], n, axis=1)
    if conf is not None:
        W = W * conf(P)
    W = W / np.clip(W.sum(0, keepdims=True), EPS, None)
    return (W[:, :, None] * P).sum(0)


def majority(P, tiebreak):
    votes = P.argmax(-1)
    counts = np.stack([(votes == c).sum(0) for c in range(P.shape[-1])], -1)
    pred = counts.argmax(-1)
    tie = counts.max(-1) == np.sort(counts, -1)[:, -2]
    pred[tie] = tiebreak.argmax(-1)[tie]
    return pred


def all_rules(Pv, yv, Pt, f_val):
    """Every fusion rule, evaluated on the same test posteriors. Returns name -> predictions."""
    out = {}
    uni = fuse(Pt)
    out["Majority voting"] = majority(Pt, uni)
    out["Uniform soft voting"] = uni.argmax(-1)
    out["Linear-F1 weighting"] = fuse(Pt, rel_linear(f_val)).argmax(-1)
    best_b, best_s = 0.0, -1.0
    for b in [0, 1, 2, 5, 10, 20, 50, 100]:
        s = metrics(yv, fuse(Pv, rel_softmax(f_val, b)).argmax(-1))["f1"]
        if s > best_s + 1e-12:
            best_s, best_b = s, b
    out["Softmax-F1 weighting"] = fuse(Pt, rel_softmax(f_val, best_b)).argmax(-1)
    out["Log-odds weighting"] = fuse(Pt, rel_logodds(f_val)).argmax(-1)
    out["Max-prob confidence only"] = fuse(Pt, None, conf_maxprob).argmax(-1)
    out["Entropy confidence only"] = fuse(Pt, None, conf_entropy).argmax(-1)
    out["Linear-F1 x max-prob"] = fuse(Pt, rel_linear(f_val), conf_maxprob).argmax(-1)
    out["Linear-F1 x entropy"] = fuse(Pt, rel_linear(f_val), conf_entropy).argmax(-1)
    out["Log-odds x max-prob"] = fuse(Pt, rel_logodds(f_val), conf_maxprob).argmax(-1)
    out[PROPOSED] = fuse(Pt, rel_logodds(f_val), conf_entropy).argmax(-1)
    out["_beta"] = best_b
    return out


# ------------------------------------------------------------- significance

def mcnemar_exact(y, p1, p2):
    r1, r2 = p1 == y, p2 == y
    b, c = int((r1 & ~r2).sum()), int((~r1 & r2).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def _macro_f1_fast(y, p):
    """binary macro-F1 without sklearn overhead (used in the bootstrap loop)"""
    f = 0.0
    for c in (0, 1):
        tp = np.count_nonzero((p == c) & (y == c))
        fp = np.count_nonzero((p == c) & (y != c))
        fn = np.count_nonzero((p != c) & (y == c))
        d = 2 * tp + fp + fn
        f += (2.0 * tp / d) if d else 0.0
    return f / 2.0


def bootstrap_diff(y, p1, p2, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    ys, a, b = y[idx], p1[idx], p2[idx]
    d = np.array([_macro_f1_fast(ys[i], a[i]) - _macro_f1_fast(ys[i], b[i]) for i in range(n_boot)])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        v = min(1.0, max(prev, (m - i) * p))
        adj[k] = v
        prev = v
    return adj


# ------------------------------------------------------------------- loading

def load_fold(d):
    Pv, Pt, Px = [], [], []
    yv = yt = None
    for m in MODEL_ORDER:
        z = np.load(os.path.join(d, f"{m}.npz"))
        Pv.append(z["val_probs"])
        Pt.append(z["test_probs"])
        if "transfer_probs" in z:
            Px.append(z["transfer_probs"])
        yv, yt = z["y_val"], z["y_test"]
    return np.stack(Pv), np.stack(Pt), (np.stack(Px) if len(Px) == len(MODEL_ORDER) else None), yv, yt


def analyze(root, ds, y_transfer=None):
    folds = [d for d in sorted(glob.glob(os.path.join(root, ds, "fold*")))
             if all(os.path.exists(os.path.join(d, f"{m}.npz")) for m in MODEL_ORDER)]
    if not folds:
        return None
    per_fold, pooled, betas, xfer_fold = [], {}, [], []
    conf_ev = {m: {"correct": [], "wrong": []} for m in MODEL_ORDER}
    for d in folds:
        Pv, Pt, Px, yv, yt = load_fold(d)
        f_val = [metrics(yv, Pv[m].argmax(-1))["f1"] for m in range(len(MODEL_ORDER))]
        preds = {}
        for m, name in enumerate(MODEL_ORDER):
            preds[MODEL_LABEL[name]] = Pt[m].argmax(-1)
            c = conf_entropy(Pt[m:m + 1])[0]
            ok = Pt[m].argmax(-1) == yt
            conf_ev[name]["correct"].extend(c[ok].tolist())
            conf_ev[name]["wrong"].extend(c[~ok].tolist())
        preds["Best single (val-selected)"] = Pt[int(np.argmax(f_val))].argmax(-1)
        rules = all_rules(Pv, yv, Pt, f_val)
        betas.append(rules.pop("_beta"))
        preds.update(rules)
        res = {k: metrics(yt, p) for k, p in preds.items()}
        for k, p in preds.items():
            pooled.setdefault(k, {"y": [], "p": []})
            pooled[k]["y"].append(yt)
            pooled[k]["p"].append(p)
        res["_val_f1"] = dict(zip(MODEL_ORDER, f_val))
        res["_best_val_model"] = MODEL_ORDER[int(np.argmax(f_val))]
        lo = rel_logodds(f_val)
        res["_w_logodds"] = (lo / lo.sum()).tolist()
        res["_w_linear"] = (np.asarray(f_val) / np.sum(f_val)).tolist()
        per_fold.append(res)
        if Px is not None and y_transfer is not None:
            xr = {MODEL_LABEL[MODEL_ORDER[m]]: Px[m].argmax(-1) for m in range(len(MODEL_ORDER))}
            xr["Best single (val-selected)"] = Px[int(np.argmax(f_val))].argmax(-1)
            xr.update({k: v for k, v in all_rules(Px, y_transfer, Px, f_val).items() if k != "_beta"})
            # rules must not peek at transfer labels: recompute without the val-tuned beta rule
            xr["Softmax-F1 weighting"] = fuse(Px, rel_softmax(f_val, betas[-1])).argmax(-1)
            xfer_fold.append({k: metrics(y_transfer, p) for k, p in xr.items()})
    names = [k for k in per_fold[0] if not k.startswith("_")]
    summary = {k: {met: {"mean": float(np.mean([r[k][met] for r in per_fold])),
                         "std": float(np.std([r[k][met] for r in per_fold], ddof=1)),
                         "folds": [r[k][met] for r in per_fold]}
                   for met in ["acc", "prec", "rec", "f1"]} for k in names}
    pooled_m, pooled_yp = {}, {}
    for k, v in pooled.items():
        y, p = np.concatenate(v["y"]), np.concatenate(v["p"])
        pooled_m[k] = metrics(y, p)
        pooled_m[k]["cm"] = confusion_matrix(y, p).tolist()
        pooled_yp[k] = (y, p)
    yP, pP = pooled_yp[PROPOSED]
    tests, raw_p = {}, {}
    for k in names:
        if k == PROPOSED:
            continue
        y, p = pooled_yp[k]
        b, c, pv = mcnemar_exact(y, pP, p)
        lo_, hi_ = bootstrap_diff(y, pP, p)
        tests[k] = {"b": b, "c": c, "p_value": pv, "d_f1": pooled_m[PROPOSED]["f1"] - pooled_m[k]["f1"],
                    "d_acc": pooled_m[PROPOSED]["acc"] - pooled_m[k]["acc"], "ci95": [lo_, hi_]}
        raw_p[k] = pv
    for k, v in holm(raw_p).items():
        tests[k]["p_holm"] = v
    xfer = None
    if xfer_fold:
        keys = list(xfer_fold[0])
        xfer = {k: {met: {"mean": float(np.mean([r[k][met] for r in xfer_fold])),
                          "std": float(np.std([r[k][met] for r in xfer_fold], ddof=1))}
                    for met in ["acc", "f1"]} for k in keys}
    conf_summary = {m: {"conf_correct": float(np.mean(v["correct"])),
                        "conf_wrong": float(np.mean(v["wrong"])) if v["wrong"] else float("nan"),
                        "n_wrong": len(v["wrong"]), "n_correct": len(v["correct"])} for m, v in conf_ev.items()}
    return {"n_folds": len(folds), "names": names, "per_fold": per_fold, "summary": summary, "pooled": pooled_m,
            "tests": tests, "confidence_evidence": conf_summary, "betas": betas, "transfer": xfer,
            "_pooled_preds": {k: (v[0].tolist(), v[1].tolist()) for k, v in pooled_yp.items()}}


# ------------------------------------------------------------------- output

def pm(m):
    return f"{100 * m['mean']:.2f}\\,$\\pm$\\,{100 * m['std']:.2f}"


def write_tables(R, ds, outdir):
    L, s = [], R["summary"]
    singles = [MODEL_LABEL[m] for m in MODEL_ORDER]
    L.append(f"%% ---- single learners: {ds}")
    for k in singles:
        L.append(f"{k} & {pm(s[k]['acc'])} & {pm(s[k]['prec'])} & {pm(s[k]['rec'])} & {pm(s[k]['f1'])} \\\\")
    L.append(f"%% ---- fusion rules: {ds}")
    for k in R["names"]:
        if k in singles:
            continue
        t = R["tests"].get(k)
        sig = "--" if t is None else (f"{t['p_holm']:.3f}" if t["p_holm"] >= 0.001 else "$<$0.001")
        L.append(f"{k} & {pm(s[k]['acc'])} & {pm(s[k]['f1'])} & {100 * R['pooled'][k]['f1']:.2f} & {sig} \\\\")
    L.append(f"%% ---- mcnemar vs proposed: {ds}")
    for k, t in R["tests"].items():
        L.append(f"{k} & {t['b']} & {t['c']} & {100 * t['d_f1']:+.2f} & [{100 * t['ci95'][0]:+.2f}, {100 * t['ci95'][1]:+.2f}] & "
                 f"{t['p_value']:.3f} & {t['p_holm']:.3f} \\\\")
    L.append(f"%% ---- confidence evidence: {ds}")
    for m, v in R["confidence_evidence"].items():
        L.append(f"{MODEL_LABEL[m]} & {v['conf_correct']:.3f} & {v['conf_wrong']:.3f} & {v['n_wrong']} \\\\")
    if R["transfer"]:
        L.append(f"%% ---- cross-corpus transfer: {ds}")
        for k, v in R["transfer"].items():
            L.append(f"{k} & {pm(v['acc'])} & {pm(v['f1'])} \\\\")
    open(os.path.join(outdir, f"tables_{ds}.tex"), "w").write("\n".join(L) + "\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/probs")
    ap.add_argument("--out", default="experiments/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    yx = np.load(os.path.join(a.root, "transfer_labels.npz"))["y"]
    out = {}
    for ds in ["evprimary-formal", "evpublic", "evprimary-informal"]:
        R = analyze(a.root, ds, y_transfer=yx if ds == "evprimary-formal" else None)
        if R is None:
            print("no complete folds for", ds)
            continue
        out[ds] = R
        print(f"\n===== {ds}: {R['n_folds']} folds =====")
        for k in R["names"]:
            sm = R["summary"][k]
            t = R["tests"].get(k)
            extra = "" if t is None else f"  dF1={100 * t['d_f1']:+.2f} p={t['p_value']:.4f} p_holm={t['p_holm']:.4f} b/c={t['b']}/{t['c']} CI=[{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]"
            print(f"{k:30s} acc={100 * sm['acc']['mean']:6.2f}+-{100 * sm['acc']['std']:4.2f} f1={100 * sm['f1']['mean']:6.2f}+-{100 * sm['f1']['std']:4.2f} pooled_f1={100 * R['pooled'][k]['f1']:6.2f}{extra}")
        print("beta per fold:", R["betas"], " best val model:", [r["_best_val_model"] for r in R["per_fold"]])
        print("log-odds weights:", [[round(x, 3) for x in r["_w_logodds"]] for r in R["per_fold"]])
        print("linear weights  :", [[round(x, 3) for x in r["_w_linear"]] for r in R["per_fold"]])
        print("confidence:", {m: (round(v["conf_correct"], 3), round(v["conf_wrong"], 3), v["n_wrong"]) for m, v in R["confidence_evidence"].items()})
        if R["transfer"]:
            print("--- cross-corpus transfer (train primary -> test public EV corpus) ---")
            for k, v in R["transfer"].items():
                print(f"   {k:30s} acc={100 * v['acc']['mean']:6.2f}+-{100 * v['acc']['std']:4.2f} f1={100 * v['f1']['mean']:6.2f}+-{100 * v['f1']['std']:4.2f}")
        write_tables(R, ds, a.out)
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "_pooled_preds"} for k, v in out.items()},
              open(os.path.join(a.out, "results.json"), "w"), indent=1)
    json.dump({k: v["_pooled_preds"] for k, v in out.items()}, open(os.path.join(a.out, "pooled_preds.json"), "w"))
    print("\nwrote", os.path.join(a.out, "results.json"))


if __name__ == "__main__":
    main()
