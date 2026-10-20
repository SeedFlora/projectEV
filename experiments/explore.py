"""Design-space exploration for the fusion rule.

Every candidate rule sees ONLY validation data when choosing its hyper-parameters;
test posteriors are touched once, for scoring. Reports per-fold and pooled macro-F1
so we can see which family (if any) genuinely beats uniform soft voting.
"""
import os, glob, math, json, itertools
import numpy as np

MODEL_ORDER = ["indobert-p1", "indobert-cahya", "indoroberta-cahya", "xlmr-base", "mbert-cased"]
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


def load(root, ds):
    out = []
    for d in sorted(glob.glob(os.path.join(root, ds, "fold*"))):
        Pv, Pt = [], []
        yv = yt = None
        ok = True
        for m in MODEL_ORDER:
            f = os.path.join(d, f"{m}.npz")
            if not os.path.exists(f):
                ok = False; break
            z = np.load(f)
            Pv.append(z["val_probs"]); Pt.append(z["test_probs"]); yv, yt = z["y_val"], z["y_test"]
        if ok:
            out.append((np.stack(Pv), yv, np.stack(Pt), yt))
    return out


def softmax_w(f, beta):
    z = beta * np.asarray(f, float); z -= z.max(); e = np.exp(z); return e / e.sum()


def logodds(f):
    f = np.clip(np.asarray(f, float), 0.5 + EPS, 1 - EPS); return np.log(f / (1 - f))


BETAS = [0, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200]
GAMMAS = [0, 0.5, 1, 2, 3, 5, 8, 12]


def evaluate(folds, name_filter=None):
    """returns rule -> (per-fold test f1 list, pooled f1)"""
    res, pooled = {}, {}

    def add(name, yt, pred):
        res.setdefault(name, []).append(macro_f1(yt, pred))
        pooled.setdefault(name, {"y": [], "p": []})
        pooled[name]["y"].append(yt); pooled[name]["p"].append(pred)

    for Pv, yv, Pt, yt in folds:
        M = len(MODEL_ORDER)
        fv = np.array([macro_f1(yv, Pv[m].argmax(-1)) for m in range(M)])
        cv, ct = conf_entropy(Pv), conf_entropy(Pt)

        add("single:best-val", yt, Pt[int(fv.argmax())].argmax(-1))
        for m in range(M):
            add(f"single:{MODEL_ORDER[m]}", yt, Pt[m].argmax(-1))
        add("uniform-soft", yt, fuse(Pt).argmax(-1))
        add("linearF1", yt, fuse(Pt, fv).argmax(-1))
        add("logodds", yt, fuse(Pt, logodds(fv)).argmax(-1))
        add("uniform x conf^1", yt, fuse(Pt, None, ct, 1).argmax(-1))
        add("logodds x conf^1", yt, fuse(Pt, logodds(fv), ct, 1).argmax(-1))

        # (A) beta-only, tuned on validation
        bb = max(BETAS, key=lambda b: macro_f1(yv, fuse(Pv, softmax_w(fv, b)).argmax(-1)))
        add("A:softmaxF1(beta*)", yt, fuse(Pt, softmax_w(fv, bb)).argmax(-1))

        # (B) gamma-only, tuned on validation
        gg = max(GAMMAS, key=lambda g: macro_f1(yv, fuse(Pv, None, cv, g).argmax(-1)))
        add("B:uniform x conf^gamma*", yt, fuse(Pt, None, ct, gg).argmax(-1))

        # (C) joint (beta, gamma) grid on validation
        bg = max(itertools.product(BETAS, GAMMAS),
                 key=lambda t: (macro_f1(yv, fuse(Pv, softmax_w(fv, t[0]), cv, t[1]).argmax(-1)), -t[0], -t[1]))
        add("C:softmaxF1(beta*) x conf^gamma*", yt, fuse(Pt, softmax_w(fv, bg[0]), ct, bg[1]).argmax(-1))

        # (D) exhaustive subset selection on validation, then uniform soft vote over the subset
        best_sub, best_s = None, -1
        for r in range(1, M + 1):
            for sub in itertools.combinations(range(M), r):
                s = macro_f1(yv, fuse(Pv[list(sub)]).argmax(-1))
                if s > best_s + 1e-12 or (abs(s - best_s) <= 1e-12 and best_sub is not None and len(sub) > len(best_sub)):
                    best_s, best_sub = s, sub
        add("D:subset-soft(val)", yt, fuse(Pt[list(best_sub)]).argmax(-1))
        add("D:subset-size", yt, yt * 0 + len(best_sub))  # placeholder, filtered out later

        # (E) subset selection + confidence weighting
        gsub = max(GAMMAS, key=lambda g: macro_f1(yv, fuse(Pv[list(best_sub)], None, cv[list(best_sub)], g).argmax(-1)))
        add("E:subset x conf^gamma*", yt, fuse(Pt[list(best_sub)], None, ct[list(best_sub)], gsub).argmax(-1))

        # (F) top-k by validation F1 (k tuned on validation)
        order = np.argsort(-fv)
        bk = max(range(1, M + 1), key=lambda k: macro_f1(yv, fuse(Pv[order[:k]]).argmax(-1)))
        add("F:top-k(val)", yt, fuse(Pt[order[:bk]]).argmax(-1))

        # (G) dynamic classifier selection: most confident model per instance
        add("G:most-confident", yt, Pt[ct.argmax(0), np.arange(Pt.shape[1])].argmax(-1))

        # (H) reliability-thresholded: drop models below (max F1 - delta), delta tuned on val
        bd = max([0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 1.0],
                 key=lambda dl: macro_f1(yv, fuse(Pv[fv >= fv.max() - dl]).argmax(-1)))
        keep = fv >= fv.max() - bd
        add("H:threshold(val)", yt, fuse(Pt[keep]).argmax(-1))

        # (I) H + confidence
        gh = max(GAMMAS, key=lambda g: macro_f1(yv, fuse(Pv[keep], None, cv[keep], g).argmax(-1)))
        add("I:threshold x conf^gamma*", yt, fuse(Pt[keep], None, ct[keep], gh).argmax(-1))

        # (J) oracle upper bound: best subset chosen on TEST (not a usable method, reference only)
        bo, bos = None, -1
        for r in range(1, M + 1):
            for sub in itertools.combinations(range(M), r):
                s = macro_f1(yt, fuse(Pt[list(sub)]).argmax(-1))
                if s > bos: bos, bo = s, sub
        add("J:oracle-subset(TEST)", yt, fuse(Pt[list(bo)]).argmax(-1))

    final = {}
    for k, v in res.items():
        if k == "D:subset-size":
            continue
        y = np.concatenate(pooled[k]["y"]); p = np.concatenate(pooled[k]["p"])
        final[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "pooled": macro_f1(y, p), "folds": v}
    return final


def diversity(folds):
    """pairwise disagreement and Q-statistic between base learners on test folds"""
    M = len(MODEL_ORDER)
    dis = np.zeros((M, M)); q = np.zeros((M, M)); n = 0
    for _, _, Pt, yt in folds:
        ok = np.stack([Pt[m].argmax(-1) == yt for m in range(M)])
        for i in range(M):
            for j in range(M):
                if i >= j: continue
                dis[i, j] += np.mean(ok[i] != ok[j])
                n11 = np.count_nonzero(ok[i] & ok[j]); n00 = np.count_nonzero(~ok[i] & ~ok[j])
                n10 = np.count_nonzero(ok[i] & ~ok[j]); n01 = np.count_nonzero(~ok[i] & ok[j])
                q[i, j] += (n11 * n00 - n01 * n10) / max(n11 * n00 + n01 * n10, 1)
        n += 1
    return dis / n, q / n


if __name__ == "__main__":
    summary = {}
    for ds in ["evprimary-formal", "evpublic", "evprimary-informal"]:
        folds = load("experiments/probs", ds)
        if not folds:
            continue
        r = evaluate(folds)
        summary[ds] = r
        print(f"\n===== {ds} ({len(folds)} folds) =====")
        base = r["uniform-soft"]["pooled"]
        for k, v in sorted(r.items(), key=lambda kv: -kv[1]["pooled"]):
            print(f"{k:36s} mean={100 * v['mean']:6.2f}+-{100 * v['std']:4.2f}  pooled={100 * v['pooled']:6.2f}  vs-uniform={100 * (v['pooled'] - base):+5.2f}")
        d, q = diversity(folds)
        iu = np.triu_indices(len(MODEL_ORDER), 1)
        print(f"mean pairwise disagreement={d[iu].mean():.4f}  mean Q-statistic={q[iu].mean():.4f}")
        print("disagreement matrix:\n", np.round(d, 4))
    json.dump(summary, open("experiments/results/explore.json", "w"), indent=1)
