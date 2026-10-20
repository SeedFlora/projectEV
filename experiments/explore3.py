"""Exploration round 3: multi-view ensembling.

The two renderings of the same corpus (raw informal text vs. lexicon-normalized
"formalized" text) are tokenized differently by every model, so learners trained on
different views should make less correlated errors than learners trained on the same
view. Folds are identical across views, so the posteriors fuse directly.
"""
import os, glob, math, itertools, json
import numpy as np

MO = ["indobert-p1", "indobert-cahya", "indoroberta-cahya", "xlmr-base", "mbert-cased"]
VIEWS = {"F": "evprimary-formal", "I": "evprimary-informal"}
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


def acc(y, p):
    return float(np.mean(y == p))


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


def load_fold(fold):
    """returns members dict key -> (val_probs, test_probs), plus yv, yt"""
    mem, yv, yt = {}, None, None
    for v, ds in VIEWS.items():
        for m in MO:
            z = np.load(f"experiments/probs/{ds}/fold{fold}/{m}.npz")
            mem[f"{m}|{v}"] = (z["val_probs"], z["test_probs"])
            yv, yt = z["y_val"], z["y_test"]
    return mem, yv, yt


def q_stat(ok_i, ok_j):
    n11 = np.count_nonzero(ok_i & ok_j); n00 = np.count_nonzero(~ok_i & ~ok_j)
    n10 = np.count_nonzero(ok_i & ~ok_j); n01 = np.count_nonzero(~ok_i & ok_j)
    return (n11 * n00 - n01 * n10) / max(n11 * n00 + n01 * n10, 1)


def run():
    pools = {
        "single-view F (5)": [f"{m}|F" for m in MO],
        "single-view I (5)": [f"{m}|I" for m in MO],
        "multi-view (10)": [f"{m}|{v}" for v in VIEWS for m in MO],
        "multi-view mono-only (6)": [f"{m}|{v}" for v in VIEWS for m in MO[:3]],
        "multi-view top2 (4)": [f"{m}|{v}" for v in VIEWS for m in MO[:2]],
    }
    rules = {}
    qstats = {"same-model, diff view": [], "diff model, same view": [], "diff model, diff view": []}

    def add(name, y, p):
        rules.setdefault(name, {"f1": [], "acc": [], "y": [], "p": []})
        rules[name]["f1"].append(macro_f1(y, p)); rules[name]["acc"].append(acc(y, p))
        rules[name]["y"].append(y); rules[name]["p"].append(p)

    for fold in range(5):
        mem, yv, yt = load_fold(fold)
        ok = {k: (v[1].argmax(-1) == yt) for k, v in mem.items()}
        keys = list(mem)
        for a, b in itertools.combinations(keys, 2):
            ma, va = a.split("|"); mb, vb = b.split("|")
            q = q_stat(ok[a], ok[b])
            if ma == mb and va != vb:
                qstats["same-model, diff view"].append(q)
            elif ma != mb and va == vb:
                qstats["diff model, same view"].append(q)
            else:
                qstats["diff model, diff view"].append(q)
        for pname, mems in pools.items():
            Pv = np.stack([mem[k][0] for k in mems])
            Pt = np.stack([mem[k][1] for k in mems])
            fv = np.array([macro_f1(yv, Pv[i].argmax(-1)) for i in range(len(mems))])
            ct = conf_entropy(Pt); cv = conf_entropy(Pv)
            add(f"{pname} | uniform", yt, fuse(Pt).argmax(-1))
            add(f"{pname} | linearF1", yt, fuse(Pt, fv).argmax(-1))
            add(f"{pname} | logodds", yt, fuse(Pt, logodds(fv)).argmax(-1))
            add(f"{pname} | logodds x conf", yt, fuse(Pt, logodds(fv), ct, 1).argmax(-1))
            g = max([0, 0.5, 1, 2, 3, 5], key=lambda gg: macro_f1(yv, fuse(Pv, logodds(fv), cv, gg).argmax(-1)))
            add(f"{pname} | logodds x conf^g*", yt, fuse(Pt, logodds(fv), ct, g).argmax(-1))
            b = max([0, 2, 5, 10, 20, 50, 100], key=lambda bb: macro_f1(yv, fuse(Pv, softmax_w(fv, bb)).argmax(-1)))
            add(f"{pname} | softmaxF1(b*)", yt, fuse(Pt, softmax_w(fv, b)).argmax(-1))
        # best single member overall
        fv_all = {k: macro_f1(yv, mem[k][0].argmax(-1)) for k in mem}
        bk = max(fv_all, key=fv_all.get)
        add("best single member (val)", yt, mem[bk][1].argmax(-1))
        for k in mem:
            add(f"member {k}", yt, mem[k][1].argmax(-1))

    out = {}
    for k, v in rules.items():
        y, p = np.concatenate(v["y"]), np.concatenate(v["p"])
        out[k] = {"f1_mean": float(np.mean(v["f1"])), "f1_std": float(np.std(v["f1"], ddof=1)),
                  "acc_mean": float(np.mean(v["acc"])), "acc_std": float(np.std(v["acc"], ddof=1)),
                  "pooled_f1": macro_f1(y, p), "pooled_acc": acc(y, p),
                  "pooled_y": y.tolist(), "pooled_p": p.tolist()}
    base = out["single-view F (5) | uniform"]["pooled_f1"]
    print("=== Q-statistics by member relationship (lower = more diverse) ===")
    for k, v in qstats.items():
        print(f"  {k:26s} mean Q={np.mean(v):.4f}  (n pairs={len(v)})")
    print("\n=== fusion results (sorted by pooled macro-F1) ===")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["pooled_f1"]):
        if k.startswith("member "):
            continue
        print(f"{k:44s} f1={100 * v['f1_mean']:6.2f}+-{100 * v['f1_std']:4.2f} pooled={100 * v['pooled_f1']:6.2f} "
              f"acc={100 * v['acc_mean']:6.2f} vs-F-uniform={100 * (v['pooled_f1'] - base):+5.2f}")
    print("\n=== individual members ===")
    for k, v in sorted([(k, v) for k, v in out.items() if k.startswith("member ")], key=lambda kv: -kv[1]["pooled_f1"]):
        print(f"{k:44s} f1={100 * v['f1_mean']:6.2f}+-{100 * v['f1_std']:4.2f} pooled={100 * v['pooled_f1']:6.2f}")
    json.dump({"rules": out, "qstats": {k: float(np.mean(v)) for k, v in qstats.items()}},
              open("experiments/results/explore3.json", "w"))


if __name__ == "__main__":
    run()
