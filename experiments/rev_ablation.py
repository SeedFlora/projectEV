"""Reliability-transform ablation (reviewer 2, second comment).

The log-odds weight borrows the weighted-majority optimum for independent voters, with validation
macro-F1 in place of accuracy. This ablation crosses the reliability statistic (macro-F1, accuracy),
the transform (log-odds, linear) and the shrinkage (none, paired-bootstrap James-Stein), with and
without the entropy confidence term (gamma = 1, 0), and scores every variant on all corpora and seeds
with the same bootstrap stream as rev_lib.pooled_run. Everything is fitted on validation data only.
"""
import os, sys, json
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rev_lib as L


def variant(stat, transform, shrink, gamma):
    def fit(c):
        R, Rb = (c.F, (lambda: c.Fb)) if stat == "F1" else (c.A, (lambda: c.Ab))
        Rs = L.shrink_js_boot_paired(R, Rb())[0] if shrink else R
        w = L.logodds(Rs) if transform == "log-odds" else np.asarray(Rs, float)
        if gamma:
            return lambda P: L._out(L.fuse(P, w, L.conf_entropy(P)))
        return lambda P: L._out(L.fuse(P, w))
    return fit


RULES = OrderedDict([("Uniform soft voting", L.r_uniform)] + [
    (f"gamma={g} {s} {t} {'shrunk' if sh else 'raw'}", variant(s, t, sh, g))
    for g in (0, 1) for s in ("F1", "accuracy") for t in ("log-odds", "linear") for sh in (False, True)])


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seeds = {ds: L.complete_seeds(ds) for ds in L.IN_DOMAIN}
    out = {"seeds_used": {L.DS_LAB[k]: v for k, v in seeds.items()}, "per_corpus": {}, "spread": {}}
    for ds in L.IN_DOMAIN:
        cv = {k: [] for k in RULES}
        for s in seeds[ds]:
            R = L.pooled_run(ds, s, rules=RULES, with_singles=False)
            for k in RULES:
                cv[k].append(L.summarise_folds(R["per_fold"][k])["mean"])
        out["per_corpus"][L.DS_LAB[ds]] = {k: {"mean": float(np.mean(v)), "per_seed": v,
                                               "diff_vs_uniform": float(np.mean(v) - np.mean(cv["Uniform soft voting"]))}
                                           for k, v in cv.items()}
    for g in (0, 1):
        names = [k for k in RULES if k.startswith(f"gamma={g}")]
        out["spread"][f"gamma={g}"] = {lab: float(max(pc[k]["mean"] for k in names) - min(pc[k]["mean"] for k in names))
                                       for lab, pc in out["per_corpus"].items()}
    json.dump(out, open(os.path.join(L.ROOT, "experiments", "results", "rev", "rev_ablation.json"), "w"), indent=1)
    print("seeds:", out["seeds_used"])
    print(f"{'variant':38s} " + " ".join(f"{lab:>16s}" for lab in out["per_corpus"]))
    for k in RULES:
        print(f"{k:38s} " + " ".join(f"{100*pc[k]['mean']:7.2f} ({100*pc[k]['diff_vs_uniform']:+5.2f})"
                                      for pc in out["per_corpus"].values()))
    print("max-min spread within each gamma (pp):", {g: {lab: round(100 * v, 2) for lab, v in d.items()}
                                                    for g, d in out["spread"].items()})


if __name__ == "__main__":
    main()
