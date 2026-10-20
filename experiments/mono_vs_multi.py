"""Pairwise significance tests between monolingual and multilingual encoders."""
import sys, json, itertools
import numpy as np
from scipy.stats import binomtest
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "experiments")
from final_analysis import load, macro_f1, MO, MLAB, holm

MONO = ["IndoBERT-p1", "IndoBERT-1.5G", "IndoRoBERTa-1.5G"]
MULTI = ["XLM-R", "mBERT"]
DS = {"EV-Norm": "evprimary-formal", "EV-Public": "evpublic", "EV-Raw": "evprimary-informal"}
out = {}
for lab, ds in DS.items():
    pooled = {MLAB[m]: {"y": [], "p": []} for m in MO}
    for fold in range(5):
        _, _, Pt, yt, _ = load(ds, fold)
        for i, m in enumerate(MO):
            pooled[MLAB[m]]["y"].append(yt)
            pooled[MLAB[m]]["p"].append(Pt[i].argmax(-1))
    P = {k: (np.concatenate(v["y"]), np.concatenate(v["p"])) for k, v in pooled.items()}
    f1 = {k: 100 * macro_f1(*v) for k, v in P.items()}
    raw, detail = {}, {}
    for a, b in itertools.product(MONO, MULTI):
        y, pa = P[a]
        _, pb = P[b]
        ra, rb = pa == y, pb == y
        nb, nc = int((ra & ~rb).sum()), int((~ra & rb).sum())
        pv = 1.0 if nb + nc == 0 else float(binomtest(nb, nb + nc, 0.5).pvalue)
        raw[f"{a} vs {b}"] = pv
        detail[f"{a} vs {b}"] = {"d_f1": f1[a] - f1[b], "b": nb, "c": nc, "p": pv}
    for k, v in holm(raw).items():
        detail[k]["p_holm"] = v
    out[lab] = {"f1": f1, "tests": detail}
    print(f"\n===== {lab} =====")
    for k, v in detail.items():
        sig = "SIG" if v["p_holm"] < 0.05 else "n.s."
        print(f"  {k:36s} dF1={v['d_f1']:+6.2f}  p={v['p']:.2e}  p_holm={v['p_holm']:.2e}  {sig}")
    allsig = all(v["p_holm"] < 0.05 for v in detail.values())
    pos = all(v["d_f1"] > 0 for v in detail.values())
    print(f"  --> all 6 mono>multi comparisons positive: {pos};  all significant: {allsig}")
json.dump(out, open("experiments/results/mono_vs_multi.json", "w"), indent=1)
print("\nwrote experiments/results/mono_vs_multi.json")
