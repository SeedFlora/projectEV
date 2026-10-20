"""Figures and the error-concentration analysis for the paper."""
import os, sys, math, json, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MO = ["indobert-p1", "indobert-cahya", "indoroberta-cahya", "xlmr-base", "mbert-cased"]
MLAB = {"indobert-p1": "IndoBERT-p1", "indobert-cahya": "IndoBERT-1.5G", "indoroberta-cahya": "IndoRoBERTa-1.5G",
        "xlmr-base": "XLM-R", "mbert-cased": "mBERT"}
SHORT = {"indobert-p1": "IB-p1", "indobert-cahya": "IB-1.5G", "indoroberta-cahya": "IR-1.5G",
         "xlmr-base": "XLM-R", "mbert-cased": "mBERT"}
DS = {"evprimary-formal": "EV-Norm", "evpublic": "EV-Public", "evprimary-informal": "EV-Raw"}
OUT = "paper"
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
                     "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})


def conf_entropy(P):
    Pc = np.clip(P, 1e-6, 1.0)
    return 1.0 - (-(Pc * np.log(Pc)).sum(-1)) / math.log(P.shape[-1])


def load(ds, fold):
    Pv, Pt = [], []
    yv = yt = ti = None
    for m in MO:
        z = np.load(f"experiments/probs/{ds}/fold{fold}/{m}.npz")
        Pv.append(z["val_probs"]); Pt.append(z["test_probs"])
        yv, yt, ti = z["y_val"], z["y_test"], z["test_idx"]
    return np.stack(Pv), yv, np.stack(Pt), yt, ti


# ------------------------------------------------- error concentration + label audit
def error_analysis():
    df = pd.read_excel("hf_space/data.xlsx")
    relabeled = ((df.label_old.notna()) & (df.label_old != df.label)).to_numpy()
    unlabeled_before = df.label_old.isna().to_numpy()
    rows = {}
    for ds in ["evprimary-formal", "evprimary-informal"]:
        n_correct = np.zeros(len(df), dtype=int)
        seen = np.zeros(len(df), dtype=bool)
        for fold in range(5):
            _, _, Pt, yt, ti = load(ds, fold)
            k = np.stack([Pt[m].argmax(-1) == yt for m in range(len(MO))]).sum(0)
            n_correct[ti] = k; seen[ti] = True
        assert seen.all()
        allwrong = n_correct == 0
        allright = n_correct == len(MO)
        rows[DS[ds]] = {
            "all_wrong_pct": 100 * allwrong.mean(),
            "all_right_pct": 100 * allright.mean(),
            "contested_pct": 100 * ((n_correct > 0) & (n_correct < len(MO))).mean(),
            "relabeled_rate_overall": 100 * relabeled.mean(),
            "relabeled_rate_in_allwrong": 100 * relabeled[allwrong].mean(),
            "relabeled_rate_in_allright": 100 * relabeled[allright].mean(),
            "n_allwrong": int(allwrong.sum()),
            "risk_ratio": float(relabeled[allwrong].mean() / max(relabeled[allright].mean(), 1e-9)),
            "mean_words_allwrong": float(df.clean_text[allwrong].str.split().str.len().mean()),
            "mean_words_allright": float(df.clean_text[allright].str.split().str.len().mean()),
            "unlabeled_before_in_allwrong": 100 * unlabeled_before[allwrong].mean(),
            "unlabeled_before_overall": 100 * unlabeled_before.mean(),
        }
        if ds == "evprimary-formal":
            rows["_examples"] = df.clean_text[allwrong].head(6).tolist()
    return rows


# --------------------------------------------------------------------- figures
def fig_confidence():
    """confidence separation between correct and incorrect predictions (column width)"""
    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    ds = "evprimary-formal"
    cc, cw = {m: [] for m in MO}, {m: [] for m in MO}
    for fold in range(5):
        _, _, Pt, yt, _ = load(ds, fold)
        for i, m in enumerate(MO):
            c = conf_entropy(Pt[i:i + 1])[0]
            ok = Pt[i].argmax(-1) == yt
            cc[m].extend(c[ok]); cw[m].extend(c[~ok])
    x = np.arange(len(MO)); w = 0.38
    ax.bar(x - w / 2, [np.mean(cc[m]) for m in MO], w, label="correct", color="#2b7a4b",
           yerr=[np.std(cc[m]) for m in MO], capsize=2, error_kw={"lw": 0.6})
    ax.bar(x + w / 2, [np.mean(cw[m]) for m in MO], w, label="incorrect", color="#b3352c",
           yerr=[np.std(cw[m]) for m in MO], capsize=2, error_kw={"lw": 0.6})
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in MO], rotation=18, ha="right")
    ax.set_ylim(0, 1.30); ax.set_ylabel("confidence $c_m(x)$")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.25)
    fig.savefig(f"{OUT}/fig_confidence.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_diversity_and_profile():
    """agreement profile at column width; Q-statistics are reported in the text"""
    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    w, xs = 0.26, np.arange(len(MO) + 1)
    for i, ds in enumerate(["evprimary-formal", "evpublic", "evprimary-informal"]):
        hist = np.zeros(len(MO) + 1); tot = 0
        for fold in range(5):
            _, _, Pt, yt, _ = load(ds, fold)
            k = np.stack([Pt[m].argmax(-1) == yt for m in range(len(MO))]).sum(0)
            hist += np.bincount(k, minlength=len(MO) + 1); tot += len(yt)
        ax.bar(xs + (i - 1) * w, 100 * hist / tot, w, label=DS[ds],
               color=["#1f4e79", "#c07a1a", "#4a7c59"][i])
    ax.set_xlabel("learners classifying the instance correctly")
    ax.set_ylabel("share of instances (\%)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_xticks(xs); ax.set_ylim(0, 88)
    fig.tight_layout(pad=0.25)
    fig.savefig(f"{OUT}/fig_profile.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_regret():
    """regret of each rule relative to uniform soft voting, across the 9 settings"""
    R = json.load(open("experiments/results/final.json"))["regret"]
    order = ["Best single (val-selected)", "Majority voting", "Softmax-F1 weighting", "Entropy confidence",
             "Max-prob confidence", "Linear-F1 weighting", "Linear-F1 x max-prob", "Log-odds weighting",
             "Log-odds x max-prob", "Log-odds x entropy", "SA-WSV (proposed)"]
    order = [o for o in order if o in R]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y = np.arange(len(order))
    for i, k in enumerate(order):
        v = R[k]
        ax.plot([100 * v["worst"], 100 * v["best"]], [i, i], color="#999", lw=1.2, zorder=1)
        ax.scatter([100 * v["worst"], 100 * v["best"]], [i, i], s=8, color="#999", zorder=2)
        ax.scatter([100 * v["mean"]], [i], s=26, color="#b3352c" if "proposed" in k else "#1f4e79",
                   marker="D" if "proposed" in k else "o", zorder=3)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace(" x ", "$\\times$").replace(" (proposed)", "*").replace(" (val-selected)", "") for k in order], fontsize=7)
    ax.set_xlabel("macro-F1 difference vs. uniform soft voting (pp)")
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/fig_regret.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_shrinkage():
    """lambda vs the ratio of inter-model spread to its standard error"""
    F = json.load(open("experiments/results/final.json"))["per_dataset"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    mk = {"evprimary-formal": "o", "evpublic": "s", "evprimary-informal": "^"}
    for ds, R in F.items():
        xs = [(max(i["F"]) - min(i["F"])) / i["se"] for i in R["info"]]
        ys = [i["lambda"] for i in R["info"]]
        ax.scatter(xs, ys, marker=mk[ds], s=26, label=DS[ds], alpha=0.85)
    ax.set_xlabel("validation-F1 spread / standard error")
    ax.set_ylabel("shrinkage factor $\\lambda$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/fig_shrinkage.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ea = error_analysis()
    print(json.dumps({k: v for k, v in ea.items() if k != "_examples"}, indent=1))
    print("\nexamples missed by every learner:")
    for e in ea["_examples"]:
        print("  -", e[:150])
    json.dump(ea, open("experiments/results/error_analysis.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    # fig_confidence() is no longer used in the manuscript; the separation is reported numerically
    fig_diversity_and_profile(); fig_shrinkage()
    print("\nfigures written to", OUT)
