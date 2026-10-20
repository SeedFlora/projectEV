"""Risk study behind the reframing of SA-WSV as a robustness / risk-control rule.

Part 1, validation-size risk. Every rule is refitted on random validation subsets of 20, 40 and
80 items (20 draws per fold) and on the full split, for every corpus, complete seed and fold, and
scored on the fold's full test portion. A unit is one (corpus, size, seed, draw); its outcome is the
macro-F1 of the pooled out-of-fold predictions. Full-size units use the bootstrap stream of
rev_lib.pooled_run and therefore reproduce the main comparison exactly. "Late-epoch accuracy
weighting" is excluded: it needs per-epoch accuracies on the full validation split.

Three views of risk are reported, because they answer different questions:
  * deployment risk: unit-level difference to uniform soft voting over the 900 reduced-size units
    (equal weight per setting once every corpus has all seeds): mean, 5th percentile, CVaR5 (mean of
    the worst 5% of units), P(loss > 0.5 pp), P(loss > 1.0 pp);
  * validation-size regret: the same rule fitted on the subset minus fitted on the full split of the
    same run, which isolates the cost of a small validation split from run-to-run differences;
  * family contrasts: every SA-WSV variant against its unshrunk counterpart and its lambda -> 0 limit
    (with the entropy term the limit is entropy-confidence voting; without it, uniform soft voting).
Uncertainty comes from a cluster bootstrap that resamples fine-tuning runs: a cluster is (corpus group,
seed), where EV-Norm and EV-Raw form one group because they are the same posts with the same folds;
clusters are resampled within each group.

Part 2, stress test. One learner is corrupted by swapping its two class probabilities on the
instances whose uniform random number falls below rho, drawn once per (seed, corpus, fold, learner,
split), so the corrupted sets are nested across rho and the learner stays confident while its accuracy
falls towards chance. Full-size fits use the pooled_run bootstrap stream at every rho (common random
numbers; rho = 0 reproduces the main comparison); a 40-item validation draw per (seed, corpus, fold,
learner) is shared by all rho. Results are averaged over the corrupted learner and over seeds.
"""
import os, sys, json, math, argparse, time
from collections import OrderedDict
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rev_lib as L

SIZES = [20, 40, 80, None]                       # None = full validation split
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_CLUSTER_BOOT = 2000
RISK_RULES = OrderedDict((k, v) for k, v in L.ALL_RULES.items() if k != "Late-epoch accuracy weighting")
STRESS_NAMES = ["Uniform soft voting", "Majority voting", "Best single (val-selected)", "Linear-F1 weighting",
                "Log-odds weighting", "SA-WSV w/o confidence", "Log-odds x entropy", "Entropy confidence",
                "SA-WSV (binomial)", "SA-WSV (bootstrap)", "SA-WSV (EB)", "Stacking (LR)", "Convex NLL weights",
                "Ensemble selection"]
STRESS_RULES = OrderedDict((k, L.ALL_RULES[k]) for k in STRESS_NAMES)
SHRINK = {"SA-WSV (binomial)": "sawsv_binom", "SA-WSV (bootstrap)": "sawsv_boot",
          "SA-WSV (bootstrap, marginal)": "sawsv_bootm", "SA-WSV (Bock)": "sawsv_bock", "SA-WSV (EB)": "sawsv_eb"}
SHRINK_LABEL = {"sawsv_binom": "lambda", "sawsv_boot": "lambda", "sawsv_bootm": "lambda",
                "sawsv_bock": "Bock factor", "sawsv_eb": "mean B_m"}
# (shrunk rule, unshrunk counterpart, lambda -> 0 limit)
FAMILIES = [("SA-WSV w/o confidence", "Log-odds weighting", "Uniform soft voting"),
            ("SA-WSV (bootstrap)", "Log-odds x entropy", "Entropy confidence"),
            ("SA-WSV (binomial)", "Log-odds x entropy", "Entropy confidence"),
            ("SA-WSV (EB)", "Log-odds x entropy", "Entropy confidence")]
OUT = os.path.join(L.ROOT, "experiments", "results", "rev")


def _subsample(yv, n, rng):
    while True:
        sel = np.sort(rng.choice(len(yv), size=n, replace=False))
        if np.bincount(yv[sel], minlength=2).min() > 0:
            return sel


def risk_task(task):
    """one unit (corpus, seed, size, draw) -> pooled macro-F1 per rule and fold-mean shrinkage factors"""
    ds, seed, size, rep = task
    ys, preds, lams = [], {}, {k: [] for k in SHRINK.values()}
    for fold in range(L.N_FOLDS):
        d = L.load_fold(ds, fold, seed)
        Pv, yv = d["Pv"], d["yv"]
        if size is None:
            rng = L.fold_rng(seed, ds, fold)
        else:
            sel = _subsample(yv, size, L.fold_rng(seed, ds, fold, rep=rep, salt=1000 + size))
            Pv, yv = Pv[:, sel], yv[sel]
            rng = L.fold_rng(seed, ds, fold, rep=rep, salt=2000 + size)
        fitted, ctx = L.fit_rules(Pv, yv, rng, rules=RISK_RULES)
        for name, f in fitted.items():
            preds.setdefault(name, []).append(f(d["Pt"])[0])
        for k in lams:
            if k in ctx.meta:
                lams[k].append(ctx.meta[k]["lam"])
        ys.append(d["yt"])
    y = np.concatenate(ys)
    return {"ds": ds, "seed": seed, "size": "full" if size is None else size, "rep": rep,
            "f1": {k: L.macro_f1(y, np.concatenate(v)) for k, v in preds.items()},
            "lam": {k: float(np.mean(v)) for k, v in lams.items() if v}}


def stress_task(task):
    ds, seed, m, size = task
    out = {rho: {"ys": [], "preds": {}, "lam_b": [], "lam_s": [], "w_m": []} for rho in RHOS}
    for fold in range(L.N_FOLDS):
        d = L.load_fold(ds, fold, seed)
        uv = L.fold_rng(seed, ds, fold, rep=m, salt=2001).random(d["Pv"].shape[1])
        ut = L.fold_rng(seed, ds, fold, rep=m, salt=2002).random(d["Pt"].shape[1])
        sel = None if size is None else _subsample(d["yv"], size, L.fold_rng(seed, ds, fold, rep=m, salt=3000 + size))
        for rho in RHOS:
            Pv, Pt = d["Pv"].copy(), d["Pt"].copy()
            Pv[m, uv < rho] = Pv[m, uv < rho][:, ::-1]
            Pt[m, ut < rho] = Pt[m, ut < rho][:, ::-1]
            yv = d["yv"]
            if sel is not None:
                Pv, yv = Pv[:, sel], yv[sel]
            rng = L.fold_rng(seed, ds, fold) if size is None else L.fold_rng(seed, ds, fold, rep=m, salt=4000 + size)
            fitted, ctx = L.fit_rules(Pv, yv, rng, rules=STRESS_RULES)
            o = out[rho]
            for name, f in fitted.items():
                o["preds"].setdefault(name, []).append(f(Pt)[0])
            o["lam_b"].append(ctx.meta["sawsv_binom"]["lam"]); o["lam_s"].append(ctx.meta["sawsv_boot"]["lam"])
            o["w_m"].append(ctx.meta["sawsv_boot"]["w"][m])
            o["ys"].append(d["yt"])
    res = []
    for rho, o in out.items():
        y = np.concatenate(o["ys"])
        res.append({"ds": ds, "seed": seed, "member": m, "rho": rho, "size": "full" if size is None else size,
                    "f1": {k: L.macro_f1(y, np.concatenate(v)) for k, v in o["preds"].items()},
                    "lam_binom": float(np.mean(o["lam_b"])), "lam_boot": float(np.mean(o["lam_s"])),
                    "w_corrupted": float(np.mean(o["w_m"]))})
    return res


# ------------------------------------------------------------------ statistics
def cvar(x, q=0.05):
    x = np.sort(np.asarray(x, float))
    k = max(1, int(math.ceil(q * len(x))))
    return float(x[:k].mean())


def stats(d):
    d = np.asarray(d, float)
    return {"mean": float(d.mean()), "p5": float(np.percentile(d, 5)), "cvar5": cvar(d),
            "p_loss_05": float((d < -0.005).mean()), "p_loss_10": float((d < -0.010).mean()),
            "p_gain_05": float((d > 0.005).mean()), "worst": float(d.min()), "n": int(len(d))}


def group_of(ds):
    return "primary" if ds.startswith("evprimary") else "public"


def cluster_bootstrap(units, fn, n_boot=N_CLUSTER_BOOT, seed=0):
    """percentile CI of fn(list of units) under resampling of (corpus group, seed) clusters within groups"""
    clusters = {}
    for u in units:
        clusters.setdefault((group_of(u["ds"]), u["seed"]), []).append(u)
    by_group = {}
    for (g, s), us in clusters.items():
        by_group.setdefault(g, []).append(us)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = []
        for g, cl in by_group.items():
            for i in rng.integers(0, len(cl), len(cl)):
                sample.extend(cl[i])
        vals.append(fn(sample))
    vals = np.asarray(vals)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarise_risk(units):
    rules = list(units[0]["f1"])
    full = {(u["ds"], u["seed"]): u for u in units if u["size"] == "full"}
    for u in units:
        u["diff"] = {k: u["f1"][k] - u["f1"][L.BASELINE] for k in rules}
        u["regret"] = {k: u["f1"][k] - full[(u["ds"], u["seed"])]["f1"][k] for k in rules}
    reduced = [u for u in units if u["size"] != "full"]
    sizes = [sz for sz in ["20", "40", "80", "full"] if any(str(u["size"]) == sz for u in units)]
    settings = sorted({(u["ds"], str(u["size"])) for u in units})
    per_setting = {f"{L.DS_LAB[ds]}/n_val={sz}": {k: float(np.mean([u["diff"][k] for u in units
                                                                   if u["ds"] == ds and str(u["size"]) == sz]))
                                                   for k in rules} for ds, sz in settings}
    n_per_setting = {f"{L.DS_LAB[ds]}/n_val={sz}": sum(1 for u in units if u["ds"] == ds and str(u["size"]) == sz)
                     for ds, sz in settings}
    out = {"rules": rules, "n_per_setting": n_per_setting, "per_setting": per_setting, "summary": {}, "risk": {},
           "risk_ci": {}, "risk_by_size": {}, "risk_by_corpus": {}, "regret_by_size": {}, "tail_clusters": {},
           "shrinkage_by_size": {}, "families": []}
    for k in rules:
        v = [per_setting[s][k] for s in per_setting]
        out["summary"][k] = {"mean": float(np.mean(v)), "worst": float(np.min(v)), "best": float(np.max(v))}
        d = [u["diff"][k] for u in reduced]
        out["risk"][k] = stats(d)
        out["risk_ci"][k] = {
            "mean": cluster_bootstrap(reduced, lambda s, k=k: float(np.mean([u["diff"][k] for u in s]))),
            "cvar5": cluster_bootstrap(reduced, lambda s, k=k: cvar([u["diff"][k] for u in s]))}
        out["risk_by_size"][k] = {sz: stats([u["diff"][k] for u in units if str(u["size"]) == sz]) for sz in sizes}
        out["regret_by_size"][k] = {sz: stats([u["regret"][k] for u in units if str(u["size"]) == sz])
                                    for sz in sizes if sz != "full"}
        out["risk_by_corpus"][k] = {L.DS_LAB[ds]: stats([u["diff"][k] for u in reduced if u["ds"] == ds])
                                    for ds in L.IN_DOMAIN if any(u["ds"] == ds for u in reduced)}
        worst = sorted(reduced, key=lambda u: u["diff"][k])[:max(1, int(math.ceil(0.05 * len(reduced))))]
        cl = {}
        for u in worst:
            cl[f"{L.DS_LAB[u['ds']]}/seed{u['seed']}"] = cl.get(f"{L.DS_LAB[u['ds']]}/seed{u['seed']}", 0) + 1
        out["tail_clusters"][k] = dict(sorted(cl.items(), key=lambda kv: -kv[1]))
    for name, key in SHRINK.items():
        out["shrinkage_by_size"][name] = {"quantity": SHRINK_LABEL[key], **{
            sz: float(np.mean([u["lam"][key] for u in units if str(u["size"]) == sz]))
            for sz in sizes if any(str(u["size"]) == sz and key in u["lam"] for u in units)}}
    for shrunk, raw, limit in FAMILIES:
        if shrunk not in rules:
            continue
        fam = {"shrunk": shrunk, "unshrunk": raw, "limit": limit}
        for other, tag in ((raw, "vs_unshrunk"), (limit, "vs_limit")):
            pd_ = [u["f1"][shrunk] - u["f1"][other] for u in reduced]
            fam[tag] = {"paired": stats(pd_),
                        "paired_mean_ci": cluster_bootstrap(reduced, lambda s, a=shrunk, b=other:
                                                            float(np.mean([u["f1"][a] - u["f1"][b] for u in s]))),
                        "cvar5_gap": cvar([u["diff"][shrunk] for u in reduced]) - cvar([u["diff"][other] for u in reduced]),
                        "cvar5_gap_ci": cluster_bootstrap(reduced, lambda s, a=shrunk, b=other:
                                                          cvar([u["diff"][a] for u in s]) - cvar([u["diff"][b] for u in s]))}
        out["families"].append(fam)
    return out


def summarise_stress(units):
    out = {}
    for size in ["full", 40]:
        sel = [u for u in units if u["size"] == size]
        if not sel:
            continue
        clean = {ds: float(np.mean([u["f1"][L.BASELINE] for u in sel if u["ds"] == ds and u["rho"] == 0.0]))
                 for ds in L.IN_DOMAIN if any(u["ds"] == ds for u in sel)}
        S = {}
        for rho in RHOS:
            row = {"f1": {}, "diff_clean_uniform": {}, "per_corpus": {}}
            for name in STRESS_NAMES:
                per_ds = {L.DS_LAB[ds]: float(np.mean([u["f1"][name] for u in sel if u["ds"] == ds and u["rho"] == rho]))
                          for ds in clean}
                row["f1"][name] = per_ds
                row["diff_clean_uniform"][name] = float(np.mean([per_ds[L.DS_LAB[ds]] - clean[ds] for ds in clean]))
            for ds in clean:
                r_units = [u for u in sel if u["rho"] == rho and u["ds"] == ds]
                row["per_corpus"][L.DS_LAB[ds]] = {"lam_binom": float(np.mean([u["lam_binom"] for u in r_units])),
                                                   "lam_boot": float(np.mean([u["lam_boot"] for u in r_units])),
                                                   "w_corrupted": float(np.mean([u["w_corrupted"] for u in r_units]))}
            pc = row["per_corpus"].values()
            row["lam_binom"] = float(np.mean([v["lam_binom"] for v in pc]))
            row["lam_boot"] = float(np.mean([v["lam_boot"] for v in pc]))
            row["w_corrupted_sawsv_boot"] = float(np.mean([v["w_corrupted"] for v in pc]))
            S[str(rho)] = row
        out[str(size)] = {"clean_uniform": {L.DS_LAB[k]: v for k, v in clean.items()}, "rho": S}
    return out


# ---------------------------------------------------------------------- figures
SHORT = {"Best single (val-selected)": "best single", "Majority voting": "majority", "Uniform soft voting": "uniform",
         "Linear-F1 weighting": "linear-F1", "Softmax-F1 weighting": "softmax-F1", "Log-odds weighting": "log-odds",
         "Max-prob confidence": "max-prob", "Entropy confidence": "entropy conf.", "Linear-F1 x max-prob": "lin-F1 x mp",
         "Log-odds x max-prob": "log-odds x mp", "Log-odds x entropy": "log-odds x entropy",
         "SA-WSV (binomial)": "SA-WSV$_1$ (binom.)", "Product rule": "product", "Uniform + TS": "uniform+TS",
         "Stacking (LR)": "stacking", "Convex NLL weights": "convex NLL", "Ensemble selection": "ens. selection",
         "SA-WSV (bootstrap)": "SA-WSV$_1$", "SA-WSV (bootstrap, marginal)": "SA (marg.)", "SA-WSV (Bock)": "SA (Bock)",
         "SA-WSV (EB)": "SA (EB)", "SA-WSV (bagged)": "SA (bagged)", "SA-WSV + TS": "SA-WSV+TS",
         "SA-WSV, rank confidence": "SA (rank)", "SA-WSV w/o confidence": "SA-WSV$_0$"}
FIG_RULES = ["Uniform soft voting", "Linear-F1 weighting", "Log-odds weighting", "SA-WSV w/o confidence",
             "Entropy confidence", "Log-odds x entropy", "SA-WSV (bootstrap)", "SA-WSV (binomial)",
             "Softmax-F1 weighting", "Stacking (LR)", "Convex NLL weights", "Ensemble selection",
             "Majority voting", "Best single (val-selected)"]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
                         "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    return plt


def fig_risk(R, path):
    """risk-return plane for selected rules, with arrows from each unshrunk rule to its shrunk version"""
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    pos = {k: (100 * R["risk"][k]["cvar5"], 100 * R["risk"][k]["mean"]) for k in FIG_RULES if k in R["risk"]}
    for shrunk, raw, limit in FAMILIES[:2]:
        for a, b, col in ((raw, shrunk, "#b3352c"), (shrunk, limit, "#999999")):
            if a in pos and b in pos:
                ax.annotate("", xy=pos[b], xytext=pos[a],
                            arrowprops=dict(arrowstyle="->", lw=0.7, color=col, shrinkA=3, shrinkB=3))
    offs = {"Uniform soft voting": (3, -7), "Linear-F1 weighting": (3, 3), "Log-odds weighting": (3, 3),
            "SA-WSV w/o confidence": (3, -7), "Entropy confidence": (-30, -8), "Log-odds x entropy": (3, 3),
            "SA-WSV (bootstrap)": (3, -7), "SA-WSV (binomial)": (-48, 3), "Softmax-F1 weighting": (3, 2),
            "Stacking (LR)": (3, 2), "Convex NLL weights": (3, -7), "Ensemble selection": (3, 2),
            "Majority voting": (3, 2), "Best single (val-selected)": (3, 2)}
    for k, (x, y) in pos.items():
        sa = k.startswith("SA-WSV")
        ax.scatter([x], [y], s=20 if sa else 11, marker="D" if sa else "o", zorder=3,
                   color="#b3352c" if sa else ("#000000" if k == L.BASELINE else "#1f4e79"))
        ax.annotate(SHORT[k], (x, y), fontsize=5.6, xytext=offs.get(k, (3, 2)), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5, ls=":"); ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("CVaR$_5$ of difference to uniform (pp)")
    ax.set_ylabel("mean difference to uniform (pp)")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def fig_stress(S, path):
    plt = _style()
    lines = [("Uniform soft voting", "#000000", "-", "o"), ("Best single (val-selected)", "#7f7f7f", "--", "v"),
             ("Entropy confidence", "#cc79a7", ":", "<"), ("SA-WSV w/o confidence", "#e69f00", "-", "^"),
             ("SA-WSV (bootstrap)", "#b3352c", "-", "D"), ("Stacking (LR)", "#009e73", "--", "x"),
             ("Convex NLL weights", "#56b4e9", ":", "+")]
    sizes = [s for s in ["full", "40"] if s in S]
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.4, 2.35), sharey=True, squeeze=False)
    for ax, sz in zip(axes[0], sizes):
        for name, col, ls, mk in lines:
            ys = [100 * S[sz]["rho"][str(r)]["diff_clean_uniform"][name] for r in RHOS]
            ax.plot(RHOS, ys, color=col, ls=ls, marker=mk, ms=3, lw=1, label=SHORT[name])
        ax.set_xlabel("corrupted fraction $\\rho$")
        ax.set_title("full validation split" if sz == "full" else "40 validation items", fontsize=7, pad=2)
    axes[0][0].set_ylabel("$\\Delta$ macro-F1 vs clean uniform (pp)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.84))
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def _round(o):
    if isinstance(o, float):
        return float(f"{o:.10g}")
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 draws, 2 seeds: development only")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    reps = 3 if a.quick else 20
    seeds = {ds: L.complete_seeds(ds) for ds in L.IN_DOMAIN}
    common = sorted(set.intersection(*[set(v) for v in seeds.values()]), key=L.SEEDS.index)
    seeds = {ds: (common[:2] if a.quick else common) for ds in L.IN_DOMAIN}
    if not a.quick and len(common) < len(L.SEEDS):
        print(f"PRELIMINARY: only seeds {common} are complete for every corpus")
    t0 = time.time()
    risk_tasks = [(ds, s, sz, r) for ds in L.IN_DOMAIN for s in seeds[ds] for sz in SIZES
                  for r in (range(reps) if sz is not None else [0])]
    stress_tasks = [(ds, s, m, sz) for ds in L.IN_DOMAIN for s in seeds[ds] for m in range(len(L.MO)) for sz in (None, 40)]
    with Pool(min(a.workers, 6)) as pool:
        risk_units = pool.map(risk_task, risk_tasks, chunksize=4)
        stress_units = [u for batch in pool.map(stress_task, stress_tasks, chunksize=2) for u in batch]
    R = summarise_risk(risk_units)
    S = summarise_stress(stress_units)
    expected = len(L.IN_DOMAIN) * len(common) * (3 * reps)
    n_reduced = sum(1 for u in risk_units if u["size"] != "full")
    assert n_reduced == (len(L.IN_DOMAIN) * len(seeds[L.IN_DOMAIN[0]]) * 3 * reps), n_reduced
    unit_table = [{"ds": u["ds"], "seed": u["seed"], "size": u["size"], "rep": u["rep"], "f1": u["f1"], "lam": u["lam"]}
                  for u in risk_units]
    os.makedirs(OUT, exist_ok=True)
    res = {"config": {"sizes": [s if s else "full" for s in SIZES], "reps": reps, "rhos": RHOS,
                      "seeds_used": {L.DS_LAB[k]: v for k, v in seeds.items()}, "quick": a.quick,
                      "n_reduced_units": n_reduced, "n_reduced_units_expected_all_seeds": expected,
                      "n_cluster_boot": N_CLUSTER_BOOT, "clusters": "(corpus group, seed); EV-Norm and EV-Raw form one group",
                      "rng": {"subsample": "fold_rng(seed, ds, fold, rep, salt=1000+n)", "fit": "fold_rng(seed, ds, fold, rep, salt=2000+n); full: fold_rng(seed, ds, fold)",
                              "stress_masks": "fold_rng(seed, ds, fold, rep=m, salt=2001/2002)", "stress_subsample": "salt=3000+n",
                              "stress_fit": "full: fold_rng(seed, ds, fold); n=40: salt=4000+n"},
                      "risk_rules": list(RISK_RULES), "stress_rules": STRESS_NAMES, "families": FAMILIES,
                      "excluded": {"Late-epoch accuracy weighting": "needs per-epoch accuracies on the full validation split"}},
           "risk": R, "stress": S, "units": unit_table}
    name = "rev_robustness_quick.json" if a.quick else "rev_robustness.json"
    json.dump(_round(res), open(os.path.join(OUT, name), "w"), indent=1)
    if not a.quick:
        fig_risk(R, os.path.join(L.ROOT, "paper", "fig_risk.pdf"))
        fig_stress(S, os.path.join(L.ROOT, "paper", "fig_stress.pdf"))
    # ---------------------------------------------------------------- printout
    print(f"seeds: {res['config']['seeds_used']}  units: risk={len(risk_units)} (reduced {n_reduced}) "
          f"stress={len(stress_units)}  time={time.time() - t0:.0f}s")
    print("\nDEPLOYMENT RISK (reduced-size units; rule - uniform, pp)  mean [cluster CI]    CVaR5 [cluster CI]   P(<-0.5) P(<-1) | settings mean/worst/best | tail clusters")
    for k in sorted(R["risk"], key=lambda k: -R["risk"][k]["cvar5"]):
        v, s, ci = R["risk"][k], R["summary"][k], R["risk_ci"][k]
        tc = ", ".join(f"{c}:{n}" for c, n in list(R["tail_clusters"][k].items())[:3])
        print(f"  {k:30s} {100*v['mean']:+5.2f} [{100*ci['mean'][0]:+5.2f},{100*ci['mean'][1]:+5.2f}]  "
              f"{100*v['cvar5']:+5.2f} [{100*ci['cvar5'][0]:+5.2f},{100*ci['cvar5'][1]:+5.2f}]  "
              f"{v['p_loss_05']:.3f}  {v['p_loss_10']:.3f} | {100*s['mean']:+5.2f} {100*s['worst']:+5.2f} {100*s['best']:+5.2f} | {tc}")
    print("\nVALIDATION-SIZE REGRET (rule on subset - same rule on full split, pp): mean / CVaR5 at n=20, 40, 80")
    for k in ["Uniform soft voting", "Linear-F1 weighting", "Log-odds weighting", "SA-WSV w/o confidence",
              "Log-odds x entropy", "SA-WSV (bootstrap)", "SA-WSV (binomial)", "Softmax-F1 weighting",
              "Stacking (LR)", "Convex NLL weights", "Ensemble selection", "Best single (val-selected)"]:
        r = R["regret_by_size"][k]
        print(f"  {k:30s} " + "  ".join(f"n={sz}: {100*r[sz]['mean']:+5.2f}/{100*r[sz]['cvar5']:+5.2f}" for sz in ["20", "40", "80"] if sz in r))
    print("\nFAMILIES (reduced units): paired mean [cluster CI]; CVaR5 gap vs uniform [cluster CI]")
    for f in R["families"]:
        for tag in ("vs_unshrunk", "vs_limit"):
            v = f[tag]
            other = f["unshrunk"] if tag == "vs_unshrunk" else f["limit"]
            print(f"  {f['shrunk']:24s} - {other:22s}: {100*v['paired']['mean']:+5.2f} [{100*v['paired_mean_ci'][0]:+5.2f},"
                  f"{100*v['paired_mean_ci'][1]:+5.2f}]   CVaR5 gap {100*v['cvar5_gap']:+5.2f} [{100*v['cvar5_gap_ci'][0]:+5.2f},"
                  f"{100*v['cvar5_gap_ci'][1]:+5.2f}]")
    print("\nshrinkage factor by validation size:", json.dumps({k: {s: (round(x, 3) if isinstance(x, float) else x)
                                                                   for s, x in v.items()} for k, v in R["shrinkage_by_size"].items()}))
    for sz, block in S.items():
        rows = block["rho"]
        print(f"\nSTRESS (n_val={sz}): difference to clean uniform (pp), mean over corpora")
        print("  rho  " + " ".join(f"{SHORT[n][:9]:>9s}" for n in STRESS_NAMES) + "  lam_b lam_s w_corr")
        for rho in RHOS:
            r = rows[str(rho)]
            print(f"  {rho:.1f} " + " ".join(f"{100*r['diff_clean_uniform'][n]:+9.2f}" for n in STRESS_NAMES)
                  + f"  {r['lam_binom']:.2f}  {r['lam_boot']:.2f}  {r['w_corrupted_sawsv_boot']:.3f}")


if __name__ == "__main__":
    main()
