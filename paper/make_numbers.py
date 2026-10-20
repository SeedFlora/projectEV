"""Extract every number quoted in the revised manuscript from experiments/results/rev/*.json.

Prints the LaTeX rows of Tables I-IV and a dictionary of the prose numbers, and writes them to
numbers.json, which verify.py checks against main.tex. Nothing here is computed from raw data; it only
reads the analysis outputs, so the manuscript cannot drift from the experiments.
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "results", "rev")
J = lambda n: json.load(open(os.path.join(R, n), encoding="utf-8"))
M, C, V, RB, T, A, K, AB = (J("rev_main.json"), J("rev_calibration.json"), J("rev_variance.json"),
                            J("rev_robustness.json"), J("rev_transfer.json"), J("rev_annotation.json"),
                            J("rev_cost.json"), J("rev_ablation.json"))
DS = ["evprimary-formal", "evpublic", "evprimary-informal"]
LAB = {"evprimary-formal": "EV-Norm", "evpublic": "EV-Public", "evprimary-informal": "EV-Raw"}
LEARNERS = ["IndoBERT-p1", "IndoBERT-1.5G", "IndoRoBERTa-1.5G", "XLM-R", "mBERT"]
U = "Uniform soft voting"
pp = lambda x: 100 * x
out = {"tables": {}, "text": {}}

# ------------------------------------------------------------------ Table I
rows = []
for m in LEARNERS:
    cells = []
    for ds in DS:
        f = M["per_corpus"][ds]["metrics"][m]["f1"]
        cells.append(f"{pp(f['mean']):.2f}$\\pm${pp(f['sd_seed']):.2f}")
    Tm = sum(C["per_learner"][ds][m]["T"]["mean"] for ds in DS) / 3
    e_raw = sum(C["per_learner_pooled"][ds][m]["ece_raw"]["mean"] for ds in DS) / 3
    e_ts = sum(C["per_learner_pooled"][ds][m]["ece_ts"]["mean"] for ds in DS) / 3
    rows.append(f"{m:16s} & " + " & ".join(cells) + f" & {Tm:.2f} & {e_raw:.3f}/{e_ts:.3f} \\\\".replace("0.0", ".0"))
out["tables"]["I"] = rows

# ------------------------------------------------------------------ Table II
T2 = ["Best single (val-selected)", "Majority voting", U, "Linear-F1 weighting", "Softmax-F1 weighting",
      "Log-odds weighting", "Entropy confidence", "Log-odds x entropy", "Product rule", "Late-epoch accuracy weighting",
      "Uniform + TS", "Stacking (LR)", "Convex NLL weights", "Ensemble selection", "SA-WSV (binomial)",
      "SA-WSV (bootstrap)", "SA-WSV w/o confidence"]
rows = []
for r in T2:
    cells = []
    for ds in DS:
        mean = pp(M["per_corpus"][ds]["metrics"][r]["f1"]["mean"])
        sig = ""
        if r != U:
            s = M["per_corpus"][ds]["significance"]["vs"][U][r]
            if s["ci95"][0] > 0 or s["ci95"][1] < 0:
                sig = "$^{\\dagger}$"
        cells.append(f"{mean:.2f}{sig}")
    rows.append(f"{r:30s} & " + " & ".join(cells) + " \\\\")
# the binomial and bootstrap SA-WSV_1 rows print identically; the manuscript shows them as one row
i_b, i_s = T2.index("SA-WSV (binomial)"), T2.index("SA-WSV (bootstrap)")
if rows[i_b].split("&", 1)[1] == rows[i_s].split("&", 1)[1]:
    rows[i_b] = "SA-WSV_1, binomial = bootstrap & " + rows[i_b].split("&", 1)[1].strip()
    del rows[i_s]
out["tables"]["II"] = rows

# ------------------------------------------------------------------ Table III
T3 = ["Best single (val-selected)", "Linear-F1 weighting", "Softmax-F1 weighting", "Log-odds weighting",
      "Entropy confidence", "Log-odds x entropy", "Stacking (LR)", "Convex NLL weights", "Ensemble selection",
      "SA-WSV (binomial)", "SA-WSV (bootstrap)", "SA-WSV w/o confidence"]
rk = RB["risk"]
sgn = lambda x: f"{x:+.2f}".replace("-", "$-$")
rows = []
for r in T3:
    full = rk["risk_by_size"][r]["full"]["mean"]
    v = rk["risk"][r]
    rows.append(f"{r:30s} & {sgn(pp(full))} & {sgn(pp(v['mean']))} & {sgn(pp(v['cvar5']))} & {100*v['p_loss_05']:.1f} \\\\")
out["tables"]["III"] = rows

# ------------------------------------------------------------------ Table IV (uniform transfer)
dirs = T["directions"]
indom = {lab: T["a_indomain_reference"][lab]["f1"][U]["mean"] for lab in T["a_indomain_reference"]}
grid = {}
for k, d in dirs.items():
    src, tgt = [s.strip() for s in k.split("->")]
    grid[(src, tgt)] = d["a"]["cells"][U]["transfer"]["mean"]
rows = []
for src in ["EV-Norm", "EV-Raw", "EV-Public"]:
    cells = []
    for tgt in ["EV-Norm", "EV-Raw", "EV-Public", "EV-Twitter", "EV-YouTube"]:
        if src == tgt:
            cells.append(f"({indom[src]:.1f})")
        elif (src, tgt) in grid:
            cells.append(f"{grid[(src, tgt)]:.1f}")
        else:
            cells.append("--")
    rows.append(f"{src:9s} & " + " & ".join(cells) + " \\\\")
out["tables"]["IV"] = rows

# ------------------------------------------------------------------ prose numbers
t = out["text"]
seeds = M["meta"]["seeds"]
t["seeds"] = seeds
mm = {lab: M["per_corpus"][ds]["mono_vs_multi"] for ds, lab in LAB.items()}
gaps_mbert = [v["d_f1_mean"] for lab in mm for k, v in mm[lab].items() if k.endswith("vs mBERT")]
t["mono_minus_mbert_range"] = [pp(min(gaps_mbert)), pp(max(gaps_mbert))]
t["mono_minus_mbert_nsig"] = {lab: {k: v["n_seeds_sig"] for k, v in mm[lab].items() if k.endswith("mBERT")} for lab in mm}
gaps_x = {lab: {k: pp(v["d_f1_mean"]) for k, v in mm[lab].items() if k.endswith("XLM-R")} for lab in mm}
t["mono_minus_xlmr"] = gaps_x
t["mono_minus_xlmr_nsig"] = {lab: {k: v["n_seeds_sig"] for k, v in mm[lab].items() if k.endswith("XLM-R")} for lab in mm}
t["mono_minus_xlmr_boot"] = {lab: {k: (v["boot"]["ci95"], v["boot"]["p_holm"]) for k, v in mm[lab].items() if k.endswith("XLM-R")}
                             for lab in mm}
norm_gain = {m: pp(M["per_corpus"]["evprimary-formal"]["metrics"][m]["f1"]["mean"] -
                   M["per_corpus"]["evprimary-informal"]["metrics"][m]["f1"]["mean"]) for m in LEARNERS}
t["normalisation_gain"] = norm_gain
t["learner_sd_seed_range"] = [pp(min(M["per_corpus"][ds]["metrics"][m]["f1"]["sd_seed"] for ds in DS for m in LEARNERS)),
                              pp(max(M["per_corpus"][ds]["metrics"][m]["f1"]["sd_seed"] for ds in DS for m in LEARNERS))]
t["uniform_f1"] = {LAB[ds]: pp(M["per_corpus"][ds]["metrics"][U]["f1"]["mean"]) for ds in DS}
t["primary"] = {LAB[ds]: {k: M["primary_analysis"]["per_corpus"][ds][k] for k in ("diff", "ci95", "p_boot", "p_holm_corpora",
                                                                               "n_seeds_positive") if k in M["primary_analysis"]["per_corpus"][ds]}
                for ds in DS}
t["secondary"] = {LAB[ds]: {k: M["secondary_analysis"]["per_corpus"][ds][k] for k in ("diff", "ci95", "p_boot", "p_holm_corpora",
                                                                                   "n_seeds_positive") if k in M["secondary_analysis"]["per_corpus"][ds]}
                  for ds in DS}
t["holm30_primary"] = {LAB[ds]: M["per_corpus"][ds]["significance"]["vs"][U]["SA-WSV (bootstrap)"]["p_holm"] for ds in DS}
t["sig_vs_uniform_holm"] = {LAB[ds]: M["per_corpus"][ds]["band"].get("significant_vs_uniform_holm") for ds in DS}
t["sig_vs_uniform_holm_rules"] = {LAB[ds]: M["per_corpus"][ds]["band"].get("significant_vs_uniform_holm_rules") for ds in DS}
orig_soft = [r for r in ["Linear-F1 weighting", "Softmax-F1 weighting", "Log-odds weighting", "Max-prob confidence",
                         "Entropy confidence", "Linear-F1 x max-prob", "Log-odds x max-prob", "Log-odds x entropy", "SA-WSV (binomial)"]]
t["equiv_orig_soft"] = {LAB[ds]: {d: sum(M["per_corpus"][ds]["significance"]["vs"][U][r][f"equiv_{d}"] for r in orig_soft)
                                  for d in ("0.005", "0.010")} for ds in DS}
t["n_orig_soft"] = len(orig_soft)
t["band_worst_seed"] = {LAB[ds]: {k: M["per_corpus"][ds]["band"][k].get("worst_seed_range_incl_uniform")
                                  for k in ("original_soft_rules", "genuine_soft_votes", "all_soft_rules")} for ds in DS}
t["learned_vs_uniform"] = {LAB[ds]: {r: (pp(M["per_corpus"][ds]["metrics"][r]["f1"]["diff_vs_uniform"]),
                                         M["per_corpus"][ds]["significance"]["vs"][U][r]["ci95"])
                                     for r in ["Stacking (LR)", "Convex NLL weights", "Ensemble selection"]} for ds in DS}
t["boot_minus_binom"] = {LAB[ds]: pp(M["per_corpus"][ds]["metrics"]["SA-WSV (bootstrap)"]["f1"]["mean"]
                                    - M["per_corpus"][ds]["metrics"]["SA-WSV (binomial)"]["f1"]["mean"]) for ds in DS}
div = {LAB[ds]: M["per_corpus"][ds]["diversity"]["mean_over_seeds"] for ds in DS}
t["Q"] = {k: v["Q_foldmean"] for k, v in div.items()}
t["rho_err"] = {k: v["rho_mean"] for k, v in div.items()}
t["m_eff"] = {k: v["m_eff"] for k, v in div.items()}
t["contested"] = {k: v["contested"] for k, v in div.items()}
t["seed_Q_same_learner"] = {LAB[ds]: M["per_corpus"][ds]["diversity"].get("seed_diversity") for ds in DS}
# variance module
sv = V["summary"]
t["var_ratio_boot_binom"] = {k: sv[k]["a_marginal_variance"] for k in sv}
t["centred_noise"] = {k: sv[k]["b_centred_noise"] for k in sv}
t["lambda"] = {k: sv[k]["c_lambda"] for k in sv}
t["heldout"] = {k: sv[k]["d_heldout_noise"] for k in sv}
t["risk_est"] = {k: sv[k]["e_estimation_risk"] for k in sv}
t["f1_vs_acc"] = {k: sv[k]["g_f1_vs_accuracy"] for k in sv}
# calibration
t["calib_corpus"] = {LAB[ds]: C["per_corpus_pooled"][ds] for ds in DS}
t["calib_temperature"] = C.get("temperature")
t["fusion_ece"] = {LAB[ds]: {r: C["fusion"][ds]["rules"][r]["ece"] for r in [U, "SA-WSV (bootstrap)", "SA-WSV + TS",
                                                                             "SA-WSV w/o confidence", "Uniform + TS"]} for ds in DS}
t["fusion_primary_tests"] = {LAB[ds]: C["fusion"][ds].get("primary") for ds in DS}
t["contested_shares"] = {LAB[ds]: C["contested"][ds].get("summary", C["contested"][ds]) for ds in DS}
# robustness
t["risk"] = {r: rk["risk"][r] for r in rk["risk"]}
t["risk_ci"] = rk["risk_ci"]
t["families"] = rk["families"]
t["regret"] = {r: rk["regret_by_size"][r] for r in rk["regret_by_size"]}
t["tail_clusters"] = rk["tail_clusters"]
t["stress"] = {sz: {rho: {"diff": v["diff_clean_uniform"], "lam_boot": v["lam_boot"], "w": v["w_corrupted_sawsv_boot"]}
                    for rho, v in RB["stress"][sz]["rho"].items()} for sz in RB["stress"]}
# transfer
t["transfer_summary"] = T["summary"]
t["transfer_rule_diffs"] = T.get("d_across_directions")
# annotation
t["annotation"] = A
# ablation
t["ablation_spread"] = AB["spread"]
# cost
t["cost"] = K.get("citable")
t["cost_timing"] = K.get("citable_cpu_timing")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False, default=str)
for name, rows in out["tables"].items():
    print(f"% ---- Table {name}")
    print("\n".join(rows))
print("\nwrote numbers.json")
