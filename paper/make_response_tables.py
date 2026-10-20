"""Appendix tables of the response letter, generated from experiments/results/rev/*.json.

Writes response_tables.tex (included by response_to_reviewers.tex). Every number comes from the
analysis outputs, like the tables of the manuscript.
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "experiments", "results", "rev")
J = lambda n: json.load(open(os.path.join(R, n), encoding="utf-8"))
M, V, C, RB, T, A, K, AB = (J("rev_main.json"), J("rev_variance.json"), J("rev_calibration.json"), J("rev_robustness.json"),
                            J("rev_transfer.json"), J("rev_annotation.json"), J("rev_cost.json"), J("rev_ablation.json"))
DS = ["evprimary-formal", "evpublic", "evprimary-informal"]
LAB = {"evprimary-formal": "EV-Norm", "evpublic": "EV-Public", "evprimary-informal": "EV-Raw"}
LRN = ["IndoBERT-p1", "IndoBERT-1.5G", "IndoRoBERTa-1.5G", "XLM-R", "mBERT"]
U = "Uniform soft voting"
RULES = ["Best single (val-selected)", "Majority voting", U, "Linear-F1 weighting", "Softmax-F1 weighting",
         "Log-odds weighting", "Max-prob confidence", "Entropy confidence", "Linear-F1 x max-prob", "Log-odds x max-prob",
         "Log-odds x entropy", "SA-WSV (binomial)", "Product rule", "Late-epoch accuracy weighting", "Uniform + TS",
         "Stacking (LR)", "Convex NLL weights", "Ensemble selection", "SA-WSV (bootstrap)", "SA-WSV (bootstrap, marginal)",
         "SA-WSV (Bock)", "SA-WSV (EB)", "SA-WSV (bagged)", "SA-WSV + TS", "SA-WSV, rank confidence", "SA-WSV w/o confidence"]
NICE = {"SA-WSV (binomial)": "SA-WSV$_1$, binomial (submitted)", "SA-WSV (bootstrap)": "SA-WSV$_1$, paired bootstrap",
        "SA-WSV (bootstrap, marginal)": "SA-WSV$_1$, marginal bootstrap", "SA-WSV (Bock)": "SA-WSV$_1$, Bock",
        "SA-WSV (EB)": "SA-WSV$_1$, empirical Bayes", "SA-WSV (bagged)": "SA-WSV$_1$, bagged weights",
        "SA-WSV + TS": "SA-WSV$_1$ + temperature scaling", "SA-WSV, rank confidence": "SA-WSV$_1$, percentile confidence",
        "SA-WSV w/o confidence": "SA-WSV$_0$ (no confidence term)", "Linear-F1 x max-prob": "Linear-F1 $\\times$ max-prob",
        "Log-odds x max-prob": "Log-odds $\\times$ max-prob", "Log-odds x entropy": "Log-odds $\\times$ entropy",
        "Uniform + TS": "Uniform + temperature scaling", "Convex NLL weights": "Convex log-loss weights"}
nice = lambda r: NICE.get(r, r)
pp = lambda x: 100 * x
s2 = lambda x: (f"{x:+.2f}").replace("-", "$-$")
ci = lambda c: f"[{s2(pp(c[0]))}, {s2(pp(c[1]))}]"
L = []


def table(caption, cols, header, rows, size="\\footnotesize", label=""):
    L.append("\\begin{table}[!htbp]\\centering" + size)
    L.append(f"\\caption*{{{caption}}}" + (f"\\label{{{label}}}" if label else ""))
    L.append("\\begin{tabular}{" + cols + "}\\toprule")
    L.append(header + " \\\\ \\midrule")
    L.extend(r + " \\\\" for r in rows)
    L.append("\\bottomrule\\end{tabular}\\end{table}\n")


# ---------------------------------------------------------------- A1 all rules
rows = []
for r in RULES:
    cells = []
    for ds in DS:
        m = M["per_corpus"][ds]["metrics"][r]["f1"]
        if r == U:
            cells.append(f"{pp(m['mean']):.2f} & (reference)")
            continue
        s = M["per_corpus"][ds]["significance"]["vs"][U][r]
        ph = s["p_holm"]
        cells.append(f"{pp(m['mean']):.2f} & {s2(pp(s['diff']))} {ci(s['ci95'])}{'$^{*}$' if ph < 0.05 else ''}")
    rows.append(nice(r) + " & " + " & ".join(cells))
table("A1. Macro-F1 of all 26 rules (mean over five seeds) and pooled difference to uniform soft voting with its 95\\% two-stage "
      "bootstrap interval. $^{*}$: significant after Holm correction over the 30 rules and learners of a corpus.",
      "lcccccc", "Rule & \\multicolumn{2}{c}{EV-Norm} & \\multicolumn{2}{c}{EV-Public} & \\multicolumn{2}{c}{EV-Raw}", rows,
      size="\\scriptsize\\setlength{\\tabcolsep}{2.5pt}")
rows = []
for name, key in (("SA-WSV$_1$ vs uniform (primary)", "primary_analysis"), ("SA-WSV$_0$ vs uniform (secondary)", "secondary_analysis")):
    cells = []
    for ds in DS:
        a = M[key]["per_corpus"][ds]
        cells.append(f"{s2(pp(a['diff']))} {ci(a['ci95'])}, $p$={a['p_boot']:.3f}, $p_\\mathrm{{Holm}}$={a['p_holm_corpora']:.3f}, {a['n_seeds_positive']}/5")
    rows.append(name + " & " + " & ".join(cells))
table("A1a. Pre-designated comparisons (pooled macro-F1 points, two-stage bootstrap, Holm over the three corpora; last entry: seeds "
      "with a positive difference).", ">{\\raggedright}p{2.3cm}>{\\raggedright}p{4.25cm}>{\\raggedright}p{4.25cm}>{\\raggedright\\arraybackslash}p{4.25cm}",
      "Comparison & EV-Norm & EV-Public & EV-Raw", rows, size="\\scriptsize")

# ---------------------------------------------------------------- A1b pools
rows = []
for pool in ("seed-averaged", "all-runs"):
    for r in ["Uniform soft voting", "Linear-F1 weighting", "Log-odds x entropy", "SA-WSV (binomial)", "SA-WSV (bootstrap)",
              "Stacking (LR)", "Convex NLL weights", "Ensemble selection", "Best single (val-selected)"]:
        cells = []
        for ds in DS:
            P = M["per_corpus"][ds]["pools"][pool]["rules"].get(r)
            cells.append("--" if P is None else (f"{pp(P['f1_mean']):.2f}" if r == U else f"{s2(pp(P['diff_vs_pool_uniform']))}"))
        rows.append(f"{pool} & {nice(r)} & " + " & ".join(cells))
q = " / ".join(f"{M['per_corpus'][ds]['pools']['all-runs']['Q']:.3f}" for ds in DS)
table(f"A1b. Multi-seed pools (descriptive): uniform macro-F1 of the pool and other rules' difference to it. Seed-averaged: each "
      f"learner's posteriors averaged over the five seeds (5 members); all-runs: 25 members. Q of the all-runs pool: {q}.",
      "llccc", "Pool & Rule & EV-Norm & EV-Public & EV-Raw", rows, size="\\scriptsize")

# ---------------------------------------------------------------- A2 risk
rk = RB["risk"]
rows = []
for r in [x for x in RULES if x != "Late-epoch accuracy weighting"]:
    v, c_, g = rk["risk"][r], rk["risk_ci"][r]["cvar5"], rk["regret_by_size"][r]
    rows.append(f"{nice(r)} & {s2(pp(rk['risk_by_size'][r]['full']['mean']))} & {s2(pp(v['mean']))} & {s2(pp(v['cvar5']))} {ci(c_)} & "
                f"{100*v['p_loss_05']:.1f} & {100*v['p_loss_10']:.1f} & " +
                " & ".join(s2(pp(g[sz]['cvar5'])) for sz in ("20", "40", "80")))
table("A2. Risk over 900 small-validation deployments (20, 40, 80 items; 20 draws per fold, five seeds, three corpora). Differences to "
      "uniform soft voting in points: mean with the full split, mean and CVaR$_5$ (cluster-bootstrap 95\\% interval) over the "
      "deployments, \\% losing over 0.5 and 1.0 points, and CVaR$_5$ of the validation-size regret (rule on the subset minus the same "
      "rule on the full split) at 20, 40 and 80 items.", "lcccccccc",
      "Rule & Full & Mean & CVaR$_5$ & $>$0.5 & $>$1.0 & R$_{20}$ & R$_{40}$ & R$_{80}$", rows, size="\\scriptsize\\setlength{\\tabcolsep}{2.5pt}")
rows = []
for f in rk["families"]:
    for tag, other in (("vs_unshrunk", f["unshrunk"]), ("vs_limit", f["limit"])):
        v = f[tag]
        rows.append(f"{nice(f['shrunk'])} & {nice(other)} & {s2(pp(v['paired']['mean']))} {ci(v['paired_mean_ci'])} & "
                    f"{s2(pp(v['cvar5_gap']))} {ci(v['cvar5_gap_ci'])}")
table("A2a. Shrinkage families over the deployments: paired mean difference and CVaR$_5$ gain (positive = lighter tail) of each "
      "shrunk rule against its unshrunk counterpart and its $\\lambda=0$ limit, with cluster-bootstrap intervals.", "llcc",
      "Shrunk rule & Compared with & Mean difference & CVaR$_5$ gain", rows, size="\\scriptsize")

# ---------------------------------------------------------------- A3 shrinkage estimators
rows = []
EST = [("js_binomial", "binomial (submitted)"), ("js_boot_marginal", "marginal bootstrap"), ("js_boot_paired", "paired bootstrap (revised)"),
       ("bock", "Bock (factor)"), ("eb", "empirical Bayes (mean $B_m$)")]
for key, name in EST:
    lam = [V["summary"][ds]["c_lambda"][key]["mean"] for ds in DS]
    risk = [V["summary"][ds]["e_estimation_risk"][key]["dev_over_raw"]["value"] for ds in DS]
    rows.append(f"{name} & " + " & ".join(f"{x:.2f}" for x in lam) + " & " + " & ".join(f"{x:.2f}" for x in risk))
gm = [V["summary"][ds]["e_estimation_risk"]["grand_mean"]["dev_over_raw"]["value"] for ds in DS]
rows.append("equal reliabilities ($\\lambda=0$) & -- & -- & -- & " + " & ".join(f"{x:.2f}" for x in gm))
table("A3. Shrinkage estimators: mean shrinkage factor per corpus (EV-Norm, EV-Public, EV-Raw) and held-out error of the centred "
      "reliability vector relative to the raw validation estimates (below 1 = closer to the test-fold reliabilities).", "lcccccc",
      "Estimator & $\\bar\\lambda$ N & $\\bar\\lambda$ P & $\\bar\\lambda$ R & risk N & risk P & risk R", rows, size="\\scriptsize")
va = V["summary"]
rows = []
for ds in DS + ["all"]:
    a, b, h = va[ds]["a_marginal_variance"], va[ds]["b_centred_noise"]["val"], va[ds]["d_heldout_noise"]["obs_over_pred"]
    nm = LAB.get(ds, "pooled")
    rows.append(f"{nm} & {a['bootstrap_over_binomial']['mean']:.3f} & {a['delta_over_binomial']['mean']:.3f} & "
                f"{b['discount_ratio_of_means']:.3f} & {b['rho_F_replicates']['mean']:.2f} & {h['binomial']['value']:.2f} "
                f"[{h['binomial']['ci95'][0]:.2f}, {h['binomial']['ci95'][1]:.2f}] & {h['paired']['value']:.2f} "
                f"[{h['paired']['ci95'][0]:.2f}, {h['paired']['ci95'][1]:.2f}]")
table("A3a. Sampling variance of validation macro-F1 (R2.1): bootstrap/binomial and delta/binomial variance ratios per learner; "
      "paired noise of the centred vector relative to the independence value and the correlation of learners' bootstrap replicates; "
      "held-out check (observed/predicted validation--test discrepancy, 95\\% interval) under the binomial-independent and the "
      "paired-bootstrap noise models.", "lcccccc",
      "Corpus & boot/binom & delta/binom & paired/indep. & corr. & held-out binom. & held-out paired", rows, size="\\scriptsize")

# ---------------------------------------------------------------- A4 calibration and ablation
rows = []
for m in LRN:
    cells = []
    for ds in DS:
        pl, pp_ = C["per_learner"][ds][m], C["per_learner_pooled"][ds][m]
        cells.append(f"{pl['T']['mean']:.2f} & {pp_['ece_raw']['mean']:.3f}/{pp_['ece_ts']['mean']:.3f} & {pl['auroc_raw']['mean']:.3f}")
    rows.append(m + " & " + " & ".join(cells))
table("A4. Calibration per learner and corpus: validation-fitted temperature, pooled test ECE before/after temperature scaling, and "
      "AUROC of the entropy confidence for the learner's own correctness (unchanged by temperature scaling).", "lccccccccc",
      "Learner & \\multicolumn{3}{c}{EV-Norm} & \\multicolumn{3}{c}{EV-Public} & \\multicolumn{3}{c}{EV-Raw}", rows,
      size="\\scriptsize\\setlength{\\tabcolsep}{2.5pt}")
rows = []
for r in [U, "Uniform + TS", "Entropy confidence", "SA-WSV (bootstrap)", "SA-WSV + TS", "SA-WSV, rank confidence", "SA-WSV w/o confidence",
          "SA-WSV (binomial)"]:
    cells = []
    for ds in DS:
        f = C["fusion"][ds]["rules"][r]
        g = lambda k: f[k]["mean"] if isinstance(f[k], dict) else f[k]
        cells.append(f"{pp(g('f1_cv')):.2f} & {g('ece'):.3f} & {g('nll'):.3f}")
    rows.append(nice(r) + " & " + " & ".join(cells))
table("A4a. Fused posteriors: macro-F1, pooled ECE and log-loss of the ensemble posterior (mean over seeds).", "lccccccccc",
      "Rule & \\multicolumn{3}{c}{EV-Norm} & \\multicolumn{3}{c}{EV-Public} & \\multicolumn{3}{c}{EV-Raw}", rows,
      size="\\scriptsize\\setlength{\\tabcolsep}{2.5pt}")
rows = []
for k in AB["per_corpus"]["EV-Norm"]:
    if k == U:
        continue
    rows.append(k.replace("gamma=", "$\\gamma$=").replace(" shrunk", ", shrunk").replace(" raw", ", raw") + " & " +
                " & ".join(f"{pp(AB['per_corpus'][lab][k]['mean']):.2f} ({s2(pp(AB['per_corpus'][lab][k]['diff_vs_uniform']))})"
                           for lab in ("EV-Norm", "EV-Public", "EV-Raw")))
table("A4b. Reliability-transform ablation (R2.2): reliability statistic $\\times$ transform $\\times$ shrinkage, with ($\\gamma=1$) "
      "and without ($\\gamma=0$) the entropy confidence; macro-F1 and difference to uniform voting.", "lccc",
      "Variant & EV-Norm & EV-Public & EV-Raw", rows, size="\\scriptsize")

# ---------------------------------------------------------------- A5 transfer
rows = []
for k, d in T["directions"].items():
    c_ = d["c"]["rules"][U]
    cv = d["d"]["ci_vs_uniform"]
    lex = T["g_lexical"]["pairs"].get(k, {})
    rows.append(f"{k.replace(' -> ', '$\\to$')} & {d['e']['uniform_f1']['mean']:.1f} & {d['a']['cells'][U]['drop']['mean']:.1f} & "
                f"{c_['rec_neg']['drop']['mean']:.1f}/{c_['rec_pos']['drop']['mean']:.1f} & {d['b']['spearman_cell_means']:.1f} & "
                f"{s2(pp(cv['SA-WSV w/o confidence']['diff'] / 100))} & {s2(pp(cv['SA-WSV (bootstrap)']['diff'] / 100))} & "
                f"{s2(pp(cv['Stacking (LR)']['diff'] / 100))} & {d['e']['gain_over_uniform']['oracle_convex_f1']['mean']:.2f} & "
                f"{lex.get('jaccard', float('nan')):.1f} & {lex.get('oov_token', float('nan')):.1f}")
table("A5. Zero-shot transfer per direction (seeds 1--4 $\\times$ 5 folds): uniform macro-F1 on the target and its drop from in "
      "domain; recall drop negative/positive; Spearman correlation of the learner order; difference to uniform for SA-WSV$_0$, "
      "SA-WSV$_1$ and stacking; gain of oracle convex weights fitted on target labels; word-type Jaccard overlap and target-token "
      "OOV rate (\\%).", "lcccccccccc",
      "Direction & F1 & drop & rec. drop & $\\rho_S$ & SA$_0$ & SA$_1$ & stack & oracle & Jacc. & OOV", rows,
      size="\\scriptsize\\setlength{\\tabcolsep}{2pt}")

# ---------------------------------------------------------------- A6 annotation
r = A["reliability"]
rows = [f"Inter-pass agreement & {100*r['agreement']:.1f}\\% of {r['n_both']} posts; $\\kappa$ = {r['kappa']:.3f} "
        f"[{r['kappa_ci95'][0]:.3f}, {r['kappa_ci95'][1]:.3f}] (= PABAK, $p_e$=0.5)",
        f"Revised posts & {r['revised_total']} ({100*r['revised_share_labelled']:.1f}\\% of labelled); "
        f"{r['neg_to_pos']} negative$\\to$positive, {r['pos_to_neg']} positive$\\to$negative"]
for lab, c_ in A["corpora"].items():
    rs = ", ".join(f"{v['rr']:.1f} [{v['rr_ci95_katz'][0]:.1f}, {v['rr_ci95_katz'][1]:.1f}]" for v in c_["rr_per_seed_K5"].values())
    rows += [f"{lab}: risk ratio per seed (K=5) & {rs}",
             f"{lab}: pooled K={c_['K_pooled']} & RR {c_['rr_pooled']['rr']:.1f} [{c_['rr_pooled']['rr_ci95_katz'][0]:.1f}, "
             f"{c_['rr_pooled']['rr_ci95_katz'][1]:.1f}]; all-wrong {c_['groups_pooled']['all_wrong']['n']} posts, "
             f"{100*c_['groups_pooled']['all_wrong']['revised_rate']:.1f}\\% revised",
             f"{lab}: ensemble accuracy & revised {100*c_['ensemble_accuracy']['revised']:.1f}\\%, unrevised "
             f"{100*c_['ensemble_accuracy']['unrevised_labelled']:.1f}\\%; agrees with first annotator on revised "
             f"{100*c_['side_taking_revised']['agrees_with_first_pass']:.1f}\\%",
             f"{lab}: AUROC for revision & ambiguity {c_['auroc_revision']['ambiguity_1_minus_p_final']['auroc']:.3f}, entropy "
             f"{c_['auroc_revision']['ensemble_entropy']['auroc']:.3f}, share wrong {c_['auroc_revision']['share_wrong_k_over_K']['auroc']:.3f}",
             f"{lab}: confident learning & {c_['confident_learning']['n_flagged']} flagged ({100*c_['confident_learning']['flag_rate']:.1f}\\%), "
             f"{100*c_['confident_learning']['revised_rate_flagged']:.1f}\\% revised vs {100*c_['confident_learning']['revised_rate_unflagged']:.1f}\\%; "
             f"recall {100*c_['confident_learning']['recall_of_revised']:.1f}\\%; residual issue rate "
             f"{100*c_['confident_learning']['estimated_label_issue_rate']:.1f}\\%"]
table("A6. Annotation reliability and error concentration on the primary corpus (revision rates on the 1,342 posts labelled in "
      "both passes).", "lp{11.2cm}", "Quantity & Value", rows, size="\\scriptsize")

# ---------------------------------------------------------------- A7 cost
pl = K["gpu_cost"]["per_learner"]
rows = []
for m in LRN:
    tr = " / ".join(f"{pl[ds][m]['train_s']['median']:.0f}" for ds in DS)
    inf = pl["evprimary-formal"][m]["infer_ms_per_inst"]["median"]
    rows.append(f"{m} & {K['citable']['params_M'][m]:.1f} & {K['citable']['peak_mem_GB_max'][m]:.2f} & {tr} & {inf:.2f}")
table(f"A7. Computational cost on one RTX 3070 Ti Laptop GPU (bf16): parameters (M), peak training memory (GB), median fine-tuning "
      f"time per fold (s; EV-Norm / EV-Public / EV-Raw) and test inference (ms per post). All 375 runs: "
      f"{K['citable']['gpu_hours_all_runs_from_medians']:.2f} GPU-hours from medians, {K['citable']['wallclock_hours_all_runs']['hours']:.2f} h "
      f"end to end. Fusion on one CPU core: fitting SA-WSV$_1$ "
      f"{K['citable_cpu_timing']['fit_isolated_ms_median']['SA-WSV (bootstrap)']:.1f} ms per fold (binomial version "
      f"{K['citable_cpu_timing']['fit_isolated_ms_median']['SA-WSV (binomial)']:.2f} ms), applying it "
      f"{K['citable_cpu_timing']['apply_us_per_inst_median']['SA-WSV (bootstrap)']:.2f} $\\mu$s per post.",
      "lcccc", "Learner & Params & Memory & Train (s) & Infer (ms)", rows, size="\\scriptsize")

open(os.path.join(HERE, "response_tables.tex"), "w", encoding="utf-8").write("\n".join(L))
print("wrote response_tables.tex with", sum(1 for x in L if x.startswith("\\begin{table}")), "tables")
