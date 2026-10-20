"""Every number quoted in the prose of main.tex, recomputed from experiments/results/rev/*.json.

CHECKS(_) returns (label, string printed in main.tex, value from the analysis, tolerance). verify.py
asserts that the printed string occurs in main.tex and agrees with the value within the tolerance
(rounding of the printed digits). Bounds ("at most", "under") are checked as inequalities here and
listed with value equal to the printed number when they hold.
"""
import json, os, re
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "results", "rev")
J = lambda n: json.load(open(os.path.join(R, n), encoding="utf-8"))
DS = ["evprimary-formal", "evpublic", "evprimary-informal"]
U = "Uniform soft voting"


def bound(printed, value, upper=True):
    """returns the printed number if the bound holds, else the offending value"""
    p = float(printed)
    ok = value <= p if upper else value >= p
    return p if ok else value


def CHECKS(_):
    M, V, C, RB, T, A, K = (J("rev_main.json"), J("rev_variance.json"), J("rev_calibration.json"),
                            J("rev_robustness.json"), J("rev_transfer.json"), J("rev_annotation.json"), J("rev_cost.json"))
    pc = M["per_corpus"]
    mm = [v for ds in DS for k, v in pc[ds]["mono_vs_multi"].items()]
    mb = [100 * v["d_f1_mean"] for v in mm if "mBERT" in [x.strip() for x in v.get("pair", "").split("vs")] or False]
    mb = [100 * v["d_f1_mean"] for ds in DS for k, v in pc[ds]["mono_vs_multi"].items() if k.endswith("mBERT")]
    xl = [100 * v["d_f1_mean"] for ds in ("evprimary-formal", "evprimary-informal")
          for k, v in pc[ds]["mono_vs_multi"].items() if k.endswith("XLM-R")]
    xl_sig = sum(v["n_seeds_sig"] for ds in ("evprimary-formal", "evprimary-informal")
                 for k, v in pc[ds]["mono_vs_multi"].items() if k.endswith("XLM-R"))
    ir_x = 100 * pc["evpublic"]["mono_vs_multi"]["IndoRoBERTa-1.5G vs XLM-R"]["d_f1_mean"]
    LRN = ["IndoBERT-p1", "IndoBERT-1.5G", "IndoRoBERTa-1.5G", "XLM-R", "mBERT"]
    ng = {m: 100 * (pc["evprimary-formal"]["metrics"][m]["f1"]["mean"] - pc["evprimary-informal"]["metrics"][m]["f1"]["mean"])
          for m in LRN}
    sds = [100 * pc[ds]["metrics"][m]["f1"]["sd_seed"] for ds in DS for m in LRN]
    P = M["primary_analysis"]["per_corpus"]; S2 = M["secondary_analysis"]["per_corpus"]
    div = {ds: pc[ds]["diversity"]["mean_over_seeds"] for ds in DS}
    va = V["summary"]["all"]
    vl = {ds: V["summary"][ds]["c_lambda"] for ds in DS}
    ho = {ds: V["summary"][ds]["d_heldout_noise"]["obs_over_pred"] for ds in DS}
    er = {ds: V["summary"][ds]["e_estimation_risk"] for ds in DS}
    risk = RB["risk"]["risk"]
    fam = {f["shrunk"]: f for f in RB["risk"]["families"]}
    st_full, st40 = RB["stress"]["full"]["rho"], RB["stress"]["40"]["rho"]
    red = [C["per_corpus_pooled"][ds]["ts_relative_reduction"]["ece"] for ds in DS]
    au = [C["per_learner"][ds][m]["auroc_raw"]["mean"] for ds in DS for m in LRN]
    fe = {ds: C["fusion"][ds]["rules"] for ds in DS}
    ece = lambda ds, r: fe[ds][r]["ece"]["mean"] if isinstance(fe[ds][r]["ece"], dict) else fe[ds][r]["ece"]
    cm = [C["contested"][ds]["correct_mass_raw"]["mean"] for ds in DS]
    cf = [C["contested"][ds]["count_frac"]["mean"] for ds in DS]
    ts = T["summary"]
    an = A["corpora"]
    rel = A["reliability"]
    ct = K["citable"]
    tm = K["citable_cpu_timing"]
    rr = [v["rr"] for v in an["EV-Norm"]["rr_per_seed_K5"].values()]
    rr2 = [v["rr"] for v in an["EV-Raw"]["rr_per_seed_K5"].values()]
    rank_raw = C["fusion"]["evprimary-informal"]["primary"]["SA-WSV, rank confidence vs SA-WSV (bootstrap)"]["diff"]
    c = [
        # abstract and III-A
        ("mono-mBERT min", "3.5", min(mb), 0.05), ("mono-mBERT max", "7.8", max(mb), 0.05),
        ("m_eff min", "2.1", min(d["m_eff"] for d in div.values()), 0.05),
        ("m_eff max", "2.4", max(d["m_eff"] for d in div.values()), 0.05),
        ("independence overstatement", "2.2-fold", 1 / va["b_centred_noise"]["val"]["discount_ratio_of_means"], 0.05),
        ("SA0 CVaR", "$-0.25$", 100 * risk["SA-WSV w/o confidence"]["cvar5"], 0.005),
        ("learned CVaR low", "$-2.3$", 100 * max(risk[r]["cvar5"] for r in ("Stacking (LR)", "Convex NLL weights", "Ensemble selection")), 0.06),
        ("learned CVaR high", "$-2.6$", 100 * min(risk[r]["cvar5"] for r in ("Stacking (LR)", "Convex NLL weights", "Ensemble selection")), 0.05),
        ("SA0 never loses >0.5", "0.5 points", 0.5 if risk["SA-WSV w/o confidence"]["p_loss_05"] == 0 else 99, 0.0),
        ("stress uniform loss", "0.55 points", -100 * st_full["0.5"]["diff_clean_uniform"][U], 0.005),
        ("max transfer drop", "31 points", ts["uniform_drop"]["max"], 0.5),
        ("xlmr gap min", "2.6 to 4.4", 2.6 if abs(min(xl) - 2.6) <= 0.05 and abs(max(xl) - 4.4) <= 0.05 else min(xl), 0.0),
        ("xlmr seed tests", "23 of 30", 23 if xl_sig == 23 else xl_sig, 0.0),
        ("IndoRoBERTa vs XLM-R", "by 1.5", -ir_x, 0.05),
        ("norm gain min", "by 1.1 to 1.5 points", 1.1 if abs(min(v for k, v in ng.items() if k != "mBERT") - 1.1) <= 0.05
         and abs(max(v for k, v in ng.items() if k != "mBERT") - 1.5) <= 0.05 else -1, 0.0),
        ("norm gain mBERT", "4.0 for mBERT", ng["mBERT"], 0.05),
        ("seed SD range", "0.27 to 1.27", 0.27 if abs(min(sds) - 0.27) <= 0.005 and abs(max(sds) - 1.27) <= 0.005 else min(sds), 0.0),
        ("T>1 share", "90\\%", 100 * C["auroc_invariance"]["frac_T_gt_1"], 0.5),
        ("T ratio", "1.3 to 1.5", 1.3 if 1.25 <= min(C["temperature"][ds]["ratio_learner_means"] for ds in DS)
         and max(C["temperature"][ds]["ratio_learner_means"] for ds in DS) < 1.55 else -1, 0.0),
        ("ECE reduction", "48 to 66", 48 if abs(100 * min(red) - 48) <= 0.5 and abs(100 * max(red) - 66) <= 0.5 else 100 * min(red), 0.0),
        ("AUROC range", "0.73 to 0.84", 0.73 if abs(min(au) - 0.73) <= 0.005 and abs(max(au) - 0.84) <= 0.005 else min(au), 0.0),
        # III-B
        ("primary EV-Norm", "$+0.06$", 100 * P["evprimary-formal"]["diff"], 0.005),
        ("primary EV-Public", "$+0.05$", 100 * P["evpublic"]["diff"], 0.005),
        ("primary EV-Raw", "$+0.49$", 100 * P["evprimary-informal"]["diff"], 0.005),
        ("primary CI Norm lo", "[-0.42", 100 * P["evprimary-formal"]["ci95"][0], 0.005),
        ("primary CI Raw lo", "[0.02", 100 * P["evprimary-informal"]["ci95"][0], 0.005),
        ("primary Holm Raw", "$p=0.12$", P["evprimary-informal"]["p_holm_corpora"], 0.005),
        ("SA0 range lo", "[-0.20", 100 * min(S2[ds]["ci95"][0] for ds in DS), 0.005),
        ("SA0 range hi", "0.49]$ of uniform", 100 * max(S2[ds]["ci95"][1] for ds in DS), 0.005),
        ("band worst seed", "0.94 points", 100 * pc["evprimary-formal"]["band"]["original_soft_rules"]["worst_seed_range_incl_uniform"], 0.005),
        # III-C
        ("Q Norm", "0.93, 0.88 and 0.92", 0.93 if all(abs(round(div[d]["Q_foldmean"], 2) - q) < 1e-9 for d, q in
                                                    zip(DS, (0.93, 0.88, 0.92))) else -1, 0.0),
        ("error corr", "0.52 to 0.58", 0.52 if abs(min(d["rho_mean"] for d in div.values()) - 0.52) <= 0.005
         and abs(max(d["rho_mean"] for d in div.values()) - 0.58) <= 0.005 else -1, 0.0),
        ("contested", "20 to 29\\%", 20 if abs(100 * min(d["contested"] for d in div.values()) - 20) <= 0.5
         and abs(100 * max(d["contested"] for d in div.values()) - 29) <= 0.5 else -1, 0.0),
        ("seed Q", "(Q 0.97)", pc["evprimary-formal"]["diversity"]["seed_diversity"]["Q_same_learner_across_seeds"], 0.005),
        ("boot/binom", "ratio 1.02", va["a_marginal_variance"]["bootstrap_over_binomial"]["mean"], 0.005),
        ("delta/binom", "ratio 1.01)", va["a_marginal_variance"]["delta_over_binomial"]["mean"], 0.0051),
        ("replicate corr", "correlated at 0.55", va["b_centred_noise"]["val"]["rho_F_replicates"]["mean"], 0.0051),
        ("discount", "is 0.46 of", va["b_centred_noise"]["val"]["discount_ratio_of_means"], 0.005),
        ("lambda binom", "0.34, 0.65 and 0.45", 0.34 if all(abs(vl[d]["js_binomial"]["mean"] - x) <= 0.005 for d, x in
                                                          zip(DS, (0.34, 0.65, 0.45))) else -1, 0.0),
        ("lambda paired", "0.67, 0.83 and 0.66", 0.67 if all(abs(vl[d]["js_boot_paired"]["mean"] - x) <= 0.005 for d, x in
                                                          zip(DS, (0.67, 0.83, 0.66))) else -1, 0.0),
        ("lambda zero", "3 of 75 folds instead of 15", 3 if sum(vl[d]["js_boot_paired"]["n_zero"] for d in DS) == 3
         and sum(vl[d]["js_binomial"]["n_zero"] for d in DS) == 15 else -1, 0.0),
        ("heldout binom", "0.48 to 0.66", 0.48 if abs(min(ho[d]["binomial"]["value"] for d in DS) - 0.48) <= 0.0051
         and abs(max(ho[d]["binomial"]["value"] for d in DS) - 0.66) <= 0.005 else -1, 0.0),
        ("heldout paired", "0.95 to 1.43", 0.95 if abs(min(ho[d]["paired"]["value"] for d in DS) - 0.95) <= 0.005
         and abs(max(ho[d]["paired"]["value"] for d in DS) - 1.43) <= 0.005 else -1, 0.0),
        ("grand mean risk", "52 and 133", 52 if abs(100 * (er["evprimary-informal"]["grand_mean"]["dev_over_raw"]["value"] - 1) - 52) <= 0.5
         and abs(100 * (er["evpublic"]["grand_mean"]["dev_over_raw"]["value"] - 1) - 133) <= 0.5 else -1, 0.0),
        ("shrink risk", "5 to 11\\%", 5 if abs(100 * (1 - max(er[d]["js_boot_paired"]["dev_over_raw"]["value"] for d in DS)) - 5) <= 0.5
         and abs(100 * (1 - min(er[d]["js_boot_paired"]["dev_over_raw"]["value"] for d in DS)) - 11) <= 0.5 else -1, 0.0),
        # III-D
        ("learned loss share", "30 to 44", 30 if abs(100 * min(risk[r]["p_loss_05"] for r in ("Stacking (LR)", "Convex NLL weights", "Ensemble selection")) - 30) <= 0.5
         and abs(100 * max(risk[r]["p_loss_05"] for r in ("Stacking (LR)", "Convex NLL weights", "Ensemble selection")) - 44) <= 0.5 else -1, 0.0),
        ("best single loss share", "loses in 69", 100 * risk["Best single (val-selected)"]["p_loss_05"], 0.5),
        ("SA0 mean", "$+0.05$ points on average", 100 * risk["SA-WSV w/o confidence"]["mean"], 0.005),
        ("family gain", "gain 0.26 points", 100 * fam["SA-WSV w/o confidence"]["vs_unshrunk"]["cvar5_gap"], 0.005),
        ("family CI", "0.21 to 0.33", 0.21 if abs(100 * fam["SA-WSV w/o confidence"]["vs_unshrunk"]["cvar5_gap_ci"][0] - 0.21) <= 0.005
         and abs(100 * fam["SA-WSV w/o confidence"]["vs_unshrunk"]["cvar5_gap_ci"][1] - 0.33) <= 0.005 else -1, 0.0),
        ("SA1 mean", "($+0.17$)", 100 * risk["SA-WSV (bootstrap)"]["mean"], 0.005),
        ("SA1 tail ratio", "2.5 times", risk["SA-WSV (bootstrap)"]["cvar5"] / risk["SA-WSV w/o confidence"]["cvar5"], 0.05),
        ("SA1 CVaR", "($-0.62$)", 100 * risk["SA-WSV (bootstrap)"]["cvar5"], 0.005),
        ("entropy CVaR", "($-0.73$)", 100 * risk["Entropy confidence"]["cvar5"], 0.005),
        ("regret SA0", "$-0.44$ points", 100 * RB["risk"]["regret_by_size"]["SA-WSV w/o confidence"]["20"]["cvar5"], 0.005),
        ("regret learned", "$-2.8$ to $-3.6$", -2.8 if abs(100 * max(RB["risk"]["regret_by_size"][r]["20"]["cvar5"] for r in ("Stacking (LR)", "Convex NLL weights", "Ensemble selection")) + 2.8) <= 0.05
         and abs(100 * min(RB["risk"]["regret_by_size"][r]["20"]["cvar5"] for r in ("Stacking (LR)", "Convex NLL weights", "Ensemble selection")) + 3.6) <= 0.05 else 0, 0.0),
        # stress
        ("stress entropy", "weighting 0.49", -100 * st_full["0.5"]["diff_clean_uniform"]["Entropy confidence"], 0.005),
        ("stress within", "within 0.05 points", bound("0.05", max(-100 * min(st_full[r]["diff_clean_uniform"][k] for r in st_full)
                                                               for k in ("Log-odds weighting", "SA-WSV w/o confidence", "SA-WSV (bootstrap)")) - 0.005), 0.0),
        ("stress lambda", "rises to 0.99", st_full["0.5"]["lam_boot"], 0.005),
        ("stress weight", "0.20 to 0.01", 0.20 if abs(st_full["0.0"]["w_corrupted_sawsv_boot"] - 0.20) <= 0.005
         and abs(st_full["0.5"]["w_corrupted_sawsv_boot"] - 0.01) <= 0.005 else -1, 0.0),
        ("stress learned 40", "0.4 to 0.5 points", 0.4 if abs(-100 * max(st40["0.5"]["diff_clean_uniform"][r] for r in ("Stacking (LR)", "Convex NLL weights", "Ensemble selection")) - 0.4) <= 0.05
         and abs(-100 * min(st40["0.5"]["diff_clean_uniform"][r] for r in ("Stacking (LR)", "Convex NLL weights", "Ensemble selection")) - 0.5) <= 0.05 else -1, 0.0),
        # III-E
        ("correct mass", "0.65 to 0.67", 0.65 if abs(min(cm) - 0.65) <= 0.005 and abs(max(cm) - 0.67) <= 0.005 else -1, 0.0),
        ("uniform mass", "against 0.60", sum(cf) / 3, 0.005),
        ("SA1 ECE", "0.043, 0.060 and 0.039", 0.043 if all(abs(ece(d, "SA-WSV (bootstrap)") - x) <= 0.0005 for d, x in
                                                         zip(DS, (0.043, 0.060, 0.039))) else -1, 0.0),
        ("uniform ECE", "0.017 to 0.020", 0.017 if all(0.0165 <= ece(d, r) < 0.0205 for d in DS for r in (U, "SA-WSV w/o confidence")) else -1, 0.0),
        ("SA1+TS Public", "(to 0.032)", ece("evpublic", "SA-WSV + TS"), 0.0005),
        ("rank conf", "$-0.72$ points", 100 * rank_raw, 0.005),
        # III-F
        ("transfer mean drop", "15.3 points", ts["uniform_drop"]["mean"], 0.05),
        ("transfer gain", "gain of 2.6", -ts["uniform_drop"]["min"], 0.05),
        ("transfer worst", "loss of 30.9", ts["uniform_drop"]["max"], 0.05),
        ("pos recall", "23.6 points", ts["recall_uniform"]["rec_pos_drop"]["mean"], 0.05),
        ("neg recall", "recall by 5.5", ts["recall_uniform"]["rec_neg_drop"]["mean"], 0.05),
        ("xlmr drop", "(13.2 points)", ts["learner_drop"]["XLM-R"]["mean"], 0.05),
        ("mbert drop", "(20.1)", ts["learner_drop"]["mBERT"]["mean"], 0.05),
        ("err corr shift", "0.55 to 0.47", 0.55 if abs(ts["diversity"]["error_corr"]["indomain"]["mean"] - 0.55) <= 0.005
         and abs(ts["diversity"]["error_corr"]["transfer"]["mean"] - 0.47) <= 0.005 else -1, 0.0),
        ("spread under 1pp", "in 8 of 10 directions", ts["rule_spread_by_set"]["submission_soft"]["transfer"]["n_directions_under_1pp"], 0.0),
        ("SA0 shift", "by $+0.07$ points", T["d_across_directions"]["SA-WSV w/o confidence"]["mean"], 0.005),
        ("oracle gain", "0.4 to 2.7", 0.4 if abs(ts["oracle_gain_over_uniform"]["oracle_convex_f1"]["min"] - 0.4) <= 0.05
         and abs(ts["oracle_gain_over_uniform"]["oracle_convex_f1"]["max"] - 2.7) <= 0.05 else -1, 0.0),
        # III-G
        ("agreement", "78.9\\%", 100 * rel["agreement"], 0.05), ("kappa", "$\\kappa=0.58$", rel["kappa"], 0.005),
        ("kappa CI", "0.54 to 0.62", 0.54 if abs(rel["kappa_ci95"][0] - 0.54) <= 0.005 and abs(rel["kappa_ci95"][1] - 0.62) <= 0.005 else -1, 0.0),
        ("revised share", "21.1\\%", 100 * rel["revised_share_labelled"], 0.05),
        ("RR Norm", "6.4 to 8.5", 6.4 if abs(min(rr) - 6.4) <= 0.05 and abs(max(rr) - 8.5) <= 0.05 else -1, 0.0),
        ("RR Raw", "9.5 to 12.7", 9.5 if abs(min(rr2) - 9.5) <= 0.05 and abs(max(rr2) - 12.7) <= 0.05 else -1, 0.0),
        ("acc revised", "61 to 67\\%", 61 if abs(100 * min(an[c_]["ensemble_accuracy"]["revised"] for c_ in an) - 61) <= 0.5
         and abs(100 * max(an[c_]["ensemble_accuracy"]["revised"] for c_ in an) - 67) <= 0.5 else -1, 0.0),
        ("acc unrevised", "95 to 96\\%", 95 if abs(100 * min(an[c_]["ensemble_accuracy"]["unrevised_labelled"] for c_ in an) - 95) <= 0.5
         and abs(100 * max(an[c_]["ensemble_accuracy"]["unrevised_labelled"] for c_ in an) - 96) <= 0.5 else -1, 0.0),
        ("side first", "33 to 39\\%", 33 if abs(100 * min(an[c_]["side_taking_revised"]["agrees_with_first_pass"] for c_ in an) - 33) <= 0.5
         and abs(100 * max(an[c_]["side_taking_revised"]["agrees_with_first_pass"] for c_ in an) - 39) <= 0.5 else -1, 0.0),
        ("auroc amb", "0.86 to 0.89", 0.86 if abs(min(an[c_]["auroc_revision"]["ambiguity_1_minus_p_final"]["auroc"] for c_ in an) - 0.86) <= 0.005
         and abs(max(an[c_]["auroc_revision"]["ambiguity_1_minus_p_final"]["auroc"] for c_ in an) - 0.89) <= 0.005 else -1, 0.0),
        ("auroc entropy", "0.83 to 0.84", 0.83 if abs(min(an[c_]["auroc_revision"]["ensemble_entropy"]["auroc"] for c_ in an) - 0.83) <= 0.005
         and abs(max(an[c_]["auroc_revision"]["ensemble_entropy"]["auroc"] for c_ in an) - 0.84) <= 0.005 else -1, 0.0),
        ("CL flags", "flags 3.8\\%", 100 * an["EV-Norm"]["confident_learning"]["flag_rate"], 0.05),
        ("CL precision", "79 to 84\\%", 79 if abs(100 * min(an[c_]["confident_learning"]["revised_rate_flagged"] for c_ in an) - 79) <= 0.5
         and abs(100 * max(an[c_]["confident_learning"]["revised_rate_flagged"] for c_ in an) - 84) <= 0.5 else -1, 0.0),
        # III-H
        ("train primary", "18 to 27 s", 18 if abs(ct["train_s_per_run_median_by_corpus"]["EV-Norm"][0] - 18) <= 0.5
         and abs(max(ct["train_s_per_run_median_by_corpus"]["EV-Norm"][1], ct["train_s_per_run_median_by_corpus"]["EV-Raw"][1]) - 27) <= 0.5 else -1, 0.0),
        ("train public", "46 to 65 s", 46 if abs(ct["train_s_per_run_median_by_corpus"]["EV-Public"][0] - 46) <= 0.5
         and abs(ct["train_s_per_run_median_by_corpus"]["EV-Public"][1] - 65) <= 0.5 else -1, 0.0),
        ("gpu hours", "3.4 GPU-hours", ct["gpu_hours_all_runs_from_medians"], 0.05),
        ("fit ms", "14 to 38 ms", 14 if abs(min(v["fit_isolated_ms"] for v in tm["per_corpus_median_SA-WSV (bootstrap)"].values()) - 14) <= 0.5
         and abs(max(v["fit_isolated_ms"] for v in tm["per_corpus_median_SA-WSV (bootstrap)"].values()) - 38) <= 0.5 else -1, 0.0),
        ("lambda MC", r"error in $\lambda$ 0.005", ct["B1000_lambda_mc_sd_median"], 0.0005),
        # method
        ("F vs A bound", "at most 0.006", bound("0.006", va["g_f1_vs_accuracy"]["abs_diff"]["max"]), 0.0),
        ("weight bound", "at most 0.003", bound("0.003", va["g_f1_vs_accuracy"]["w_maxabs_diff"]["raw"]["max"]), 0.0),
    ]
    out = []
    for label, printed, value, tol in c:
        m = re.search(r"[-+]?\d+(?:\.\d+)?", printed.replace("$", ""))
        ok = m is not None and abs(float(m.group(0)) - float(value)) <= tol + 1e-9
        out.append((label, printed, ok, float(value)))
    return out
