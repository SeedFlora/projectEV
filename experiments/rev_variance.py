"""Sampling variance of validation macro-F1 and the shrinkage estimators.

Reviewer 2 (comment 1) notes that F(1-F)/n, the variance in the original James-Stein factor, is not the
sampling variance of macro-F1; reviewer 1 asks for further shrinkage estimators. Per unit the script
(a) measures the variance of each learner's validation macro-F1 by the binomial formula, a paired bootstrap
and the delta method; (b) measures the noise in the centred vector d = F - mean(F), which is what the
shrinkage factor needs; (c) computes lambda under every estimator in rev_lib; (d) checks each noise model
against the fold's held-out test portion; (e) scores the estimators by held-out estimation risk; (f) measures
how far the resulting weights are from uniform; (g) compares macro-F1 with accuracy and counts the
effectively independent voters.

Design decisions
* A unit is one (corpus, fine-tuning seed, fold), over every seed complete for that corpus
  (L.complete_seeds), so the script runs unchanged once all five seeds exist; the seeds used are stored, and
  a PRELIMINARY banner is printed while any corpus has fewer than five.
* One paired bootstrap per split and unit: B = 2000 i.i.d. resamples of the validation split, then B of the
  fold's test portion, both drawn from L.fold_rng(seed, ds, fold, salt=5); within a split all five learners
  share the indices. This is the construction of the revised SA-WSV (rev_lib.Ctx, which draws B = 1000 with
  salt 0); the Monte Carlo difference in lambda is reported, and a class-stratified bootstrap (salt 6) is a
  sensitivity check only.
* Delta method: binary macro-F1 is a function g of the four confusion-matrix cell proportions p, which are
  multinomial under i.i.d. sampling, so Var = g'(diag(p) - pp')g / n with a central-difference gradient.
* Everything fitted (lambda, shrunken F, weights) uses the validation split only. Test labels enter only the
  held-out checks (d) and (e), which score fixed validation-side estimates; the quantities chosen with test
  labels, the risk-minimising common lambda and noise multiplier, are named "oracle".
* Results lead per corpus. EV-Norm and EV-Raw are two renderings of the same posts with the same folds and
  labels, so they are also summarised jointly ("evprimary", printed Norm+Raw) as one corpus for inference.
  The pooled summary "all" keeps in "value"/"ci95"/"mean" the ratio of sums or mean over all units, which
  weights the corpora by noise magnitude and seed count and counts the primary corpus twice; its
  "equal_corpus_weights" and "mean_equal_corpus_weights" entries weight the two corpora (Norm+Raw and
  EV-Public) equally, and these are what the report prints as pooled*.
* Units of one fold share their validation and test instances across seeds, and EV-Norm and EV-Raw share
  theirs too, so their noise may be correlated. CIs over units come from a two-stage bootstrap (10,000
  replicates): folds with replacement within each corpus, then seeds with replacement within every drawn
  fold, as in the two-stage bootstrap of the main comparisons; resampling folds alone would rest on five
  clusters only. Same-seed runs of EV-Norm and EV-Raw also share initialisation and data order, so one seed
  draw is applied to both renderings whenever they have the same seeds. The naive i.i.d.-unit bootstrap is
  stored as "ci95_units".
* The oracle common lambda, sum d_val.d_test / sum ||d_val||^2, is a ||d_val||^2-weighted mean of the per-unit
  optimal lambdas, so it is compared only with the estimators' lambdas weighted the same way, and with the
  oracle multiplier kappa* of the paired noise trace (kappa = 1 is the paired estimator).
* At lambda = 0 the reliability weights are equal but SA-WSV keeps the entropy confidence c_m(x), so the rule
  reduces to entropy-confidence voting, not to uniform soft voting; this is checked on every test fold.
* The by-epoch split of the optimism F_val - F_test carries CIs and a learner-specific (unit-centred) part;
  it can be consistent with checkpoint selection on the validation split but is not a causal test.
* Spread-to-noise keeps the paper's definition, mean range of validation F over mean binomial SE of a
  validation macro-F1. The correlation-aware variant divides the range by the independence-equivalent SE of
  learner differences, sqrt(tr Sigma_d / (M-1)); it is a different quantity, not an update of the paper's.
"""
import os, sys, json, time
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rev_lib as L

B, SALT, SALT_STRAT, N_CI = 2000, 5, 6, 10000
M = len(L.MO)
OUT_JSON = os.path.join(L.ROOT, "experiments", "results", "rev", "rev_variance.json")
OUT_FIG = os.path.join(L.ROOT, "paper", "fig_lambda.pdf")
EST = OrderedDict([("raw", "raw F"), ("js_binomial", "JS binomial (original)"),
                   ("js_boot_marginal", "JS bootstrap, marginal"), ("js_boot_paired", "JS paired bootstrap (revised)"),
                   ("bock", "Bock"), ("eb", "empirical Bayes"), ("grand_mean", "grand mean (lambda=0)")])
SHRINK = ["js_binomial", "js_boot_marginal", "js_boot_paired", "bock", "eb"]
NOISE = OrderedDict([("binomial", "binomial, independent"), ("marginal", "marginal bootstrap"),
                     ("paired", "paired bootstrap")])
STYLE = {"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--", "axes.spines.top": False,
         "axes.spines.right": False, "figure.dpi": 300, "font.family": "serif",
         "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42}
MARK = {"evprimary-formal": ("o", "#0072B2"), "evpublic": ("s", "#E69F00"), "evprimary-informal": ("^", "#009E73")}
LAB = dict(L.DS_LAB, evprimary="Norm+Raw", all="pooled*")
CI_ID = dict(L.DS_ID, evprimary=8, all=9)            # RNG stream of each summary's CI resampling
KAPPA = np.round(np.linspace(0.0, 8.0, 801), 2)     # grid of the oracle noise multiplier
KAPPA_SHOW = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def group(ds):
    """corpus for inference: EV-Norm and EV-Raw are two renderings of the same posts (same folds and labels)"""
    return "evprimary" if ds.startswith("evprimary") else ds


# ======================================================================== per unit
def _mf1_cells(p):
    """binary macro-F1 from confusion-matrix cells (..., 4) ordered TP, FN, FP, TN (class 1 positive)"""
    tp, fn, fp, tn = np.moveaxis(p, -1, 0)
    return tp / (2 * tp + fn + fp) + tn / (2 * tn + fn + fp)


def delta_var(y, pred, h=1e-6):
    """delta-method variance of each learner's macro-F1 under multinomial sampling of its confusion matrix"""
    y = np.asarray(y)[None]
    p = np.stack([(pred == 1) & (y == 1), (pred == 0) & (y == 1), (pred == 1) & (y == 0),
                  (pred == 0) & (y == 0)], -1).mean(1)
    E = np.eye(4) * h
    g = (_mf1_cells(p[:, None] + E) - _mf1_cells(p[:, None] - E)) / (2 * h)
    return ((g * g * p).sum(1) - (g * p).sum(1) ** 2) / y.shape[1]


def noise(Fb):
    """marginal variances, paired trace tr(Sigma_d), its independence value and the mean correlation of the
    learners' bootstrap replicates of F"""
    var = Fb.var(1, ddof=1)
    R = np.corrcoef(Fb)
    return {"var": var, "tr": float((Fb - Fb.mean(0)).var(1, ddof=1).sum()), "tr_indep": float((M - 1) * var.mean()),
            "rho": float(R[~np.eye(M, dtype=bool)].mean())}


def estimates(F, Fb, n):
    """validation-only estimates of the learners' macro-F1: name -> (F_tilde, lambda, info)"""
    out = OrderedDict(raw=(F.copy(), 1.0, {}))
    for k, (Fs, info) in zip(SHRINK, (L.shrink_js_binomial(F, n), L.shrink_js_boot_marginal(F, Fb),
                                      L.shrink_js_boot_paired(F, Fb), L.shrink_bock(F, Fb), L.shrink_eb(F, Fb))):
        out[k] = (np.asarray(Fs, float), float(info["lam"]), info)
    out["grand_mean"] = (np.full(len(F), F.mean()), 0.0, {})
    return out


def wnorm(F):
    w = L.logodds(F)
    return w / w.sum()


def unit(ds, seed, fold):
    d = L.load_fold(ds, fold, seed)
    yv, yt, pv, pt = d["yv"], d["yt"], d["Pv"].argmax(-1), d["Pt"].argmax(-1)
    nv, nt = len(yv), len(yt)
    rng = L.fold_rng(seed, ds, fold, salt=SALT)
    iv, it = rng.integers(0, nv, size=(B, nv)), rng.integers(0, nt, size=(B, nt))
    Fv, Av, Ft = L.macro_f1_batch(yv, pv), L.accuracy_batch(yv, pv), L.macro_f1_batch(yt, pt)
    Fbv, Abv = L.macro_f1_batch(yv, pv, iv), L.accuracy_batch(yv, pv, iv)
    Fbt = L.macro_f1_batch(yt, pt, it)
    Fbs = L.macro_f1_batch(yv, pv, L.stratified_boot_index(yv, B, L.fold_rng(seed, ds, fold, salt=SALT_STRAT)))
    Nv, Nt, Ns = noise(Fbv), noise(Fbt), noise(Fbs)
    est = estimates(Fv, Fbv, nv)
    assert abs(est["js_boot_paired"][2]["noise"] - Nv["tr"]) < 1e-12
    ctx = L.Ctx(d["Pv"], yv, L.fold_rng(seed, ds, fold))          # the resamples SA-WSV (bootstrap) itself uses
    dv, dt = Fv - Fv.mean(), Ft - Ft.mean()
    w = OrderedDict((k, wnorm(v[0])) for k, v in est.items())
    okv, okt = pv == yv, pt == yt
    # epoch of the kept checkpoint (first maximum of validation macro-F1, as in training); 0 if no history
    best = [int(np.argmax([e["f1"] for e in d["hist"][m]])) + 1 if d["hist"].get(m) else 0 for m in L.MO]
    # at lambda = 0 every reliability weight is equal, but SA-WSV keeps the entropy confidence c_m(x); compare its
    # test predictions with entropy-confidence voting and with uniform soft voting
    ce = L.conf_entropy(d["Pt"])
    p_ent, p_uni = L.fuse(d["Pt"], None, ce).argmax(-1), L.fuse(d["Pt"]).argmax(-1)
    lam0 = OrderedDict()
    for k in SHRINK:
        if est[k][1] <= 0:
            p = L.fuse(d["Pt"], L.logodds(est[k][0]), ce).argmax(-1)
            lam0[k] = {"n_test": nt, "n_diff_entropy": int((p != p_ent).sum()), "n_diff_uniform": int((p != p_uni).sum())}
    return dict(
        ds=ds, seed=seed, fold=fold, n_val=nv, n_test=nt, F_val=Fv, A_val=Av, F_test=Ft, best_epoch=best,
        var_binom=Fv * (1 - Fv) / nv, var_boot=Nv["var"], var_delta=delta_var(yv, pv),
        var_acc_binom=Av * (1 - Av) / nv, var_acc_boot=Abv.var(1, ddof=1), var_boot_strat=Ns["var"],
        var_binom_test=Ft * (1 - Ft) / nt, var_boot_test=Nt["var"],
        tr_paired_val=Nv["tr"], tr_indep_val=Nv["tr_indep"], rho_val=Nv["rho"], tr_paired_strat=Ns["tr"],
        tr_paired_test=Nt["tr"], tr_indep_test=Nt["tr_indep"], rho_test=Nt["rho"],
        lam=OrderedDict((k, est[k][1]) for k in SHRINK), bock_eff_dim=float(est["bock"][2]["eff_dim"]),
        lam_paired_method=L.shrink_js_boot_paired(ctx.F, ctx.Fb)[1]["lam"],
        lam_paired_strat=L.shrink_js_boot_paired(Fv, Fbs)[1]["lam"],
        Ftilde=OrderedDict((k, v[0]) for k, v in est.items()),
        # (d) if validation and test estimate the same F: E||d_val - d_test||^2 = tr Sigma_d(val) + tr Sigma_d(test);
        # the binomial prediction uses each split's own plug-in F
        dd_obs=float(((dv - dt) ** 2).sum()),
        dd_pred={"binomial": (M - 1) * float(np.mean(Fv * (1 - Fv)) / nv + np.mean(Ft * (1 - Ft)) / nt),
                 "marginal": Nv["tr_indep"] + Nt["tr_indep"], "paired": Nv["tr"] + Nt["tr"]},
        # common component c = mean_m(F_val - F_test); sum_m (F_val - F_test)^2 = M c^2 + ||d_val - d_test||^2
        cm_obs=float((Fv - Ft).mean()), cm_pred=float(Fbv.mean(0).var(ddof=1) + Fbt.mean(0).var(ddof=1)),
        risk_dev=OrderedDict((k, float(((v[0] - v[0].mean() - dt) ** 2).sum())) for k, v in est.items()),
        risk_level=OrderedDict((k, float(((v[0] - Ft) ** 2).sum())) for k, v in est.items()),
        dv_dt=float(dv @ dt), dv_dv=float(dv @ dv), dt_dt=float(dt @ dt),
        w=w, l1_uniform=OrderedDict((k, float(np.abs(v - 1.0 / M).sum())) for k, v in w.items()),
        l1_binom_vs_paired=float(np.abs(w["js_binomial"] - w["js_boot_paired"]).sum()),
        w_F_vs_A_raw=float(np.abs(w["raw"] - wnorm(Av)).max()),
        w_F_vs_A_paired=float(np.abs(w["js_boot_paired"] - wnorm(L.shrink_js_boot_paired(Av, Abv)[0])).max()),
        ev_val=L.effective_voters(okv), ev_test=L.effective_voters(okt),
        q_val=L.q_statistic(okv)[0], q_test=L.q_statistic(okt)[0], lam0_rule=lam0)


# ===================================================================== summaries
def mmm(x, G=None):
    """mean, min and max; given corpus masks G over the units, also the equal-corpus-weight mean"""
    x = np.asarray(x, float).ravel()
    out = {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}
    if G is not None:
        k = len(x) // len(G[0])                          # one value per unit, or one per learner-unit
        out["mean_equal_corpus_weights"] = float(np.mean([x[np.repeat(g, k)].mean() for g in G]))
    return out


def _ci(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))] if len(r) else [float("nan")] * 2


def boot_stat(f, n, Ws, G=None, feq=None):
    """statistic f(W) of the unit multiplicities W (..., n): its value at W = 1 and a percentile CI per resampling
    scheme. Given corpus masks G, also its equal-corpus-weight version feq (default: the mean over corpora of f
    within each corpus) with the two-stage CI. Replicates where f is undefined are dropped and counted."""
    one = np.ones(n)
    out = OrderedDict()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["value"] = float(f(one))
        for k, W in Ws.items():
            r = np.asarray(f(W), float)
            out[k] = _ci(r)
            if not np.isfinite(r).all():
                out[f"{k}_n_undefined"] = int((~np.isfinite(r)).sum())
        if G is not None:
            feq = feq or (lambda W: np.mean([f(W * g) for g in G], 0))
            r = np.asarray(feq(Ws["ci95"]), float)
            out["equal_corpus_weights"] = OrderedDict(value=float(feq(one)), ci95=_ci(r))
            if not np.isfinite(r).all():
                out["equal_corpus_weights"]["ci95_n_undefined"] = int((~np.isfinite(r)).sum())
    return out


def ratio_ci(num, den, Ws, G=None):
    """ratio of unit means sum(num)/sum(den) with a percentile CI per resampling scheme"""
    return boot_stat(lambda W: (W @ num) / (W @ den), len(num), Ws, G)


def mean_ci(x, Ws, G=None):
    return boot_stat(lambda W: (W @ x) / W.sum(-1), len(x), Ws, G)


def twostage_W(units, n, rng):
    """(n, U) unit multiplicities of a two-stage bootstrap: folds with replacement within each corpus (EV-Norm
    and EV-Raw form one corpus, as they share their posts and folds), then seeds with replacement within every
    drawn fold. Same-seed runs of EV-Norm and EV-Raw also share splits, labels, initialisation and data order,
    so when both renderings have the same seeds one seed draw is applied to both."""
    W = np.zeros((n, len(units)))
    for g in sorted({group(u["ds"]) for u in units}):
        folds = sorted({u["fold"] for u in units if group(u["ds"]) == g})
        cf = rng.multinomial(len(folds), np.full(len(folds), 1.0 / len(folds)), size=n)
        for j, f in enumerate(folds):
            cells = OrderedDict()                                         # rendering -> {seed: unit column}
            for i, u in enumerate(units):
                if group(u["ds"]) == g and u["fold"] == f:
                    cells.setdefault(u["ds"], OrderedDict())[u["seed"]] = i
            shared = len({tuple(c) for c in cells.values()}) == 1
            for r, ds in enumerate(sorted(cells)):
                sd = list(cells[ds])
                if r == 0 or not shared:
                    # c draws of the fold, each resampling its S seeds: Multinomial(c * S, 1/S) in total
                    draw = rng.multinomial(cf[:, j] * len(sd), np.full(len(sd), 1.0 / len(sd)))
                W[:, [cells[ds][s] for s in sd]] = draw
    return W


def summarise(U, name):
    rng = np.random.default_rng([CI_ID.get(name, 9), 55])
    Ws = {"ci95": twostage_W(U, N_CI, rng),
          "ci95_units": rng.multinomial(len(U), np.full(len(U), 1.0 / len(U)), size=N_CI).astype(float)}
    gs = sorted({group(u["ds"]) for u in U})
    G = [np.array([group(u["ds"]) == g for u in U]) for g in gs] if len(gs) > 1 else None
    col = lambda f: np.array([f(u) if callable(f) else u[f] for u in U], float)
    cat = lambda k: np.concatenate([np.asarray(u[k], float) for u in U])
    mm = lambda x: mmm(x, G)
    rci = lambda num, den: ratio_ci(num, den, Ws, G)
    mci = lambda x: mean_ci(x, Ws, G)
    S = OrderedDict(n_units=len(U))
    if G is not None:
        S["pooling"] = {"corpora": gs, "n_units": [int(g.sum()) for g in G],
                        "equal_corpus_weights": "mean over these corpora of the statistic computed within each"}

    vbin, vb, vd = cat("var_binom"), cat("var_boot"), cat("var_delta")
    S["a_marginal_variance"] = {
        "n_val": sorted({int(u["n_val"]) for u in U}),
        "se_mean": {"binomial": float(np.sqrt(vbin).mean()), "bootstrap": float(np.sqrt(vb).mean()),
                    "delta": float(np.sqrt(vd).mean())},
        "bootstrap_over_binomial": mm(vb / vbin), "delta_over_binomial": mm(vd / vbin),
        "bootstrap_over_delta": mm(vb / vd),
        "accuracy_bootstrap_over_binomial": mm(cat("var_acc_boot") / cat("var_acc_binom")),
        "stratified_over_iid_bootstrap": mm(cat("var_boot_strat") / vb),
        "per_learner": OrderedDict((L.MLAB[m], {k: float(np.mean([u[k][i] for u in U]))
                                                for k in ("F_val", "A_val", "var_binom", "var_boot", "var_delta")})
                                   for i, m in enumerate(L.MO))}

    S["b_centred_noise"] = OrderedDict((sp, {
        "tr_paired": mm(col(f"tr_paired_{sp}")), "tr_independent": mm(col(f"tr_indep_{sp}")),
        "discount": mm(col(f"tr_paired_{sp}") / col(f"tr_indep_{sp}")),
        "discount_ratio_of_means": float(col(f"tr_paired_{sp}").sum() / col(f"tr_indep_{sp}").sum()),
        "discount_ratio_of_means_ci": rci(col(f"tr_paired_{sp}"), col(f"tr_indep_{sp}")),
        "rho_F_replicates": mm(col(f"rho_{sp}"))}) for sp in ("val", "test"))
    S["b_centred_noise"]["val"]["stratified_over_iid_trace"] = mm(col("tr_paired_strat") / col("tr_paired_val"))

    lam = OrderedDict((k, col(lambda u, k=k: u["lam"][k])) for k in SHRINK)
    spread = col(lambda u: u["F_val"].max() - u["F_val"].min())
    se_bin = col(lambda u: np.sqrt(u["var_binom"].mean()))         # SE of a validation macro-F1 (the paper's)
    se_dif = np.sqrt(col("tr_paired_val") / (M - 1))               # independence-equivalent SE of differences
    c = OrderedDict((k, dict(mm(v), n_zero=int((v <= 0).sum()))) for k, v in lam.items())
    c["paired_minus_binomial"] = dict(mm(lam["js_boot_paired"] - lam["js_binomial"]),
                                      frac_paired_larger=float((lam["js_boot_paired"] > lam["js_binomial"]).mean()))
    c["bock_eff_dim"] = dict(mm(col("bock_eff_dim")), n_le_2=int((col("bock_eff_dim") <= 2).sum()))
    # "binomial"/"paired": per-unit ratios; "paper_definition": mean range / mean binomial SE, as in the paper
    c["spread_to_noise"] = {"binomial": mm(spread / se_bin), "paired": mm(spread / se_dif),
                            "paper_definition": rci(spread, se_bin), "range": mm(spread), "se_binomial": mm(se_bin),
                            "range_over_se_of_differences": rci(spread, se_dif)}
    c["monte_carlo"] = {"absdiff_vs_method_B1000_salt0": mm(np.abs(lam["js_boot_paired"] - col("lam_paired_method"))),
                        "absdiff_vs_stratified_bootstrap": mm(np.abs(lam["js_boot_paired"] - col("lam_paired_strat")))}
    c["lambda_zero_rule"] = OrderedDict()
    for k in SHRINK:
        z = [u["lam0_rule"][k] for u in U if k in u["lam0_rule"]]
        c["lambda_zero_rule"][k] = {
            "n_units": len(z), "identical_to_entropy_confidence": sum(v["n_diff_entropy"] == 0 for v in z),
            "identical_to_uniform_soft_voting": sum(v["n_diff_uniform"] == 0 for v in z),
            "instances_differing_from_entropy_confidence": sum(v["n_diff_entropy"] for v in z),
            "instances_differing_from_uniform": sum(v["n_diff_uniform"] for v in z),
            "instances": sum(v["n_test"] for v in z)}
    S["c_lambda"] = c

    obs = col("dd_obs")
    pred = OrderedDict((k, col(lambda u, k=k: u["dd_pred"][k])) for k in NOISE)
    diff = np.stack([u["F_val"] - u["F_test"] for u in U])
    pbin = np.stack([u["var_binom"] + u["var_binom_test"] for u in U])
    pbt = np.stack([u["var_boot"] + u["var_boot_test"] for u in U])
    per = OrderedDict()
    for i, m in [(None, "all learners")] + [(i, L.MLAB[m]) for i, m in enumerate(L.MO)]:
        sl = slice(None) if i is None else [i]
        x, vb_ = diff[:, sl], pbt[:, sl]
        var_about = float(((x - x.mean(0)) ** 2).sum(0).mean() / max(len(x) - 1, 1))
        per[m] = {"optimism": mci(x.mean(1)), "mean_sq_diff": float((x ** 2).mean()),
                  "var_about_optimism": var_about, "pred_var_binomial": float(pbin[:, sl].mean()),
                  "pred_var_bootstrap": float(vb_.mean()),
                  "msd_over_pred_bootstrap": rci((x ** 2).sum(1), vb_.sum(1)),
                  "var_about_optimism_over_pred_bootstrap": var_about / float(vb_.mean())}
    be, z2 = np.stack([u["best_epoch"] for u in U]), diff ** 2 / pbt
    dc = diff - diff.mean(1, keepdims=True)      # learner-specific part of F_val - F_test: the part that enters d
    # test is never used for selection, so its noise is pure sampling noise (paired bootstrap); what the held-out
    # discrepancy leaves after removing it is the noise in d_val, compared with each model's validation term
    implied = obs - col("tr_paired_test")
    pred_val = OrderedDict([("binomial", col(lambda u: (M - 1) * u["var_binom"].mean())),
                            ("marginal", col("tr_indep_val")), ("paired", col("tr_paired_val"))])
    cm, cmp_ = col("cm_obs"), col("cm_pred")
    dh = {"dd_obs_mean": float(obs.mean()),
          "dd_pred_mean": OrderedDict((k, float(v.mean())) for k, v in pred.items()),
          "obs_over_pred": OrderedDict((k, rci(obs, v)) for k, v in pred.items()),
          "val_noise_heldout_implied_over_pred": OrderedDict((k, rci(implied, v)) for k, v in pred_val.items()),
          "common_component": {"mean": mci(cm), "mean_sq_over_pred": rci(cm ** 2, cmp_),
                               "var_about_mean_over_pred": float(cm.var(ddof=1) / cmp_.mean())},
          "per_learner": per}
    if G is not None:
        dh["common_component"]["var_about_mean_over_pred_equal_corpus_weights"] = float(np.mean(
            [cm[g].var(ddof=1) / cmp_[g].mean() for g in G]))
    # by epoch of the kept checkpoint: per-unit sums over the learners whose kept epoch is e, so the CIs come from
    # the same resampling of units; "optimism_centred" is the learner-specific part
    eps = sorted(e for e in set(be.ravel().tolist()) if e > 0)
    sel = lambda e, x: ((be == e) * x).sum(1)
    dh["by_selected_epoch"] = OrderedDict((str(e), {
        "n_learner_units": int((be == e).sum()), "optimism": float(diff[be == e].mean()),
        "sq_diff_over_pred_bootstrap": float(z2[be == e].mean()),
        "optimism_ci": rci(sel(e, diff), sel(e, 1.0)), "optimism_centred": rci(sel(e, dc), sel(e, 1.0)),
        "sq_diff_over_pred_bootstrap_ci": rci(sel(e, z2), sel(e, 1.0))}) for e in eps)
    if len(eps) > 1:
        e1, e2 = eps[0], eps[-1]
        con = lambda x: boot_stat(lambda W: (W @ sel(e1, x)) / (W @ sel(e1, 1.0)) - (W @ sel(e2, x)) / (W @ sel(e2, 1.0)),
                                  len(U), Ws, G)
        dh["by_selected_epoch_contrast"] = {"epochs": [e1, e2], "optimism": con(diff), "optimism_centred": con(dc),
                                            "sq_diff_over_pred_bootstrap": con(z2)}
    S["d_heldout_noise"] = dh

    rd = OrderedDict((k, col(lambda u, k=k: u["risk_dev"][k])) for k in EST)
    rl = OrderedDict((k, col(lambda u, k=k: u["risk_level"][k])) for k in EST)
    trt = col("tr_paired_test")
    e = OrderedDict((k, {
        "dev_risk": float(rd[k].mean()), "level_risk": float(rl[k].mean()),
        "dev_risk_noise_corrected": float((rd[k] - trt).mean()),
        "dev_over_raw": rci(rd[k], rd["raw"]), "level_over_raw": rci(rl[k], rl["raw"]),
        "dev_noise_corrected_over_raw": rci(rd[k] - trt, rd["raw"] - trt),
        "dev_minus_raw": mci(rd[k] - rd["raw"])}) for k in EST)
    e["test_noise_trace_paired_mean"] = float(trt.mean())
    e["test_noise_share_of_raw_dev_risk"] = rci(trt, rd["raw"])
    e["paired_vs_binomial"] = {"dev_ratio": rci(rd["js_boot_paired"], rd["js_binomial"]),
                               "dev_noise_corrected_ratio": rci(rd["js_boot_paired"] - trt, rd["js_binomial"] - trt),
                               "dev_diff": mci(rd["js_boot_paired"] - rd["js_binomial"])}
    # oracle: the one common lambda minimising sum_u ||lambda d_val - d_test||^2, chosen with test labels;
    # its minimum risk is sum d_test^2 - (sum d_val.d_test)^2 / sum d_val^2
    a_, b_, c_ = col("dv_dt"), col("dv_dv"), col("dt_dt")
    ratio = lambda A, Bb, C, R, T: (C - A ** 2 / Bb - T) / (R - T)
    e["oracle_common_lambda"] = dict(rci(a_, b_), **{
        f"dev{tag}_over_raw": boot_stat(lambda W, t=t: ratio(W @ a_, W @ b_, W @ c_, W @ rd["raw"], t * (W @ trt)),
                                        len(U), Ws, G)
        for tag, t in (("", 0.0), ("_noise_corrected", 1.0))})
    # the oracle is the ||d_val||^2-weighted mean of the per-unit optimal lambdas d_val.d_test / ||d_val||^2, so the
    # estimators' lambdas are compared with it only after the same weighting
    e["oracle_common_lambda"]["estimator_lambda_ss_weighted"] = OrderedDict((k, rci(lam[k] * b_, b_)) for k in SHRINK)
    # oracle noise multiplier: lambda_u(kappa) = max(0, 1 - (M-3)/(M-1) kappa tr Sigma_d(val) / ||d_val||^2), so
    # kappa = 1 is the paired-bootstrap estimator and kappa = (M-1) mean_m F(1-F)/n / tr Sigma_d the binomial one
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (M - 3) / (M - 1) * col("tr_paired_val") / b_
        lk = np.where(b_[:, None] > 0, np.clip(1.0 - slope[:, None] * KAPPA[None], 0.0, None), 0.0)
    Rk = lk ** 2 * b_[:, None] - 2.0 * lk * a_[:, None] + c_[:, None]           # (U, K) deviation risk
    assert np.allclose(Rk[:, np.flatnonzero(KAPPA == 1.0)[0]], rd["js_boot_paired"])
    rel = lambda W: (W @ Rk) / np.asarray(W @ rd["raw"])[..., None]
    ks = lambda W: KAPPA[np.argmin(rel(W), -1)]
    e["oracle_noise_multiplier"] = dict(
        boot_stat(ks, len(U), Ws, G, feq=lambda W: KAPPA[np.argmin(np.mean([rel(W * g) for g in G], 0), -1)]),
        grid=[float(KAPPA[0]), float(KAPPA[-1]), float(KAPPA[1] - KAPPA[0])],
        ci95_frac_at_grid_max=float((ks(Ws["ci95"]) == KAPPA[-1]).mean()),
        kappa_binomial_equivalent=mm(pred_val["binomial"] / col("tr_paired_val")),
        dev_over_raw_at_kappa=OrderedDict((f"{k:g}", rci(Rk[:, np.flatnonzero(KAPPA == k)[0]], rd["raw"]))
                                          for k in KAPPA_SHOW))
    S["e_estimation_risk"] = e

    S["f_weights"] = {"l1_from_uniform": OrderedDict((k, mm(col(lambda u, k=k: u["l1_uniform"][k]))) for k in EST),
                      "l1_binomial_vs_paired": mm(col("l1_binom_vs_paired"))}

    Fv, Av = cat("F_val"), cat("A_val")
    dF = np.concatenate([u["F_val"] - u["F_val"].mean() for u in U])
    dA = np.concatenate([u["A_val"] - u["A_val"].mean() for u in U])
    S["g_f1_vs_accuracy"] = {
        "abs_diff": mm(np.abs(Fv - Av)),
        "pearson_r": float(np.corrcoef(Fv, Av)[0, 1]), "pearson_r_within_unit": float(np.corrcoef(dF, dA)[0, 1]),
        "w_maxabs_diff": {"raw": mm(col("w_F_vs_A_raw")), "paired_shrunk": mm(col("w_F_vs_A_paired"))}}
    S["g_effective_voters"] = OrderedDict((sp, {"rho_mean": mm(col(lambda u, sp=sp: u[f"ev_{sp}"]["rho_mean"])),
                                                "m_eff": mm(col(lambda u, sp=sp: u[f"ev_{sp}"]["m_eff"])),
                                                "q_statistic": mm(col(f"q_{sp}"))}) for sp in ("val", "test"))
    return S


# ======================================================================== output
def fig_lambda(U, path):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot([0, 1], [0, 1], color="0.3", lw=0.8, ls=":", zorder=1)
    for ds in L.IN_DOMAIN:
        x = [u["lam"]["js_binomial"] for u in U if u["ds"] == ds]
        y = [u["lam"]["js_boot_paired"] for u in U if u["ds"] == ds]
        mk, colr = MARK[ds]
        ax.scatter(x, y, marker=mk, s=20, color=colr, edgecolors="k", linewidths=0.4, alpha=0.9, zorder=2,
                   label=L.DS_LAB[ds])
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(r"$\lambda$, binomial noise (original)")
    ax.set_ylabel(r"$\lambda$, paired bootstrap (revised)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def _clean(o):
    """JSON-ready copy; floats rounded to 10 significant digits so BLAS-level noise cannot change the file,
    undefined values (NaN) written as null"""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(f"{float(o):.10g}") if np.isfinite(o) else None
    return o


def _m(s):
    """mean of an mmm summary; in the pooled summary the equal-corpus-weight mean"""
    return s.get("mean_equal_corpus_weights", s["mean"])


def _r(r):
    """value and CI of a boot_stat result; in the pooled summary the equal-corpus-weight version"""
    return r.get("equal_corpus_weights", r)


def _f(s, f=".3f"):
    return f"{_m(s):{f}} [{s['min']:{f}}, {s['max']:{f}}]"


def _c(r, f=".3f", k=1.0):
    r = _r(r)
    return f"{k * r['value']:{f}} [{k * r['ci95'][0]:{f}}, {k * r['ci95'][1]:{f}}]"


def _v(r, ref=1.0):
    """position of the 95% CI relative to ref"""
    lo, hi = _r(r)["ci95"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "CI undefined"
    return f"CI < {ref:g}" if hi < ref else f"CI > {ref:g}" if lo > ref else f"CI includes {ref:g}"


def banner(seeds):
    short = OrderedDict((ds, len(s)) for ds, s in seeds.items() if len(s) < len(L.SEEDS))
    if short:
        bar = "!" * 112
        print(f"{bar}\nPRELIMINARY: fewer than {len(L.SEEDS)} complete seeds for "
              + ", ".join(f"{L.DS_LAB[ds]} ({n})" for ds, n in short.items())
              + ". Numbers and qualitative conclusions can still change; regenerate with all seeds before citing.\n"
              + bar)
    return bool(short)


def report(S, seeds):
    rows = L.IN_DOMAIN + ["evprimary", "all"]
    banner(seeds)
    print(f"units = corpus x seed x fold; B = {B} paired bootstrap resamples per split; 95% CIs over units from a "
          f"two-stage bootstrap (folds, then seeds within fold; one seed draw for EV-Norm and EV-Raw when they share seeds)")
    print("rows: per corpus first; Norm+Raw = EV-Norm and EV-Raw jointly (the same posts and folds: one corpus for "
          "inference); pooled* = equal corpus weights (Norm+Raw 1/2, EV-Public 1/2) for means, ratios and their CIs; "
          "counts, fractions, correlations, minima and maxima are over all units")
    print("seeds used: " + "; ".join(f"{L.DS_LAB[ds]} {seeds[ds]} ({S[ds]['n_units']} units)" for ds in L.IN_DOMAIN))

    print("\nheadline, per corpus (95% CI against 1)")
    for k in rows:
        d, e = S[k]["d_heldout_noise"], S[k]["e_estimation_risk"]
        ob, pv, ko = d["obs_over_pred"], e["paired_vs_binomial"]["dev_ratio"], e["oracle_noise_multiplier"]
        print(f"  {LAB[k]:9s} held-out obs/pred: binomial-independent {_c(ob['binomial'], '.2f')} ({_v(ob['binomial'])}), "
              f"paired bootstrap {_c(ob['paired'], '.2f')} ({_v(ob['paired'])}) | deviation risk revised/original "
              f"{_c(pv)} ({_v(pv)}) | oracle noise multiplier kappa* {_c(ko, '.2f')} (paired 1, binomial-equivalent "
              f"{_m(ko['kappa_binomial_equivalent']):.2f})")

    print("\n(a) variance of validation macro-F1 / binomial F(1-F)/n, mean [min, max] over learner-units")
    for k in L.IN_DOMAIN:
        a = S[k]["a_marginal_variance"]
        print(f"  {LAB[k]:9s} n_val={a['n_val'][0]}  SE binomial {a['se_mean']['binomial']:.4f} bootstrap "
              f"{a['se_mean']['bootstrap']:.4f} delta {a['se_mean']['delta']:.4f} | bootstrap/binomial "
              f"{_f(a['bootstrap_over_binomial'])}  delta/binomial {_f(a['delta_over_binomial'])}  "
              f"(accuracy: bootstrap/binomial {_f(a['accuracy_bootstrap_over_binomial'])})")

    print("\n(b) noise in d = F - mean(F): correlation discount = paired tr(Sigma_d) / independence value (M-1) mean "
          "sigma^2; ratio of means [95% CI], then per-unit mean [min, max]")
    for k in rows:
        b = S[k]["b_centred_noise"]
        print(f"  {LAB[k]:9s} val discount {_c(b['val']['discount_ratio_of_means_ci'])}, per unit "
              f"{_f(b['val']['discount'])} (1 - rho = {1 - _m(b['val']['rho_F_replicates']):.3f}, rho of F replicates "
              f"{_f(b['val']['rho_F_replicates'])})  test discount {_c(b['test']['discount_ratio_of_means_ci'])}  "
              f"stratified/iid trace {_m(b['val']['stratified_over_iid_trace']):.3f}")

    print("\n(c) lambda per unit, mean [min, max] (#units with lambda = 0, i.e. equal reliability weights)")
    for k in rows:
        c = S[k]["c_lambda"]
        s2n = c["spread_to_noise"]
        print(f"  {LAB[k]:9s} " + "  ".join(f"{EST[e].split(' (')[0]}: {_f(c[e], '.2f')} ({c[e]['n_zero']})" for e in SHRINK))
        print(f"  {'':9s} paired - binomial {_f(c['paired_minus_binomial'], '+.3f')}, larger in "
              f"{100 * c['paired_minus_binomial']['frac_paired_larger']:.0f}% of units; Bock effective dimension "
              f"{_f(c['bock_eff_dim'], '.2f')} (<= 2 in {c['bock_eff_dim']['n_le_2']}); |lambda(B=2000) - "
              f"lambda(method, B=1000)| max {c['monte_carlo']['absdiff_vs_method_B1000_salt0']['max']:.3f}")
        print(f"  {'':9s} spread-to-noise, paper's definition (mean range of validation F / mean binomial SE of a "
              f"validation macro-F1): {_c(s2n['paper_definition'], '.2f')}, mean range {_m(s2n['range']):.3f}, mean SE "
              f"{_m(s2n['se_binomial']):.4f}; a different quantity, range / independence-equivalent SE of learner "
              f"differences sqrt(tr Sigma_d/(M-1)): {_c(s2n['range_over_se_of_differences'], '.2f')}")
    print("  at lambda = 0 the reliability weights are equal, but SA-WSV keeps the entropy confidence c_m(x), so it "
          "reduces to entropy-confidence voting, not to uniform soft voting. Test folds, all units:")
    for k, v in S["all"]["c_lambda"]["lambda_zero_rule"].items():
        if v["n_units"]:
            print(f"    {EST[k]:30s} lambda = 0 in {v['n_units']} units; test predictions identical to entropy-confidence "
                  f"voting in {v['identical_to_entropy_confidence']}, to uniform soft voting in "
                  f"{v['identical_to_uniform_soft_voting']} ({v['instances_differing_from_uniform']} of {v['instances']} "
                  f"instances differ from uniform)")

    print("\n(d) held-out check: mean ||d_val - d_test||^2 observed / predicted [95% CI]")
    for k in rows:
        d = S[k]["d_heldout_noise"]
        cc = d["common_component"]
        absolute = k != "all"                     # absolute means are over all units, so only per corpus
        print(f"  {LAB[k]:9s} " + (f"observed {d['dd_obs_mean']:.2e} | " if absolute else "") + "  ".join(
            f"{NOISE[n]}: " + (f"pred {d['dd_pred_mean'][n]:.2e}, " if absolute else "")
            + f"obs/pred {_c(d['obs_over_pred'][n], '.2f')} ({_v(d['obs_over_pred'][n])})" for n in NOISE))
        print(f"  {'':9s} validation-side noise implied by held-out data / model: " + "  ".join(
            f"{NOISE[n]} {_c(d['val_noise_heldout_implied_over_pred'][n], '.2f')}" for n in NOISE)
              + f"; common component mean(F_val - F_test): variance about its mean / paired prediction "
                f"{cc.get('var_about_mean_over_pred_equal_corpus_weights', cc['var_about_mean_over_pred']):.2f}")
    print("  optimism F_val - F_test in pp [95% CI]; mean (F_val - F_test)^2 / bootstrap-predicted variance")
    for k in rows:
        p = S[k]["d_heldout_noise"]["per_learner"]
        print(f"  {LAB[k]:9s} " + "  ".join(f"{m}: {_c(v['optimism'], '+.2f', 100)} ({_r(v['msd_over_pred_bootstrap'])['value']:.2f})"
                                            for m, v in p.items()))
    print("  by epoch of the kept checkpoint (first maximum of validation macro-F1): learner-units n, optimism F_val - "
          "F_test in pp, its learner-specific (unit-centred) part, which is what enters d, in pp, and (F_val - F_test)^2 / "
          "bootstrap-predicted variance, each [95% CI]. Larger optimism after an early kept epoch would be consistent "
          "with checkpoint selection on the validation split; this is not a causal test.")
    for k in rows:
        d = S[k]["d_heldout_noise"]
        print(f"  {LAB[k]:9s} " + "  ".join(
            f"epoch {ep}: n={v['n_learner_units']}, {_c(v['optimism_ci'], '+.2f', 100)}, centred "
            f"{_c(v['optimism_centred'], '+.2f', 100)}, sq/pred {_c(v['sq_diff_over_pred_bootstrap_ci'], '.2f')}"
            for ep, v in d["by_selected_epoch"].items()))
        ct = d.get("by_selected_epoch_contrast")
        if ct:
            print(f"  {'':9s} epoch {ct['epochs'][0]} minus epoch {ct['epochs'][1]}: optimism {_c(ct['optimism'], '+.2f', 100)} "
                  f"pp ({_v(ct['optimism'], 0)}), centred {_c(ct['optimism_centred'], '+.2f', 100)} pp "
                  f"({_v(ct['optimism_centred'], 0)}), sq/pred {_c(ct['sq_diff_over_pred_bootstrap'], '+.2f')}")

    print("\n(e) held-out estimation risk relative to raw F: deviation | noise-corrected deviation | level [95% CI]")
    for k in rows:
        e = S[k]["e_estimation_risk"]
        o, ko = e["oracle_common_lambda"], e["oracle_noise_multiplier"]
        print(f"  {LAB[k]} (test-noise share of raw deviation risk {_r(e['test_noise_share_of_raw_dev_risk'])['value']:.2f})")
        for n in EST:
            if n != "raw":
                print(f"    {EST[n]:32s} {_c(e[n]['dev_over_raw'])} | {_c(e[n]['dev_noise_corrected_over_raw'])} | "
                      f"{_c(e[n]['level_over_raw'])}")
        print(f"    {'revised / original (deviation)':32s} {_c(e['paired_vs_binomial']['dev_ratio'])} | "
              f"{_c(e['paired_vs_binomial']['dev_noise_corrected_ratio'])}")
        print(f"    oracle (test labels, analysis only): common lambda {_c(o, '.2f')}, its noise-corrected risk/raw "
              f"{_r(o['dev_noise_corrected_over_raw'])['value']:.2f}; estimators' lambdas with the same ||d_val||^2 "
              f"weighting: " + ", ".join(f"{EST[n].split(' (')[0]} {_c(o['estimator_lambda_ss_weighted'][n], '.2f')}"
                                         for n in ("js_binomial", "js_boot_paired", "bock", "eb")))
        print(f"    oracle noise multiplier kappa* {_c(ko, '.2f')} (kappa = 1: paired bootstrap; binomial-equivalent "
              f"kappa {_f(ko['kappa_binomial_equivalent'], '.2f')}); deviation risk/raw at kappa "
              + ", ".join(f"{kk}: {_r(v)['value']:.3f}" for kk, v in ko["dev_over_raw_at_kappa"].items()))

    print("\n(f) L1 distance of normalised log-odds weights from uniform, mean [max]")
    for k in rows:
        f = S[k]["f_weights"]
        print(f"  {LAB[k]:9s} " + "  ".join(f"{EST[n].split(' (')[0]}: {_m(v):.4f} [{v['max']:.4f}]"
                                            for n, v in f["l1_from_uniform"].items() if n != "grand_mean")
              + f"  | binomial vs paired: {_f(f['l1_binomial_vs_paired'], '.4f')}")

    print("\n(g) validation macro-F1 vs accuracy; error correlation and effective voters (participation ratio)")
    for k in rows:
        g, v = S[k]["g_f1_vs_accuracy"], S[k]["g_effective_voters"]
        print(f"  {LAB[k]:9s} |F-A| mean {_m(g['abs_diff']):.4f} max {g['abs_diff']['max']:.4f}, r {g['pearson_r']:.4f} "
              f"(within unit {g['pearson_r_within_unit']:.4f}), max |w_F - w_A| raw {g['w_maxabs_diff']['raw']['max']:.4f} "
              f"shrunk {g['w_maxabs_diff']['paired_shrunk']['max']:.4f} | rho val {_f(v['val']['rho_mean'])} test "
              f"{_f(v['test']['rho_mean'])}, M_eff val {_f(v['val']['m_eff'], '.2f')} test {_f(v['test']['m_eff'], '.2f')}")
    banner(seeds)


def main():
    t0 = time.perf_counter()
    seeds = OrderedDict((ds, L.complete_seeds(ds)) for ds in L.IN_DOMAIN)
    U = [unit(ds, s, f) for ds in L.IN_DOMAIN for s in seeds[ds] for f in range(L.N_FOLDS)]
    S = OrderedDict((ds, summarise([u for u in U if u["ds"] == ds], ds)) for ds in L.IN_DOMAIN)
    S["evprimary"] = summarise([u for u in U if group(u["ds"]) == "evprimary"], "evprimary")
    S["all"] = summarise(U, "all")
    cfg = {"B": B, "salt": SALT, "salt_stratified_check": SALT_STRAT, "n_ci": N_CI, "M": M,
           "models": [L.MLAB[m] for m in L.MO], "seeds_used": seeds,
           "n_units": OrderedDict((ds, S[ds]["n_units"]) for ds in L.IN_DOMAIN),
           "preliminary": any(len(s) < len(L.SEEDS) for s in seeds.values()), "n_seeds_expected": len(L.SEEDS),
           "estimators": EST, "noise_models": NOISE,
           "summaries": {"per corpus": L.IN_DOMAIN,
                         "evprimary": "EV-Norm and EV-Raw jointly (same posts, folds and labels): one corpus for inference",
                         "all": "value/ci95/ci95_units/mean: ratio of sums or mean over all units (weights corpora by "
                                "noise magnitude and seed count; the primary corpus enters twice); equal_corpus_weights "
                                "and mean_equal_corpus_weights: mean over the corpora evprimary and evpublic of the "
                                "statistic computed within each (printed as pooled*)"},
           "bootstrap": "per unit and split, B i.i.d. resamples of instances shared by all learners (paired), "
                        "from rev_lib.fold_rng(seed, ds, fold, salt=5): validation indices first, then test",
           "ci": "ci95: percentile, two-stage bootstrap over units (folds with replacement within corpus, EV-Norm and "
                 "EV-Raw sharing folds, then seeds with replacement within each drawn fold, one seed draw for both "
                 "renderings when they have the same seeds); ci95_units: naive i.i.d. units",
           "ci_seed_draw_shared_across_renderings": seeds["evprimary-formal"] == seeds["evprimary-informal"],
           "spread_to_noise": {"paper_definition": "mean over units of the range of validation macro-F1 / mean over "
                                                   "units of sqrt(mean_m F_m(1-F_m)/n_val), as in the paper",
                               "range_over_se_of_differences": "mean range / mean sqrt(tr Sigma_d(paired) / (M-1)), the "
                                                               "independence-equivalent SE of learner differences",
                               "binomial, paired": "the same two ratios averaged per unit"},
           "oracle": "oracle_common_lambda and oracle_noise_multiplier are chosen with test labels and are "
                     "analysis-only quantities; estimator_lambda_ss_weighted weights each unit's lambda by "
                     "||d_val||^2 as the oracle implicitly does",
           "kappa": "lambda_u(kappa) = max(0, 1 - (M-3)/(M-1) * kappa * tr Sigma_d(val, paired) / ||d_val||^2); grid "
                    f"{KAPPA[0]:g}..{KAPPA[-1]:g} step {KAPPA[1] - KAPPA[0]:.2f}; kappa* minimises held-out deviation risk "
                    "(pooled*: the mean over corpora of risk/raw)"}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(_clean({"config": cfg, "summary": S, "units": U}), fh, indent=1)
    fig_lambda(U, OUT_FIG)
    report(S, seeds)
    print(f"\nwrote {OUT_JSON} and {OUT_FIG} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
