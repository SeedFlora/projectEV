"""Multi-seed main comparison for the revision: every fusion rule on every fine-tuning seed.

Answers the requests for multiple fine-tuning seeds and for learned-weighting and recent ensemble
baselines. For each corpus and each complete seed, L.pooled_run fits every rule of L.ALL_RULES per
fold on that fold's validation split only and scores the fold's test portion; the five single
learners are scored alongside.

Design decisions
* Seeds. Every corpus uses the seeds of --seeds (default: all five) whose runs are complete, read
  once at start-up (L.complete_seeds: posteriors and training histories present). Missing seeds,
  seed sets that differ between corpora and rules missing in some seed are printed as warnings and
  stored in meta; --strict aborts instead and is the setting for the paper run.
* Aggregation over seeds. A seed's score is its CV mean (mean of the five fold values). "mean"
  averages the CV means over seeds, "sd_seed" is their SD (ddof=1, null with one seed), "sd_fold"
  the mean over seeds of the fold SDs and "sd_all" the SD over all seed x fold values. Differences
  to uniform soft voting are taken within each seed, then averaged. "pooled_f1" is the macro-F1 of
  the pooled out-of-fold predictions averaged over seeds; the bootstrap works on that quantity.
* Primary analysis (pre-specified): SA-WSV (bootstrap) minus uniform soft voting, one two-sided
  two-stage bootstrap test per corpus, Holm over the three corpora. The same test of SA-WSV w/o
  confidence is the secondary analysis (its own Holm family of three corpora).
* Significance of all names. One two-stage bootstrap per corpus for all names with common random
  numbers: every replicate resamples seeds with replacement and test instances within folds,
  jointly for all names, so a difference of replicates is a replicate of the paired difference.
  Two-sided percentile p-values are Holm-corrected within corpus and reference in two families:
  every other name, learners included ("p_holm", the specified family), and the other rules only
  ("p_holm_rules"). Equivalence is TOST at 5% per side: the 90% CI lies inside (-delta, +delta).
* What the bootstrap conditions on. It does not resample the CV partition or the validation split,
  which all seeds share, so every rule's weights see one validation sample per fold (rev_robustness.py
  refits the rules on validation subsamples). Resampling S seeds with replacement also understates
  the between-seed variance by (S-1)/S; the primary and secondary analyses therefore add the
  bootstrap with the seed stage inflated by S/(S-1) and a t-interval over the S per-seed differences
  (seed variability only, conditional on the test set).
* Band. The submitted claim (every soft-voting rule within 0.8 points of uniform soft voting, "a band
  of 0.8 points") concerns one fine-tuning run, so |rule - uniform| and the range including uniform
  are taken on each seed's CV means and reported for the worst seed, for three sets: the submission's
  ten soft-voting rules (uniform included), genuine soft votes (no stacking, no product rule) and all
  soft rules (every rule except best single and majority voting). The seed-averaged values are kept
  and labelled as such; averaging over seeds shrinks the run-to-run differences.
* Exact McNemar per seed on the pooled out-of-fold predictions, Holm within corpus and seed over all
  other names (learners and rules in one family); the monolingual-versus-multilingual tests
  replicate mono_vs_multi.py per seed.
* Diversity. m_eff is the participation ratio (tr R)^2 / tr(R^2) of the learners' correctness
  correlations, an effective dimensionality. The number of independent voters whose average has the
  same variance, M / (1 + (M-1) rho_mean) under equicorrelation, is n_eff_variance (smaller).
* Multi-seed pools (at least two seeds): "seed-averaged" averages each learner's posteriors over
  seeds (5 members), "all-runs" keeps every seed x learner run as a member (5S members). Rules are
  refitted on the pool's validation posteriors with L.fold_rng(0, ds, fold, salt=99); the
  late-epoch rule is excluded because it needs per-run training histories. Pool results are
  descriptive: their CIs resample test instances only (against the single-seed ensembles also the
  single-seed side's seeds) and are conditional on the runs pooled, so no p-values or equivalence
  flags are reported; pool effects change with the seeds pooled.
* Units: every metric is a fraction in [0, 1]; every difference is name minus reference.
* Nothing is fitted on test labels; all randomness is seeded, so two runs give identical JSON.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one BLAS thread per worker: bounded CPU use, reproducible sums
import sys, math, json, time, argparse, itertools
from collections import OrderedDict
from multiprocessing import Pool
import numpy as np
from scipy.stats import t as student_t
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_lib as L

N_BOOT = 5000
METRICS = ["acc", "prec", "rec", "f1"]
DELTAS = (0.005, 0.010)
BAND_CLAIM = 0.008                      # the submission: soft-voting rules "within 0.8 points" of uniform
BINOM = "SA-WSV (binomial)"
SECONDARY = "SA-WSV w/o confidence"     # secondary analysis; L.PRIMARY is the primary one
LEARNERS = [L.MLAB[m] for m in L.MO]
MONO, MULTI = ["IndoBERT-p1", "IndoBERT-1.5G", "IndoRoBERTa-1.5G"], ["XLM-R", "mBERT"]
NOT_SOFT = ("Best single (val-selected)", "Majority voting")
NOT_GENUINE = ("Stacking (LR)", "Product rule")      # soft rules that are not a weighted average of posteriors
POOL_RULES = ["Best single (val-selected)", "Majority voting", L.BASELINE, "Linear-F1 weighting", "Log-odds x entropy",
              BINOM, L.PRIMARY, "SA-WSV (EB)", "Stacking (LR)", "Convex NLL weights", "Ensemble selection",
              "Uniform + TS"]
POOL_LAM = ["sawsv_binom", "sawsv_boot", "sawsv_eb"]
LAM_KEYS = ["sawsv_binom", "sawsv_bootm", "sawsv_boot", "sawsv_bock", "sawsv_eb"]
POOL_BASIS = ("descriptive: test instances resampled within folds (boot_vs_single_seed also resamples the single-seed "
              "side's seeds); conditional on the fine-tuning runs pooled, hence no p-values or equivalence flags")
OUT = os.path.join(L.ROOT, "experiments", "results", "rev", "rev_main.json")


def sd(x):
    x = np.asarray(x, float).ravel()
    return float(x.std(ddof=1)) if x.size > 1 else None


def ratio(a, b):
    return float(a / b) if b > 0 else None


def equiv(ci90):
    return {f"equiv_{d:.3f}": bool(-d < ci90[0] and ci90[1] < d) for d in DELTAS}


def mmm(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def boot_summary(d):
    """percentile CIs and two-sided p-value of bootstrap replicates d, as in L.compare_from_reps"""
    return OrderedDict(ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                       ci90=[float(np.percentile(d, 5)), float(np.percentile(d, 95))],
                       p_boot=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def descriptive(c):
    """a bootstrap comparison without its p-value (multi-seed pools)"""
    return OrderedDict(diff=c["diff"], ci95=c["ci95"], ci90=c["ci90"])


def n_eff_variance(ev):
    """independent voters whose average has the variance of the average of M voters with correlation rho_mean"""
    return float(ev["M"] / (1 + (ev["M"] - 1) * ev["rho_mean"]))


# ------------------------------------------------------------------------- (a) scores
def seed_metrics(runs, seeds, names, y):
    out = OrderedDict()
    for k in names:
        e = OrderedDict()
        for met in METRICS:
            V = np.array([[r[met] for r in runs[s]["per_fold"][k]] for s in seeds])            # (S, folds)
            U = np.array([[r[met] for r in runs[s]["per_fold"][L.BASELINE]] for s in seeds])
            cv, fsd = V.mean(1), V.std(1, ddof=1)
            d = cv - U.mean(1)
            e[met] = OrderedDict(
                mean=float(cv.mean()), sd_seed=sd(cv), sd_fold=float(fsd.mean()), sd_all=sd(V),
                diff_vs_uniform=float(d.mean()), n_seeds_beat_uniform=int((d > 0).sum()),
                n_seeds_tie_uniform=int((d == 0).sum()), worst_seed_diff_vs_uniform=float(d.min()),
                best_seed_diff_vs_uniform=float(d.max()), worst_fold_diff_vs_uniform=float((V - U).min()),
                per_seed=OrderedDict((str(s), {"cv_mean": float(cv[i]), "fold_sd": float(fsd[i]), "folds": V[i].tolist(),
                                               "diff_vs_uniform": float(d[i])}) for i, s in enumerate(seeds)))
        pf = [L.macro_f1(y, runs[s]["pred"][k]) for s in seeds]
        e["pooled_f1"], e["pooled_f1_per_seed"] = float(np.mean(pf)), pf
        out[k] = e
    return out


# ------------------------------------------------------------------- (b) bootstrap
def significance(y, fold, runs, seeds, names):
    P = OrderedDict((k, np.stack([runs[s]["pred"][k] for s in seeds])) for k in names)
    point, reps = L.two_stage_bootstrap_all(y, P, strata=fold, n_boot=N_BOOT, seed=0)
    vs, fam = OrderedDict(), OrderedDict()
    for ref in (L.PRIMARY, L.BASELINE):
        cmp = OrderedDict((k, L.compare_from_reps(point, reps, k, ref)) for k in names if k != ref)
        ph = L.holm({k: v["p_boot"] for k, v in cmp.items()})                            # all other names
        pr = L.holm({k: v["p_boot"] for k, v in cmp.items() if k not in LEARNERS})       # other rules only
        for k, v in cmp.items():
            v["p_holm"], v["p_holm_rules"] = ph[k], pr.get(k); v.update(equiv(v["ci90"]))
        vs[ref], fam[ref] = cmp, OrderedDict(p_holm=len(ph), p_holm_rules=len(pr))
    return {"point_pooled_f1": point, "vs": vs, "holm_family_size": fam}, point, reps


# ------------------------------------------------------ primary and secondary analyses
def confirmatory(y, fold, runs, seeds, point, reps, met):
    """per-corpus part of the primary (L.PRIMARY) and secondary (SECONDARY) tests against uniform soft
    voting; Holm over the corpora follows in main. The instance and seed draws repeat those of
    L.two_stage_bootstrap_all(seed=0), so the plain replicates are significance()'s; the seed-inflated
    version scales the resampled seeds' deviations from the all-seed mean by sqrt(S/(S-1))."""
    S, ref = len(seeds), L.BASELINE
    names = [L.PRIMARY, SECONDARY, ref]
    rng = np.random.default_rng(0)
    idx = L.stratified_boot_index(fold, N_BOOT, rng)
    sidx, cols = rng.integers(0, S, size=(N_BOOT, S)), np.arange(N_BOOT)[:, None]
    F = L.macro_f1_batch(y, np.concatenate([np.stack([runs[s]["pred"][k] for s in seeds]) for k in names]), idx)
    F = F.reshape(len(names), S, N_BOOT)
    out = OrderedDict()
    for key, k in (("primary", L.PRIMARY), ("secondary", SECONDARY)):
        D = F[names.index(k)] - F[-1]                          # (S, B): per-seed difference on each resample
        rep = D[sidx, cols].mean(1)
        assert np.allclose(rep, reps[k] - reps[ref], rtol=0, atol=1e-12), (key, "bootstrap stream differs")
        d = np.array(met[k]["pooled_f1_per_seed"]) - np.array(met[ref]["pooled_f1_per_seed"])
        e = OrderedDict(name=k, reference=ref, n_seeds=S, **L.compare_from_reps(point, reps, k, ref))
        e.update(n_seeds_positive=int((d > 0).sum()), per_seed_diff=OrderedDict((str(s), float(v)) for s, v in zip(seeds, d)),
                 diff_cv_mean=met[k]["f1"]["diff_vs_uniform"], n_seeds_beat_uniform_cv=met[k]["f1"]["n_seeds_beat_uniform"],
                 seed_inflated=None, seed_t_interval=None)
        if S >= 2:
            e["seed_inflated"] = boot_summary(D.mean(0) + math.sqrt(S / (S - 1)) * (rep - D.mean(0)))
            h = float(student_t.ppf(0.975, S - 1) * d.std(ddof=1) / math.sqrt(S))
            e["seed_t_interval"] = OrderedDict(mean=float(d.mean()), sd=float(d.std(ddof=1)),
                                               ci95=[float(d.mean() - h), float(d.mean() + h)])
        out[key] = e
    return out


def over_corpora(conf, key, status, name):
    """Holm over the corpora for one of the confirmatory comparisons"""
    per = OrderedDict((ds, c[key]) for ds, c in conf.items())
    ph = L.holm({ds: e["p_boot"] for ds, e in per.items()})
    pi = L.holm({ds: e["seed_inflated"]["p_boot"] for ds, e in per.items() if e["seed_inflated"]})
    for ds, e in per.items():
        e["p_holm_corpora"] = ph[ds]
        if e["seed_inflated"]:
            e["seed_inflated"]["p_holm_corpora"] = pi[ds]
    return OrderedDict(status=status, comparison=f"{name} - {L.BASELINE}",
                       quantity="pooled out-of-fold macro-F1 averaged over seeds (as significance.point_pooled_f1)",
                       test="two-sided two-stage bootstrap p (seeds and test instances within folds), Holm over corpora",
                       n_corpora=len(per), per_corpus=per)


# --------------------------------------------------------------------- (c) McNemar
def mcnemar_tests(y, runs, seeds, names):
    """b = instances the name gets right and the reference wrong, c = the reverse"""
    out = OrderedDict()
    for ref in (L.PRIMARY, BINOM):
        per = OrderedDict((k, OrderedDict()) for k in names if k != ref)
        for s in seeds:
            pr = runs[s]["pred"]
            res = {k: L.mcnemar(y, pr[k], pr[ref]) for k in per}
            ph = L.holm({k: v[2] for k, v in res.items()})
            for k, (b, c, p) in res.items():
                per[k][str(s)] = {"b": b, "c": c, "p": p, "p_holm": ph[k]}
        out[ref] = OrderedDict((k, {"n_seeds_sig": int(sum(v["p_holm"] < 0.05 for v in ps.values())),
                                    "median_p_holm": float(np.median([v["p_holm"] for v in ps.values()])),
                                    "per_seed": ps}) for k, ps in per.items())
    return out


# ------------------------------------------------------- (d) mono- vs multilingual
def mono_vs_multi(y, runs, seeds, point, reps):
    pairs = [(a, b, f"{a} vs {b}") for a, b in itertools.product(MONO, MULTI)]
    per = OrderedDict()
    for s in seeds:
        pr, res = runs[s]["pred"], OrderedDict()
        for a, b, key in pairs:
            nb, nc, p = L.mcnemar(y, pr[a], pr[b])
            res[key] = {"d_f1": L.macro_f1(y, pr[a]) - L.macro_f1(y, pr[b]), "b": nb, "c": nc, "p": p}
        for key, v in L.holm({key: v["p"] for key, v in res.items()}).items():
            res[key]["p_holm"] = v
        per[str(s)] = res
    boot = OrderedDict((key, L.compare_from_reps(point, reps, a, b)) for a, b, key in pairs)
    bh = L.holm({key: v["p_boot"] for key, v in boot.items()})
    out = OrderedDict()
    for _, _, key in pairs:
        d = np.array([per[str(s)][key]["d_f1"] for s in seeds])
        out[key] = OrderedDict(d_f1_mean=float(d.mean()), d_f1_min=float(d.min()), d_f1_max=float(d.max()),
                               n_seeds_sig=int(sum(per[str(s)][key]["p_holm"] < 0.05 for s in seeds)),
                               n_seeds_positive=int((d > 0).sum()), boot=dict(boot[key], p_holm=bh[key]),
                               per_seed=OrderedDict((str(s), per[str(s)][key]) for s in seeds))
    return out


# ----------------------------------------------------------------- (e) diversity
def diversity(y, fold, runs, seeds):
    OK = {s: np.stack([runs[s]["pred"][k] == y for k in LEARNERS]) for s in seeds}
    per = OrderedDict()
    for s in seeds:
        ok = OK[s]
        q, dis = L.q_statistic(ok); ev = L.effective_voters(ok)
        qf = np.array([L.q_statistic(ok[:, fold == f]) for f in range(L.N_FOLDS)])     # submitted paper: per fold
        prof = np.bincount(ok.sum(0), minlength=len(LEARNERS) + 1) / len(y)
        per[str(s)] = OrderedDict(Q=q, disagreement=dis, Q_foldmean=float(qf[:, 0].mean()),
                                  disagreement_foldmean=float(qf[:, 1].mean()), rho_mean=ev["rho_mean"],
                                  m_eff=ev["m_eff"], n_eff_variance=n_eff_variance(ev),
                                  contested=float(1 - prof[0] - prof[-1]), profile=prof.tolist())
    mean = OrderedDict((k, float(np.mean([per[str(s)][k] for s in seeds]))) for k in per[str(seeds[0])] if k != "profile")
    mean["profile"] = np.mean([per[str(s)]["profile"] for s in seeds], 0).tolist()
    out = OrderedDict(mean_over_seeds=mean, per_seed=per, seed_diversity=None)
    if len(seeds) >= 2:
        same = OrderedDict((lab, L.q_statistic(np.stack([OK[s][i] for s in seeds]))) for i, lab in enumerate(LEARNERS))
        cross = np.array([L.q_statistic(np.stack([OK[s][i], OK[t][j]])) for s, t in itertools.combinations(seeds, 2)
                          for i in range(len(LEARNERS)) for j in range(len(LEARNERS)) if i != j])
        out["seed_diversity"] = OrderedDict(
            Q_same_learner_across_seeds=float(np.mean([v[0] for v in same.values()])),
            disagreement_same_learner_across_seeds=float(np.mean([v[1] for v in same.values()])),
            Q_between_learners_within_seed=mean["Q"], disagreement_between_learners_within_seed=mean["disagreement"],
            Q_between_learners_across_seeds=float(cross[:, 0].mean()),
            disagreement_between_learners_across_seeds=float(cross[:, 1].mean()),
            per_learner=OrderedDict((lab, {"Q": v[0], "disagreement": v[1]}) for lab, v in same.items()))
    return out


# ------------------------------------------------------------- (f) multi-seed pools
def pools(ds, y, fold, runs, seeds, met):
    if len(seeds) < 2:
        return None
    rules = OrderedDict((k, L.ALL_RULES[k]) for k in POOL_RULES)
    out = OrderedDict()
    for kind, combine in (("seed-averaged", np.mean), ("all-runs", np.concatenate)):
        f1, pred, lam = ({k: [] for k in POOL_RULES}, {k: [] for k in POOL_RULES}, {k: [] for k in POOL_LAM})
        ok, ys, memb = [], [], []
        for f in range(L.N_FOLDS):
            D = [L.load_fold(ds, f, s) for s in seeds]
            for d in D[1:]:
                for key in ("yv", "yt", "val_idx", "test_idx"):
                    assert np.array_equal(d[key], D[0][key]), (ds, f, key)
            Pv, Pt, yv, yt = combine([d["Pv"] for d in D], 0), combine([d["Pt"] for d in D], 0), D[0]["yv"], D[0]["yt"]
            fitted, ctx = L.fit_rules(Pv, yv, L.fold_rng(0, ds, f, salt=99), rules=rules)
            assert list(fitted) == POOL_RULES, list(fitted)
            for k, fn in fitted.items():
                p = fn(Pt)[0]; f1[k].append(L.macro_f1(yt, p)); pred[k].append(p)
            for k in POOL_LAM:
                lam[k].append(ctx.meta[k]["lam"])
            ok.append(Pt.argmax(-1) == yt[None]); ys.append(yt)
            memb.append(L.macro_f1_batch(yt, Pt.argmax(-1)))
        assert np.array_equal(np.concatenate(ys), y)
        ok = np.concatenate(ok, 1)
        q, dis = L.q_statistic(ok); ev = L.effective_voters(ok)
        pp = OrderedDict((k, np.concatenate(v)) for k, v in pred.items())
        point, reps = L.two_stage_bootstrap_all(y, {k: v[None] for k, v in pp.items()}, strata=fold, n_boot=N_BOOT, seed=0)
        cmp = OrderedDict((k, L.compare_from_reps(point, reps, k, L.BASELINE)) for k in POOL_RULES if k != L.BASELINE)
        # pool versus the single-seed five-learner ensemble of the same rule
        P2 = OrderedDict()
        for k in (L.BASELINE, L.PRIMARY):
            P2["pool|" + k] = np.repeat(pp[k][None], len(seeds), 0)
            P2["single|" + k] = np.stack([runs[s]["pred"][k] for s in seeds])
        pt2, rp2 = L.two_stage_bootstrap_all(y, P2, strata=fold, n_boot=N_BOOT, seed=0)
        U, rows = np.array(f1[L.BASELINE]), OrderedDict()
        for k in POOL_RULES:
            v = np.array(f1[k])
            rows[k] = OrderedDict(f1_mean=float(v.mean()), f1_sd_fold=sd(v), f1_folds=v.tolist(),
                                  diff_vs_pool_uniform=float((v - U).mean()), pooled_f1=point[k],
                                  single_seed_f1_mean=met[k]["f1"]["mean"],
                                  gain_vs_single_seed=float(v.mean() - met[k]["f1"]["mean"]))
            if k in cmp:                                       # descriptive (POOL_BASIS): no p-value, no TOST flags
                rows[k]["boot_vs_pool_uniform"] = descriptive(cmp[k])
            if "pool|" + k in P2:
                rows[k]["boot_vs_single_seed"] = descriptive(L.compare_from_reps(pt2, rp2, "pool|" + k, "single|" + k))
        mf = np.array(memb).mean(0)
        if kind == "seed-averaged":
            members = OrderedDict((lab, {"f1_mean": float(mf[i]), "single_seed_f1_mean": met[lab]["f1"]["mean"]})
                                  for i, lab in enumerate(LEARNERS))
        else:
            members = {"f1_mean_of_members": float(mf.mean()), "oracle_f1_best_member_by_test": float(mf.max())}
        out[kind] = OrderedDict(n_members=int(ok.shape[0]), seeds=list(seeds), ci_basis=POOL_BASIS, Q=q, disagreement=dis,
                                rho_mean=ev["rho_mean"], m_eff=ev["m_eff"], n_eff_variance=n_eff_variance(ev),
                                lambdas=OrderedDict((k, dict(mmm(v), folds=v)) for k, v in lam.items()),
                                single_seed_uniform_f1_mean=met[L.BASELINE]["f1"]["mean"], members=members, rules=rows)
    return out


# ------------------------------------------------------------------ (g) shrinkage
def shrinkage(ds, runs, seeds):
    est = OrderedDict()
    for key in LAM_KEYS:
        lam = np.array([[m[key]["lam"] for m in runs[s]["meta"]] for s in seeds])
        est[key] = OrderedDict(mean=float(lam.mean()), sd=sd(lam), min=float(lam.min()), max=float(lam.max()),
                               frac_zero=float((lam == 0).mean()), frac_ge_099=float((lam >= 0.99).mean()),
                               per_seed=OrderedDict((str(s), lam[i].tolist()) for i, s in enumerate(seeds)))
    rows = []
    for s in seeds:
        for f in range(L.N_FOLDS):
            d, meta = L.load_fold(ds, f, s), runs[s]["meta"][f]
            F = L.macro_f1_batch(d["yv"], d["Pv"].argmax(-1)); n, M = len(d["yv"]), len(F)
            var = float(np.mean(F * (1 - F)) / n)
            mb, mm, mp = meta["sawsv_binom"], meta["sawsv_bootm"], meta["sawsv_boot"]
            assert abs(mb["ss"] - float(((F - F.mean()) ** 2).sum())) < 1e-12 and abs(mb["noise"] - (M - 1) * var) < 1e-12
            rows.append(OrderedDict(
                seed=s, fold=f, n_val=n, spread=float(F.max() - F.min()), se_binom=math.sqrt(var),
                se_boot_marginal=math.sqrt(mm["noise"] / (M - 1)), ratio_paper=ratio(F.max() - F.min(), math.sqrt(var)),
                snr_binom=ratio(math.sqrt(mb["ss"]), math.sqrt(mb["noise"])),
                snr_boot=ratio(math.sqrt(mp["ss"]), math.sqrt(mp["noise"])),
                noise_bootm_over_binom=ratio(mm["noise"], mb["noise"]), noise_boot_over_binom=ratio(mp["noise"], mb["noise"]),
                lam_binom=mb["lam"], lam_boot=mp["lam"]))
    summ = OrderedDict((k, mmm([r[k] for r in rows if r[k] is not None]))
                       for k in ("spread", "se_binom", "se_boot_marginal", "ratio_paper", "snr_binom", "snr_boot",
                                 "noise_bootm_over_binom", "noise_boot_over_binom"))
    summ["ratio_paper_of_means"] = summ["spread"]["mean"] / summ["se_binom"]["mean"]     # as quoted in the paper
    return OrderedDict(estimators=est, spread_to_noise=summ, per_fold=rows)


# -------------------------------------------------------------------- (h) band
def band(met, sig, names, seeds):
    """|rule - uniform| and range (uniform included) of CV-mean macro-F1 per rule set. per_seed and
    worst_seed_* are single-run values; max_abs_diff_vs_uniform, argmax, signed_diff and
    range_incl_uniform are seed-averaged (CV means averaged over seeds before the max or range)."""
    rules = [k for k in names if k not in LEARNERS]
    soft = [k for k in rules if k not in NOT_SOFT and k != L.BASELINE]
    vs = sig["vs"][L.BASELINE]

    def one(rs):
        per = OrderedDict()
        for s in map(str, seeds):
            d = OrderedDict((k, met[k]["f1"]["per_seed"][s]["diff_vs_uniform"]) for k in rs)
            cv = OrderedDict((k, met[k]["f1"]["per_seed"][s]["cv_mean"]) for k in rs + [L.BASELINE])
            arg, top, bot = max(rs, key=lambda k: abs(d[k])), max(cv, key=cv.get), min(cv, key=cv.get)
            per[s] = OrderedDict(max_abs_diff_vs_uniform=abs(d[arg]), argmax=arg, signed_diff=d[arg],
                                 range_incl_uniform=cv[top] - cv[bot], top=top, bottom=bot)
        wd = max(per, key=lambda s: per[s]["max_abs_diff_vs_uniform"])
        wr = max(per, key=lambda s: per[s]["range_incl_uniform"])
        arg = max(rs, key=lambda k: abs(met[k]["f1"]["diff_vs_uniform"]))
        means = [met[k]["f1"]["mean"] for k in rs + [L.BASELINE]]
        return OrderedDict(
            rules=rs, n_rules_incl_uniform=len(rs) + 1,
            worst_seed_max_abs_diff_vs_uniform=per[wd]["max_abs_diff_vs_uniform"], worst_seed_max_abs_diff_seed=int(wd),
            worst_seed_max_abs_diff_rule=per[wd]["argmax"], worst_seed_signed_diff=per[wd]["signed_diff"],
            worst_seed_range_incl_uniform=per[wr]["range_incl_uniform"], worst_seed_range_seed=int(wr),
            worst_seed_range_top=per[wr]["top"], worst_seed_range_bottom=per[wr]["bottom"],
            n_seeds_max_abs_diff_within_claim=int(sum(v["max_abs_diff_vs_uniform"] <= BAND_CLAIM for v in per.values())),
            n_seeds_range_within_claim=int(sum(v["range_incl_uniform"] <= BAND_CLAIM for v in per.values())),
            per_seed=per,
            per_seed_max_abs_diff=OrderedDict((s, v["max_abs_diff_vs_uniform"]) for s, v in per.items()),
            mean_over_seeds_of_per_seed_max=float(np.mean([v["max_abs_diff_vs_uniform"] for v in per.values()])),
            seed_averaged_note="max_abs_diff_vs_uniform, argmax, signed_diff and range_incl_uniform use CV means "
                               "averaged over seeds; not a single-run statement",
            max_abs_diff_vs_uniform=abs(met[arg]["f1"]["diff_vs_uniform"]), argmax=arg,
            signed_diff=met[arg]["f1"]["diff_vs_uniform"], range_incl_uniform=float(max(means) - min(means)),
            **{f"equivalent_{d:.3f}": [k for k in rs if vs[k][f"equiv_{d:.3f}"]] for d in DELTAS})
    out = OrderedDict(original_soft_rules=one([k for k in soft if k in L.ORIGINAL_RULES]),
                      genuine_soft_votes=one([k for k in soft if k not in NOT_GENUINE]), all_soft_rules=one(soft))
    out.update(significant_vs_uniform_holm=[k for k in rules if k in vs and vs[k]["p_holm"] < 0.05],
               significant_vs_uniform_holm_rules=[k for k in rules if k in vs and vs[k]["p_holm_rules"] < 0.05],
               significant_vs_uniform_unadjusted=[k for k in rules if k in vs and vs[k]["p_boot"] < 0.05],
               ranking_f1=sorted(rules, key=lambda k: -met[k]["f1"]["mean"]))
    return out


def across_corpora(per_corpus):
    """risk view of every rule: its seed-averaged difference to uniform on each corpus"""
    first = next(iter(per_corpus.values()))
    out = OrderedDict()
    for k in [k for k in first["names"] if k not in LEARNERS and k != L.BASELINE]:
        rs = [r for r in per_corpus.values() if k in r["metrics"]]
        d = np.array([r["metrics"][k]["f1"]["diff_vs_uniform"] for r in rs])
        vs = [r["significance"]["vs"][L.BASELINE][k] for r in rs]
        out[k] = OrderedDict(mean_diff_vs_uniform=float(d.mean()), worst_corpus_diff=float(d.min()),
                             best_corpus_diff=float(d.max()),
                             worst_seed_diff=float(min(r["metrics"][k]["f1"]["worst_seed_diff_vs_uniform"] for r in rs)),
                             n_corpora_beat_uniform=int((d > 0).sum()), n_corpora=len(rs),
                             n_corpora_sig_holm=int(sum(v["p_holm"] < 0.05 for v in vs)),
                             n_corpora_sig_holm_rules=int(sum(v["p_holm_rules"] < 0.05 for v in vs)))
    return out


# --------------------------------------------------------------------- driver
def analyse(job):
    ds, seeds = job
    t0 = time.perf_counter()
    runs = OrderedDict((s, L.pooled_run(ds, s, rules=L.ALL_RULES)) for s in seeds)
    y, fold = runs[seeds[0]]["y"], runs[seeds[0]]["fold"]
    for s in seeds[1:]:
        assert np.array_equal(runs[s]["y"], y) and np.array_equal(runs[s]["fold"], fold), (ds, s)
    names = [k for k in runs[seeds[0]]["pred"] if all(k in runs[s]["pred"] for s in seeds)]
    dropped = sorted({k for s in seeds for k in runs[s]["pred"]} - set(names))
    met = seed_metrics(runs, seeds, names, y)
    sig, point, reps = significance(y, fold, runs, seeds, names)
    for k in names:
        assert abs(point[k] - met[k]["pooled_f1"]) < 1e-12
    res = OrderedDict(label=L.DS_LAB[ds], seeds=list(seeds), n=int(len(y)), names=names, dropped_names=dropped,
                      metrics=met, significance=sig, mcnemar=mcnemar_tests(y, runs, seeds, names),
                      mono_vs_multi=mono_vs_multi(y, runs, seeds, point, reps), diversity=diversity(y, fold, runs, seeds),
                      pools=pools(ds, y, fold, runs, seeds, met), shrinkage=shrinkage(ds, runs, seeds))
    res["band"] = band(met, sig, names, seeds)
    return ds, res, confirmatory(y, fold, runs, seeds, point, reps, met), time.perf_counter() - t0


def clean(o):
    if isinstance(o, dict):
        return OrderedDict((str(k), clean(v)) for k, v in o.items())
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def report(ds, r):
    pt = lambda v: f"{100 * v:+6.2f}"
    S, met, vs = r["seeds"], r["metrics"], r["significance"]["vs"][L.BASELINE]
    fam = r["significance"]["holm_family_size"][L.BASELINE]
    out = [f"\n===== {r['label']} ({ds}): seeds {S}, n = {r['n']}; macro-F1 in points, differences vs uniform =====",
           f"(pHall: Holm over all {fam['p_holm']} other names, learners included; pHrul: over the {fam['p_holm_rules']} "
           f"other rules)",
           f"{'name':30s} {'F1':>6s} {'sd_s':>5s} {'dCV':>6s} {'dPool':>6s} {'95% CI (pool)':>15s} {'pHall':>6s} "
           f"{'pHrul':>6s} eq.5 eq1 beat"]
    for k in r["names"]:
        m = met[k]["f1"]
        sds = "    -" if m["sd_seed"] is None else f"{100 * m['sd_seed']:5.2f}"
        if k == L.BASELINE:
            out.append(f"{k:30s} {100 * m['mean']:6.2f} {sds}  (reference)")
            continue
        b = vs[k]
        phr = "     -" if b["p_holm_rules"] is None else f"{b['p_holm_rules']:6.3f}"
        out.append(f"{k:30s} {100 * m['mean']:6.2f} {sds} {pt(m['diff_vs_uniform'])} {pt(b['diff'])} "
                   f"[{100 * b['ci95'][0]:+5.2f},{100 * b['ci95'][1]:+5.2f}] {b['p_holm']:6.3f} {phr} "
                   f"{'y' if b['equiv_0.005'] else 'n':>4s} {'y' if b['equiv_0.010'] else 'n':>3s} "
                   f"{m['n_seeds_beat_uniform']}/{len(S)}")
    for ref, mc in r["mcnemar"].items():
        sig = lambda learner: [f"{k} ({v['n_seeds_sig']}/{len(S)})" for k, v in mc.items()
                               if v["n_seeds_sig"] and (k in LEARNERS) == learner]
        out.append(f"McNemar vs {ref} (Holm over the {len(mc)} other names, learners included, per seed), p_holm<0.05 "
                   f"in some seed: fusion rules: {', '.join(sig(False)) or 'none'}; "
                   f"single learners: {', '.join(sig(True)) or 'none'}")
    out.append("mono vs multi (pooled McNemar per seed, Holm over 6; bootstrap CI of the seed-averaged difference):")
    for key, v in r["mono_vs_multi"].items():
        b = v["boot"]
        out.append(f"  {key:30s} dF1 {pt(v['d_f1_mean'])} [{100 * v['d_f1_min']:+.2f}..{100 * v['d_f1_max']:+.2f}] "
                   f"sig {v['n_seeds_sig']}/{len(S)}  CI95 [{100 * b['ci95'][0]:+.2f},{100 * b['ci95'][1]:+.2f}] p_holm {b['p_holm']:.4f}")
    dv = r["diversity"]["mean_over_seeds"]
    line = (f"diversity: Q {dv['Q']:.3f} (fold-mean {dv['Q_foldmean']:.3f}), disagreement {dv['disagreement']:.3f}, "
            f"participation ratio m_eff {dv['m_eff']:.2f} (variance-equivalent independent voters "
            f"{dv['n_eff_variance']:.2f}), all-correct {100 * dv['profile'][-1]:.1f}%, none {100 * dv['profile'][0]:.1f}%, "
            f"contested {100 * dv['contested']:.1f}%")
    sdv = r["diversity"]["seed_diversity"]
    if sdv:
        line += (f"; Q same learner across seeds {sdv['Q_same_learner_across_seeds']:.3f} vs between learners "
                 f"{sdv['Q_between_learners_within_seed']:.3f} (across seeds {sdv['Q_between_learners_across_seeds']:.3f})")
    out.append(line)
    sh = r["shrinkage"]
    out.append("lambda: " + ", ".join(f"{k[6:]} {v['mean']:.3f} (zero {100 * v['frac_zero']:.0f}%, >=.99 {100 * v['frac_ge_099']:.0f}%)"
                                      for k, v in sh["estimators"].items()))
    s2n = sh["spread_to_noise"]
    out.append(f"spread {s2n['spread']['mean']:.3f}, SE binomial {s2n['se_binom']['mean']:.4f} vs bootstrap {s2n['se_boot_marginal']['mean']:.4f}; "
               f"spread/SE (paper) {s2n['ratio_paper_of_means']:.2f}; sqrt(ss/noise) binomial {s2n['snr_binom']['mean']:.2f}, "
               f"paired bootstrap {s2n['snr_boot']['mean']:.2f}; noise paired/binomial {s2n['noise_boot_over_binom']['mean']:.2f}")
    for kind, p in (r["pools"] or {}).items():
        rr = p["rules"]
        out.append(f"pool {kind} (M={p['n_members']}, Q {p['Q']:.3f}; descriptive, conditional on the runs pooled): "
                   f"uniform {100 * rr[L.BASELINE]['f1_mean']:.2f} (single-seed {100 * p['single_seed_uniform_f1_mean']:.2f}); "
                   f"lambda boot {p['lambdas']['sawsv_boot']['mean']:.2f}; "
                   + ", ".join(f"{k} {pt(v['diff_vs_pool_uniform'])}" for k, v in rr.items() if k != L.BASELINE))
    bd = r["band"]
    out.append(f"band (CV means, points; submitted claim {100 * BAND_CLAIM:.1f}): 'worst seed' = one run, 'seed-avg' = CV "
               f"means averaged over seeds first")
    for key in ("original_soft_rules", "genuine_soft_votes", "all_soft_rules"):
        v = bd[key]
        out.append(f"  {key:19s} ({v['n_rules_incl_uniform']:2d} incl. uniform): max |rule - uniform| worst seed "
                   f"{100 * v['worst_seed_max_abs_diff_vs_uniform']:.2f} ({v['worst_seed_max_abs_diff_rule']}, seed "
                   f"{v['worst_seed_max_abs_diff_seed']}), seed-avg {100 * v['max_abs_diff_vs_uniform']:.2f}; range worst seed "
                   f"{100 * v['worst_seed_range_incl_uniform']:.2f} (seed {v['worst_seed_range_seed']}), seed-avg "
                   f"{100 * v['range_incl_uniform']:.2f}; seeds within claim: |diff| "
                   f"{v['n_seeds_max_abs_diff_within_claim']}/{len(S)}, range {v['n_seeds_range_within_claim']}/{len(S)}; "
                   f"equivalent 1.0 pt {len(v['equivalent_0.010'])}/{len(v['rules'])}, 0.5 pt "
                   f"{len(v['equivalent_0.005'])}/{len(v['rules'])}")
    out.append(f"vs uniform (two-stage bootstrap): Holm all names: {', '.join(bd['significant_vs_uniform_holm']) or 'none'}; "
               f"Holm rules only: {', '.join(bd['significant_vs_uniform_holm_rules']) or 'none'}; "
               f"unadjusted p<0.05: {', '.join(bd['significant_vs_uniform_unadjusted']) or 'none'}")
    return out


def report_confirmatory(a):
    pt = lambda v: f"{100 * v:+6.2f}"
    ci = lambda c: f"[{100 * c[0]:+5.2f},{100 * c[1]:+5.2f}]"
    out = [f"\n===== {a['status']} analysis: {a['comparison']}, pooled macro-F1 points, Holm over {a['n_corpora']} corpora =====",
           "(seed-inflated: seed-stage variance multiplied by S/(S-1); t over seeds: seed variability only, test set "
           "fixed; pos: seeds with a positive difference)",
           f"{'corpus':10s} {'S':>2s} {'diff':>6s} {'95% CI':>15s} {'p_boot':>7s} {'pHolm':>6s} {'pos':>4s} | "
           f"{'seed-inflated 95% CI':>20s} {'pHolm':>6s} | {'t over seeds 95% CI':>19s}"]
    for ds, e in a["per_corpus"].items():
        si, st = e["seed_inflated"], e["seed_t_interval"]
        line = (f"{L.DS_LAB[ds]:10s} {e['n_seeds']:2d} {pt(e['diff'])} {ci(e['ci95']):>15s} {e['p_boot']:7.4f} "
                f"{e['p_holm_corpora']:6.4f} {str(e['n_seeds_positive']) + '/' + str(e['n_seeds']):>4s}")
        if si:
            line += f" | {ci(si['ci95']):>20s} {si['p_holm_corpora']:6.4f} | {ci(st['ci95']):>19s}"
        out.append(line)
    return out


def seed_warnings(jobs, requested):
    w = [f"{L.DS_LAB[ds]}: seeds {[s for s in requested if s not in ss]} are not complete, using {ss}"
         for ds, ss in jobs if ss != requested]
    if len({tuple(ss) for _, ss in jobs}) > 1:
        w.append("seed sets differ between corpora: across_corpora and the Holm-over-corpora tests mix seed counts")
    return w


def shown(p):
    try:
        return os.path.relpath(p, L.ROOT)
    except ValueError:                  # another drive (Windows)
        return os.path.abspath(p)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=3, help="worker processes, one corpus each (at most 6)")
    ap.add_argument("--seeds", type=int, nargs="+", default=L.SEEDS, help="restrict to these seeds (default: all complete)")
    ap.add_argument("--strict", action="store_true",
                    help="abort unless every corpus has every requested seed and every rule (the paper run)")
    ap.add_argument("--out", default=OUT, help="output JSON (default: experiments/results/rev/rev_main.json)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    req = sorted(set(a.seeds), key=lambda s: (L.SEEDS.index(s) if s in L.SEEDS else len(L.SEEDS), s))
    jobs = [(ds, L.complete_seeds(ds, req)) for ds in L.IN_DOMAIN]          # fixed for the whole run
    warn = seed_warnings(jobs, req)
    for w in warn:
        print("WARNING:", w)
    if a.strict and warn:
        sys.exit("--strict: " + "; ".join(warn))
    assert all(s for _, s in jobs), jobs
    with Pool(max(1, min(a.workers, 6, len(jobs)))) as pool:
        results = pool.map(analyse, jobs)
    dropped = OrderedDict((L.DS_LAB[ds], res["dropped_names"]) for ds, res, _, _ in results if res["dropped_names"])
    if dropped:
        warn.append(f"names missing in some seed and dropped: {dict(dropped)}")
        print("WARNING:", warn[-1])
        if a.strict:
            sys.exit("--strict: " + warn[-1])
    conf = OrderedDict((ds, c) for ds, _, c, _ in results)
    cmd = ("py -3.12 -u experiments/rev_main.py" + ("" if req == L.SEEDS else " --seeds " + " ".join(map(str, req)))
           + (" --strict" if a.strict else ""))
    out = OrderedDict(meta=OrderedDict(
        script="experiments/rev_main.py", run_command=cmd, seeds=OrderedDict((ds, s) for ds, s in jobs),
        seeds_requested=req, strict=a.strict, same_seeds_all_corpora=len({tuple(s) for _, s in jobs}) == 1, warnings=warn,
        n_boot=N_BOOT, boot_seed=0, equivalence_deltas=list(DELTAS), band_claim=BAND_CLAIM, primary=L.PRIMARY,
        secondary=SECONDARY, original=BINOM, baseline=L.BASELINE, learners=LEARNERS, rules=list(L.ALL_RULES),
        not_genuine_soft_votes=list(NOT_GENUINE), pool_rules=POOL_RULES,
        notes=["metrics are fractions in [0,1]; every diff is name minus reference",
               "mean/sd_seed/sd_fold/sd_all and diff_vs_uniform use per-seed CV means (mean of 5 fold values)",
               "bootstrap diffs, CIs and p-values refer to pooled out-of-fold macro-F1 averaged over seeds; "
               "p_boot = 0 means below 2/n_boot, so p_holm = 0 means below m*2/n_boot for a family of m; "
               "replicates with zero difference count towards both tails (conservative for near-identical rules)",
               "primary_analysis (pre-specified): SA-WSV (bootstrap) - uniform, Holm over the 3 corpora; secondary_analysis: "
               "SA-WSV w/o confidence - uniform, its own Holm over the 3 corpora",
               "the two-stage bootstrap conditions on the CV partition and on each fold's validation split, which all seeds "
               "share (rule weights see one validation sample per fold; rev_robustness.py refits on validation subsamples); "
               "resampling S seeds understates the between-seed variance by (S-1)/S: seed_inflated scales the resampled "
               "seeds' deviations by sqrt(S/(S-1)); seed_t_interval is a t-interval over the S per-seed pooled differences "
               "(seed variability only, conditional on the test set)",
               "equiv_<delta>: 90% CI inside (-delta, +delta), i.e. TOST at 5% per side, not multiplicity-corrected",
               "Holm families within corpus and reference: p_holm = all other names, learners included (the specified "
               "family); p_holm_rules = the other rules only (null for learners); McNemar: all other names, within seed",
               "McNemar b = name (or monolingual learner) right and reference (or multilingual learner) wrong, c = reverse",
               "band: per_seed and worst_seed_* are single-run values on per-seed CV means; max_abs_diff_vs_uniform, "
               "argmax, signed_diff and range_incl_uniform are seed-averaged, which shrinks run-to-run differences; "
               "band_claim is the submission's 'within 0.8 points'; genuine_soft_votes = all soft rules except "
               "not_genuine_soft_votes; original_soft_rules = the submission's ten soft-voting rules without uniform",
               "m_eff = participation ratio (tr R)^2/tr(R^2) of the correctness correlations (effective dimensionality); "
               "n_eff_variance = M/(1+(M-1)*rho_mean), the variance-equivalent number of independent voters",
               "seeds are read once at start-up; meta.warnings lists incomplete seeds, differing seed sets and dropped "
               "names; --strict refuses to run with any of them",
               "pools: rules refitted on the pool's validation posteriors; results are descriptive: " + POOL_BASIS]),
        primary_analysis=over_corpora(conf, "primary", "primary (pre-specified)", L.PRIMARY),
        secondary_analysis=over_corpora(conf, "secondary", "secondary", SECONDARY),
        per_corpus=OrderedDict((ds, res) for ds, res, _, _ in results))
    out["across_corpora"] = across_corpora(out["per_corpus"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(clean(out), fh, indent=1)
    os.replace(a.out + ".tmp", a.out)                  # readers never see a half-written file
    for ds, res, _, dt in results:
        print("\n".join(report(ds, res)))
        print(f"[{L.DS_LAB[ds]}: {dt:.1f} s in worker]")
    print("\n===== across corpora: seed-averaged macro-F1 difference to uniform soft voting, points =====")
    print(f"{'rule':30s} {'mean':>6s} {'worst':>6s} {'best':>6s} {'w.seed':>6s} beat sigHolm sigHrul")
    for k, v in sorted(out["across_corpora"].items(), key=lambda kv: -kv[1]["mean_diff_vs_uniform"]):
        print(f"{k:30s} {100 * v['mean_diff_vs_uniform']:+6.2f} {100 * v['worst_corpus_diff']:+6.2f} "
              f"{100 * v['best_corpus_diff']:+6.2f} {100 * v['worst_seed_diff']:+6.2f} "
              f"{v['n_corpora_beat_uniform']}/{v['n_corpora']}  {v['n_corpora_sig_holm']}/{v['n_corpora']}     "
              f"{v['n_corpora_sig_holm_rules']}/{v['n_corpora']}")
    for key in ("primary_analysis", "secondary_analysis"):
        print("\n".join(report_confirmatory(out[key])))
    for w in warn:
        print("WARNING:", w)
    print(f"\nseeds used: {dict(jobs)}")
    print(f"wrote {shown(a.out)} in {time.perf_counter() - t0:.1f} s")
