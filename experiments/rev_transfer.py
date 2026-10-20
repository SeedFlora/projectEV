"""Cross-corpus generalisation of the learners and the fusion rules (reviewer 1: deeper transfer analysis).

Design decisions
* Directions. Every in-domain corpus is a source; its targets are the corpora that share no instances with it
  and whose posteriors are stored for all five learners in every fold (evprimary-* -> evpublic, evtwitter,
  evyoutube; evpublic -> the four others).
* Seeds. Seeds are read once at start-up (L.complete_seeds). Every direction is analysed on the COMMON seed set,
  the seeds of 1-4 that hold all ten directions (1-4 once training has finished), so every cell, drop and
  cross-direction summary rests on the same seeds. Seed 42, the submitted runs, holds only EV-Norm -> EV-Public;
  it is analysed as a separate row (seed42_row) that must reproduce the submitted transfer numbers in final.json.
* Units. A unit is one (seed, fold) set of five fold models. Every fold model scores the whole target corpus,
  so a cell is the mean and SD (ddof=1) over units of the target macro-F1; sd_seed is the SD of the per-seed
  means. The in-domain reference is the same rule's CV value on the source over the same units; drops and
  differences are formed per unit, then averaged. Units share the target instances and, within a seed, most of
  their training data, so unit SDs and sign counts are descriptive only.
* Intervals. Each main rule minus uniform under shift gets percentile CIs from a two-stage bootstrap: a replicate
  resamples the target instances (one index for every fold model, rule and seed) and the seeds; a seed's value
  is the mean over its five fold models. Instance resamples are shared by the directions whose targets are the
  same posts (EV-Norm and EV-Raw are one set of posts), seed resamples by the directions of one source, so the
  same replicates give a CI for the mean over the directions. p_boot is two-sided and not multiplicity-corrected.
* No target label fits anything. Every rule is fitted on the SOURCE fold's validation split
  (L.pooled_run(..., transfer=True)); target labels only score. The one exception is section (e), whose
  quantities carry the prefix oracle_ and are analysis-only upper bounds fitted in-sample on the target labels:
  convex weights minimising target log-loss (L.convex_nll_weights, the objective and solver of L.r_convex_nll),
  the macro-F1-maximising static weights on a simplex grid of step 1/20 (which contains uniform and every single
  learner), the best single learner on the target, and the macro-F1-maximising decision threshold of the uniform
  posterior.
* Comparators. The rule "Best single (val-selected)" takes, per (seed, fold), the learner with the highest
  macro-F1 on that fold's validation split. It is the submission's comparator and the one behind every ensemble
  gain and per-class best-learner comparison, in domain and under shift. "best_val_learner" is one global
  learner per direction with the highest mean validation macro-F1 over all of the direction's units: a hindsight
  baseline that pools every unit's validation data, used for transfer-side rankings. In domain the other folds'
  validation splits hold about a third of each fold's test portion, so its in-domain values are reference only.
* Rule spread (max - min of cell means) over three sets: the submission's ten soft-voting rules
  (L.ORIGINAL_RULES without Best single and Majority), its twelve rules, and every soft-voting rule fitted here
  (all rules except Best single and Majority, 24).
* Diversity. Q and the pairwise error correlation measure how dependent the learners' errors are. Disagreement
  and the contested share also rise when accuracy falls, so the transfer disagreement is set against the value
  that the transfer accuracies imply at the unit's in-domain pairwise error correlation.
* Summaries. Transfer quantities are summarised over the directions, in-domain quantities over the sources (a
  source's in-domain units are shared by its directions). The directions are not independent: those of one
  source share its fold models, and EV-Norm and EV-Raw are the same 1,382 posts in two renderings.
* Lexical statistics use the texts exactly as fed to the models (run_seeds.build_targets, imported lazily so
  that worker processes never load torch), lower-cased alphanumeric tokens and the vocabulary of the whole
  source corpus; the reconstructed labels are asserted equal to L.target_labels. Two additions guard the
  interpretation: the share of target texts with a near-duplicate in the source (token-set Jaccard), which
  rules out leakage, and two preprocessing indicators (function-word share and a heuristic affix share),
  because a released corpus may ship stemmed or stopword-filtered text.
* Units in the JSON: macro-F1, recalls, accuracies, rates, priors, Jaccard, OOV and shares in percent
  (differences in percentage points); Q, error correlation, disagreement, AUROC, log-loss, thresholds and weights
  as raw values. Nothing is fitted on test or target labels outside the oracle_ keys (the in-domain values of
  best_val_learner aside, see Comparators); all randomness is seeded, so two runs give identical JSON.
"""
import os, sys, re, json, time, argparse, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one BLAS thread per worker: bounded CPU use, reproducible sums
from collections import OrderedDict, Counter
from multiprocessing import Pool
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_lib as L

OUT = os.path.join(L.ROOT, "experiments", "results", "rev", "rev_transfer.json")
LEARNERS = [L.MLAB[m] for m in L.MO]
MONO, MULTI = LEARNERS[:3], LEARNERS[3:]
UNI = L.BASELINE
BEST1 = "Best single (val-selected)"
MAIN = [UNI, L.PRIMARY, "SA-WSV (binomial)", "SA-WSV w/o confidence", BEST1, "Stacking (LR)", "Convex NLL weights"]
NOT_SOFT = {BEST1, "Majority voting"}
ORDER = list(L.DS_LAB)                  # corpus order for tables
COMMON_CANDIDATES = [s for s in L.SEEDS if s != 42]       # seed 42 holds only EV-Norm -> EV-Public
SEED42_ROW = ("evprimary-formal", "evpublic")
GRID_STEPS = 20
N_BOOT = 5000
POSTS = {"evprimary-formal": 0, "evprimary-informal": 0, "evpublic": 1, "evtwitter": 2, "evyoutube": 3}  # same posts
TOKEN = re.compile(r"[a-z0-9]+")
FUNCTION_WORDS = {"yang", "dan", "di", "ini", "itu", "tidak", "ada", "dengan", "untuk", "ke", "dari", "akan", "juga",
                  "saya", "kita"}
SHORT = {"IndoBERT-p1": "IB-p1", "IndoBERT-1.5G": "IB-1.5G", "IndoRoBERTa-1.5G": "IR-1.5G", "XLM-R": "XLM-R",
         "mBERT": "mBERT", UNI: "Uniform", L.PRIMARY: "SA-WSV(boot)", "SA-WSV (binomial)": "SA-WSV(binom)",
         "SA-WSV w/o confidence": "SA-WSV w/o conf", BEST1: "Best single", "Stacking (LR)": "Stacking",
         "Convex NLL weights": "Convex NLL"}
SET_LABEL = {"submission_soft": "submitted soft-voting rules", "submission_all": "submitted rules",
             "all_soft": "soft-voting rules (all)"}


def dkey(src, tgt):
    return f"{L.DS_LAB[src]} -> {L.DS_LAB[tgt]}"


def rule_sets(names):
    """the rule sets of the spread: the submission's soft-voting rules, its twelve rules, every soft-voting rule"""
    rules = [k for k in names if k not in LEARNERS]
    return OrderedDict([("submission_soft", [k for k in L.ORIGINAL_RULES if k not in NOT_SOFT]),
                        ("submission_all", list(L.ORIGINAL_RULES)),
                        ("all_soft", [k for k in rules if k not in NOT_SOFT])])


# ========================================================== per-unit quantities (workers)
def _metrics(y, p):
    m = L.full_metrics(y, p)
    return {"f1": 100 * m["f1"], "acc": 100 * m["acc"], "rec_neg": 100 * m["rec_neg"], "rec_pos": 100 * m["rec_pos"],
            "pos_rate": 100 * float(np.mean(p == 1))}


def pair_corr(ok):
    """pairwise correlation of the learners' correctness indicators ok (M, n); 0 for a learner that is always
    right or always wrong"""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(np.corrcoef(ok.astype(float)))


def implied_disagreement(acc, R):
    """mean pairwise disagreement of learners with accuracies acc (M,) whose correctness indicators have the
    pairwise correlation R: P(i != j) = a_i + a_j - 2 (a_i a_j + R_ij s_i s_j), s = sqrt(a (1 - a))"""
    s = np.sqrt(acc * (1.0 - acc))
    i, j = np.triu_indices(len(acc), 1)
    return float(np.mean(acc[i] + acc[j] - 2.0 * (acc[i] * acc[j] + R[i, j] * s[i] * s[j])))


def _diversity(P, y, R_ref=None):
    """R_ref: in-domain pairwise error correlation of the same fold models, for the accuracy-implied disagreement"""
    ok = P.argmax(-1) == y[None]
    q, dis = L.q_statistic(ok)
    ev, k, acc = L.effective_voters(ok), ok.sum(0), ok.mean(1)
    assert abs(implied_disagreement(acc, pair_corr(ok)) - dis) < 1e-9      # the identity the comparison rests on
    out = {"Q": q, "disagreement": dis, "error_corr": ev["rho_mean"], "m_eff": ev["m_eff"],
           "accuracy_mean": 100 * float(acc.mean()),
           "all_correct": 100 * float(np.mean(k == len(P))), "all_wrong": 100 * float(np.mean(k == 0)),
           "contested": 100 * float(np.mean((k > 0) & (k < len(P))))}
    if R_ref is not None:
        out["disagreement_implied_by_accuracy"] = implied_disagreement(acc, R_ref)
    return out


def _aurocs(P, y):
    a = {name: L.auroc(P[m][:, 1], y) for m, name in enumerate(LEARNERS)}
    a[UNI] = L.auroc(L.fuse(P)[:, 1], y)
    return a


def simplex_grid(M, steps):
    """every weight vector with entries in {0, 1/steps, ..., 1} that sums to one (stars and bars)"""
    rows = []
    for bars in itertools.combinations(range(steps + M - 1), M - 1):
        b = (-1,) + bars + (steps + M - 1,)
        rows.append([b[i + 1] - b[i] - 1 for i in range(M)])
    return np.array(rows, float) / steps


def _f1c(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2.0 * tp / np.maximum(d, 1), 0.0)


def oracle_grid(P, y, W, chunk=1024):
    """ORACLE: static convex weights maximising the target macro-F1 over the grid W, in-sample"""
    y1 = y == 1
    n1, n0 = int(y1.sum()), int((~y1).sum())
    best, arg = -1.0, None
    for i in range(0, len(W), chunk):
        Wc = W[i:i + chunk]
        pred = Wc @ P[:, :, 1] > Wc @ P[:, :, 0]            # argmax of the fused posterior, ties -> class 0
        tp = pred[:, y1].sum(1); fp = pred.sum(1) - tp
        f = 0.5 * (_f1c(tp, fp, n1 - tp) + _f1c(n0 - fp, n1 - tp, fp))
        j = int(np.argmax(f))
        if f[j] > best:
            best, arg = float(f[j]), Wc[j]
    return best, arg


def oracle_threshold(y, s):
    """ORACLE: threshold t on s = P(positive), predicting s > t, that maximises the target macro-F1, in-sample"""
    o = np.argsort(-s, kind="stable"); ys, ss = y[o], s[o]
    n, n1 = len(y), int(y.sum())
    tp = np.concatenate([[0], np.cumsum(ys)]); fp = np.arange(n + 1) - tp      # top k predicted positive
    f = 0.5 * (_f1c(tp, fp, n1 - tp) + _f1c((n - n1) - fp, n1 - tp, fp))
    f[1:-1][ss[:-1] <= ss[1:]] = -1.0                   # a cut must separate two distinct scores
    k = int(np.argmax(f))
    return float(f[k]), float(1.0 if k == 0 else 0.0 if k == n else 0.5 * (ss[k - 1] + ss[k]))


def _oracles(Px, y, meta, W):
    M, n = Px.shape[:2]
    pu = L.fuse(Px)
    wv = np.asarray(meta["convex_w"]); pv = L.fuse(Px, wv)
    wo = L.convex_nll_weights(Px[:, np.arange(n), y])   # objective and solver of L.r_convex_nll, on the TARGET
    po = L.fuse(Px, wo)
    guard = min(L.nll(y, L.fuse(Px, v)) for v in [np.full(M, 1.0 / M)] + list(np.eye(M)))
    assert L.nll(y, po) <= guard + 1e-12, "target-fitted convex weights worse than uniform or a single learner"
    g, gw = oracle_grid(Px, y, W)
    t, thr = oracle_threshold(y, pu[:, 1])
    return {"uniform_f1": 100 * L.macro_f1(y, pu.argmax(-1)), "uniform_nll": L.nll(y, pu),
            "convex_val_f1": 100 * L.macro_f1(y, pv.argmax(-1)), "convex_val_nll": L.nll(y, pv),
            "convex_val_w": wv.tolist(),
            "oracle_convex_f1": 100 * L.macro_f1(y, po.argmax(-1)), "oracle_convex_nll": L.nll(y, po),
            "oracle_convex_w": wo.tolist(), "oracle_convex_minus_val_nll": L.nll(y, po) - L.nll(y, pv),
            "oracle_grid_f1": 100 * g, "oracle_grid_w": gw.tolist(),
            "oracle_best_learner_f1": 100 * max(L.macro_f1(y, Px[m].argmax(-1)) for m in range(len(Px))),
            "oracle_threshold_uniform_f1": 100 * t, "oracle_threshold_uniform": thr}


def run_source_seed(job):
    """every per-unit quantity for one (source, seed); rules are fitted per fold on the source validation split"""
    ds, seed, targets = job
    R = L.pooled_run(ds, seed, rules=L.ALL_RULES, transfer=True)
    names = list(R["pred"])
    assert all(len(R["per_fold"][k]) == L.N_FOLDS for k in names), f"a rule is missing in some fold: {ds} seed {seed}"
    W = simplex_grid(len(L.MO), GRID_STEPS)
    units = []
    for fold in range(L.N_FOLDS):
        d = L.load_fold(ds, fold, seed)
        sel = R["fold"] == fold
        b1 = LEARNERS[R["meta"][fold]["best_single"]]     # this fold's validation choice of the Best single rule
        assert np.array_equal(R["pred"][BEST1][sel], R["pred"][b1][sel])
        base = {"seed": seed, "fold": fold, "indomain": {k: _metrics(R["y"][sel], R["pred"][k][sel]) for k in names},
                "val_f1": dict(zip(LEARNERS, (100 * L.macro_f1_batch(d["yv"], d["Pv"].argmax(-1))).tolist())),
                "best_single_choice": b1, "div_in": _diversity(d["Pt"], d["yt"]), "auroc_in": _aurocs(d["Pt"], d["yt"])}
        r_in = pair_corr(d["Pt"].argmax(-1) == d["yt"][None])
        for tgt in targets:
            Px, y, X = d["xfer"][tgt], L.target_labels(tgt), R["xfer"][tgt]
            units.append(dict(base, src=ds, tgt=tgt, transfer={k: _metrics(y, X[k][fold]) for k in names},
                              pred={k: X[k][fold].astype(np.int8) for k in MAIN},
                              div_out=_diversity(Px, y, r_in), auroc_out=_aurocs(Px, y),
                              oracle=_oracles(Px, y, R["meta"][fold], W)))
    return units


def xfer_targets(ds, seed):
    """targets with transfer posteriors of all five learners in every fold"""
    common = set.intersection(*[set(L.load_fold(ds, f, seed)["xfer"]) for f in range(L.N_FOLDS)])
    return [t for t in ORDER if t in common]


# ===================================================================== aggregation
def _agg(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "n": int(len(v))}


def _agg_s(v, sid):
    """_agg plus the SD (ddof=1) of the per-seed means"""
    m = np.array([v[sid == s].mean() for s in dict.fromkeys(sid.tolist())])
    return dict(_agg(v), sd_seed=float(m.std(ddof=1)) if len(m) > 1 else None, n_seeds=int(len(m)))


def _signed(v):
    v = np.asarray(v, float)
    return dict(_agg(v), n_pos=int((v > 0).sum()), n_neg=int((v < 0).sum()))


def _col(U, part, k, m="f1"):
    return np.array([u[part][k][m] for u in U])


def section_a(U, names):
    sid = np.array([u["seed"] for u in U])
    cells = OrderedDict()
    for k in names:
        tr, ind = _col(U, "transfer", k), _col(U, "indomain", k)
        per_seed = OrderedDict((str(s), float(tr[sid == s].mean())) for s in dict.fromkeys(sid.tolist()))
        cells[k] = {"transfer": _agg_s(tr, sid), "indomain": _agg_s(ind, sid), "drop": _agg_s(ind - tr, sid),
                    "transfer_per_seed": per_seed}
    val = OrderedDict((k, float(np.mean([u["val_f1"][k] for u in U]))) for k in LEARNERS)
    best = max(val, key=val.get)
    gain, margin = OrderedDict(), {}
    for part in ("indomain", "transfer"):
        uni = _col(U, part, UNI)
        margin[part] = {"best_single_rule": uni - _col(U, part, BEST1), "best_val_learner": uni - _col(U, part, best)}
        gain[part] = {"uniform_minus_best_single_rule": _signed(margin[part]["best_single_rule"]),
                      "uniform_minus_best_val_learner": _signed(margin[part]["best_val_learner"]),
                      "uniform_minus_mean_learner": _signed(uni - np.mean([_col(U, part, k) for k in LEARNERS], 0))}
    change = OrderedDict((f"uniform_minus_{c}", _signed(margin["transfer"][c] - margin["indomain"][c]))
                         for c in ("best_single_rule", "best_val_learner"))
    return {"best_val_learner": best, "val_f1_mean": val,
            "best_single_rule_choices": dict(Counter(u["best_single_choice"] for u in U).most_common()),
            "cells": cells, "ensemble_gain": gain, "ensemble_gain_shift_minus_indomain": change}


def section_b(U, A):
    fin = np.array([A["cells"][k]["indomain"]["mean"] for k in LEARNERS])
    fout = np.array([A["cells"][k]["transfer"]["mean"] for k in LEARNERS])
    rho_u = np.array([spearmanr([u["indomain"][k]["f1"] for k in LEARNERS], [u["transfer"][k]["f1"] for k in LEARNERS])[0]
                      for u in U])
    best = A["best_val_learner"]
    rank_out = dict(zip(LEARNERS, rankdata(-fout).tolist()))
    mono = {p: np.mean([_col(U, p, k) for k in MONO], 0) for p in ("indomain", "transfer")}
    pairs = OrderedDict((f"{a} - {b}", {p: _signed(_col(U, p, a) - _col(U, p, b)) for p in ("indomain", "transfer")})
                        for a, b in itertools.product(MONO, MULTI))
    top_u = [u["best_single_choice"] == max(LEARNERS, key=lambda k: u["transfer"][k]["f1"]) for u in U]
    return {"spearman_cell_means": float(spearmanr(fin, fout)[0]),
            "spearman_per_unit": dict(_agg(rho_u), n_pos=int((rho_u > 0).sum())),
            "rank_indomain": dict(zip(LEARNERS, rankdata(-fin).tolist())), "rank_transfer": rank_out,
            "best_val_learner": best, "best_val_learner_rank_transfer": rank_out[best],
            "top_learner_on_target": LEARNERS[int(np.argmax(fout))],
            "n_units_val_selected_learner_top_on_target": int(sum(top_u)), "n_units": len(U),
            "mono_minus_multi_mean": {p: _signed(mono[p] - np.mean([_col(U, p, k) for k in MULTI], 0)) for p in mono},
            "mean_mono_minus": OrderedDict((m, {p: _signed(mono[p] - _col(U, p, m)) for p in mono}) for m in MULTI),
            "mono_minus_multi_pair_range": {p: [min(v[p]["mean"] for v in pairs.values()),
                                                max(v[p]["mean"] for v in pairs.values())] for p in mono},
            "n_pairs_mono_better": {p: int(sum(v[p]["mean"] > 0 for v in pairs.values())) for p in mono},
            "pairs": pairs}


def section_c(U, A, src, tgt, names):
    out = {"source_prior_pos": 100 * float(L.target_labels(src).mean()),
           "target_prior_pos": 100 * float(L.target_labels(tgt).mean()),
           "best_single_rule": BEST1, "best_val_learner": A["best_val_learner"], "rules": OrderedDict()}
    for k in names:
        r = {}
        for m in ("rec_neg", "rec_pos"):
            ind, tr = _col(U, "indomain", k, m), _col(U, "transfer", k, m)
            r[m] = {"indomain": float(ind.mean()), "transfer": float(tr.mean()), "drop": _agg(ind - tr)}
        dn, dp = r["rec_neg"]["drop"]["mean"], r["rec_pos"]["drop"]["mean"]
        r["neg_share_of_recall_drop"] = dn / (dn + dp) if dn + dp > 0 else None
        pin, pout = _col(U, "indomain", k, "pos_rate"), _col(U, "transfer", k, "pos_rate")
        r["pos_rate"] = {"indomain": float(pin.mean()), "transfer": float(pout.mean()),
                         "transfer_minus_target_prior": float(pout.mean()) - out["target_prior_pos"]}
        out["rules"][k] = r
    return out


def _spread(U, A, ks):
    out = OrderedDict(n_rules=len(ks))
    for p in ("transfer", "indomain"):
        mu = {k: A["cells"][k][p]["mean"] for k in ks}
        hi, lo = max(mu, key=mu.get), min(mu, key=mu.get)
        per_unit = [max(u[p][k]["f1"] for k in ks) - min(u[p][k]["f1"] for k in ks) for u in U]
        out[p] = {"max_minus_min": mu[hi] - mu[lo], "max_rule": hi, "min_rule": lo,
                  "max_abs_diff_to_uniform": max(abs(mu[k] - mu[UNI]) for k in ks),
                  "per_unit_max_minus_min": _agg(per_unit)}
    return out


def section_d(U, names, A):
    rules = [k for k in names if k not in LEARNERS]
    by_set = OrderedDict((s, _spread(U, A, ks)) for s, ks in rule_sets(names).items())
    diff = OrderedDict((k, {p: _signed(_col(U, p, k) - _col(U, p, UNI)) for p in ("transfer", "indomain")}) for k in rules)
    return {"soft_voting_rules": rule_sets(names)["all_soft"],
            "spread": OrderedDict((p, by_set["all_soft"][p]) for p in ("transfer", "indomain")),
            "spread_by_set": by_set, "diff_to_uniform": diff}


def section_e(U, A):
    O = [u["oracle"] for u in U]
    f1k = ["uniform_f1", "convex_val_f1", "oracle_convex_f1", "oracle_grid_f1", "oracle_best_learner_f1",
           "oracle_threshold_uniform_f1"]
    out = OrderedDict((k, _agg([o[k] for o in O])) for k in f1k + ["uniform_nll", "convex_val_nll", "oracle_convex_nll",
                                                                   "oracle_threshold_uniform"])
    gap = np.array([o["oracle_convex_minus_val_nll"] for o in O])  # <= 0: the oracle is optimal on the target
    out["oracle_convex_minus_val_nll"] = dict(_agg(gap), max=float(gap.max()))
    drop = A["cells"][UNI]["drop"]["mean"]
    out["gain_over_uniform"] = OrderedDict()
    for k in f1k[1:]:
        g = np.array([o[k] - o["uniform_f1"] for o in O])
        out["gain_over_uniform"][k] = dict(_signed(g), min=float(g.min()), max=float(g.max()),
                                           share_of_uniform_drop=float(g.mean() / drop) if drop > 0 else None)
    out["weights_mean"] = {w: dict(zip(LEARNERS, np.mean([o[w] for o in O], 0).tolist()))
                           for w in ("convex_val_w", "oracle_convex_w", "oracle_grid_w")}
    return out


def section_f(U):
    out = OrderedDict((k, {"indomain": _agg([u["div_in"][k] for u in U]), "transfer": _agg([u["div_out"][k] for u in U])})
                      for k in U[0]["div_in"])
    d = out["disagreement"]
    d["transfer_implied_by_accuracy"] = _agg([u["div_out"]["disagreement_implied_by_accuracy"] for u in U])
    rise = d["transfer"]["mean"] - d["indomain"]["mean"]
    d["share_of_rise_implied_by_accuracy"] = ((d["transfer_implied_by_accuracy"]["mean"] - d["indomain"]["mean"]) / rise
                                              if rise > 0 else None)
    out["auroc"] = OrderedDict((k, {"indomain": _agg([u["auroc_in"][k] for u in U]),
                                    "transfer": _agg([u["auroc_out"][k] for u in U])}) for k in U[0]["auroc_in"])
    return out


def analyse(U, s, t):
    """sections (a)-(f) of one direction from its units"""
    names = list(U[0]["transfer"])
    assert all(list(u["transfer"]) == names for u in U)
    R = OrderedDict(source=s, target=t, seeds=list(dict.fromkeys(u["seed"] for u in U)), n_units=len(U))
    R["a"] = section_a(U, names)
    R["b"] = section_b(U, R["a"])
    R["c"] = section_c(U, R["a"], s, t, names)
    R["d"] = section_d(U, names, R["a"])
    R["e"] = section_e(U, R["a"])
    R["f"] = section_f(U)
    R["per_unit_f1"] = {"seed_fold": [[u["seed"], u["fold"]] for u in U],
                        "transfer": OrderedDict((k, _col(U, "transfer", k).tolist()) for k in names),
                        "indomain": OrderedDict((k, _col(U, "indomain", k).tolist()) for k in names)}
    return R


# ========================================== two-stage bootstrap under shift (main rules minus uniform)
def boot_f1(y, preds, idx, chunk=500):
    """macro-F1 of every row of preds (K, n) on every resample in idx (B, n), from the resampled confusion
    counts; bit-identical to L.macro_f1_batch(y, preds, idx) (integer counts), one matrix product per chunk"""
    n = preds.shape[1]
    y1 = (y == 1).astype(float)
    pos = (preds == 1).astype(float)
    T, Fp = (pos * y1).T, (pos * (1.0 - y1)).T             # (n, K): predicted positive, right / wrong
    out = np.empty((len(preds), len(idx)))
    for i in range(0, len(idx), chunk):
        ii = idx[i:i + chunk]; b = len(ii)
        C = np.bincount((ii + n * np.arange(b)[:, None]).ravel(), minlength=b * n).reshape(b, n).astype(float)
        n1 = C @ y1
        tp, fp = C @ T, C @ Fp
        fn, tn = n1[:, None] - tp, (n - n1)[:, None] - fp
        out[:, i:i + b] = (0.5 * (_f1c(tp, fp, fn) + _f1c(tn, fn, fp))).T
    return out


def boot_direction(U, tgt, idx, sidx):
    """two-stage bootstrap of the seed-averaged target macro-F1 (percent) of the MAIN rules. Replicate b
    resamples the target instances idx[b] (the same for every fold model, rule and seed) and the seeds sidx[b];
    a seed's value is the mean over its fold models. Returns ({rule: point}, {rule: (B,) replicates})."""
    y = L.target_labels(tgt)
    seeds = list(dict.fromkeys(u["seed"] for u in U))
    by = [[u for u in U if u["seed"] == s] for s in seeds]
    assert len({len(b) for b in by}) == 1, "seeds with different numbers of folds"
    X = np.stack([[[u["pred"][k] for u in b] for b in by] for k in MAIN])        # (K, S, F, n)
    K, S, F, n = X.shape
    flat = X.reshape(-1, n)
    assert np.array_equal(boot_f1(y, flat, idx[:16]), L.macro_f1_batch(y, flat, idx[:16])), "boot_f1 != macro_f1_batch"
    f = 100 * boot_f1(y, flat, idx).reshape(K, S, F, -1).mean(2)                # (K, S, B)
    point = OrderedDict((k, float(100 * np.mean([L.macro_f1(y, p) for p in X[i].reshape(-1, n)])))
                        for i, k in enumerate(MAIN))
    col = np.arange(len(idx))[:, None]
    return point, OrderedDict((k, f[i][sidx, col].mean(1)) for i, k in enumerate(MAIN))


def shift_intervals(D, UD, n_boot):
    """percentile CIs of each main rule minus uniform for every direction of D (units UD[key]) and for the mean over
    D's directions. Instance resamples are shared by the directions whose targets are the same posts (POSTS),
    seed resamples by the directions of one source; different posts and sources are resampled independently."""
    sidx, point, reps = {}, OrderedDict(), OrderedDict()
    for g in dict.fromkeys(POSTS[R["target"]] for R in D.values()):
        keys = [k for k, R in D.items() if POSTS[R["target"]] == g]
        n = len(L.target_labels(D[keys[0]]["target"]))
        idx = L.stratified_boot_index(np.zeros(n, int), n_boot, np.random.default_rng([0, g]))   # one stratum
        for key in keys:
            R = D[key]
            if R["source"] not in sidx:
                S = len(R["seeds"])
                sidx[R["source"]] = np.random.default_rng([1, L.DS_ID[R["source"]]]).integers(0, S, size=(n_boot, S))
            assert len(L.target_labels(R["target"])) == n
            point[key], reps[key] = boot_direction(UD[key], R["target"], idx, sidx[R["source"]])
    ci = OrderedDict()
    for key, R in D.items():
        ci[key] = OrderedDict((k, L.compare_from_reps(point[key], reps[key], k, UNI)) for k in MAIN if k != UNI)
        for k in MAIN:      # the point estimates are the cell means and the mean unit differences
            assert abs(point[key][k] - R["a"]["cells"][k]["transfer"]["mean"]) < 1e-9
            assert k == UNI or abs(ci[key][k]["diff"] - R["d"]["diff_to_uniform"][k]["transfer"]["mean"]) < 1e-9
    pm = {k: float(np.mean([point[key][k] for key in D])) for k in MAIN}
    rm = {k: np.mean([reps[key][k] for key in D], 0) for k in MAIN}
    return ci, OrderedDict((k, L.compare_from_reps(pm, rm, k, UNI)) for k in MAIN if k != UNI)


# ===================================================================== cross-direction summaries
def across_directions(D, pooled):
    """direction-level mean difference to uniform, summarised over the directions (cf. the in-domain regret table);
    band = best - worst direction; the main rules add CI counts and the CI of the mean over the directions"""
    out = OrderedDict()
    for k in D[next(iter(D))]["d"]["diff_to_uniform"]:
        v = {dk: R["d"]["diff_to_uniform"][k]["transfer"]["mean"] for dk, R in D.items()}
        x = np.array(list(v.values()))
        tot = {f: sum(R["d"]["diff_to_uniform"][k]["transfer"][f] for R in D.values()) for f in ("n_pos", "n_neg", "n")}
        out[k] = {"mean": float(x.mean()), "worst": float(x.min()), "best": float(x.max()), "band": float(x.max() - x.min()),
                  "worst_direction": min(v, key=v.get), "n_directions_pos": int((x > 0).sum()), "n_directions": len(x),
                  "n_units_pos": tot["n_pos"], "n_units_neg": tot["n_neg"], "n_units": tot["n"]}
        if k in pooled:
            ci = [R["d"]["ci_vs_uniform"][k]["ci95"] for R in D.values()]
            out[k].update(n_directions_ci95_above_zero=int(sum(c[0] > 0 for c in ci)),
                          n_directions_ci95_below_zero=int(sum(c[1] < 0 for c in ci)), mean_over_directions=pooled[k])
    return out


def cross_summary(D):
    """transfer quantities summarised over the directions, in-domain quantities over the sources"""
    SRC = OrderedDict()                 # a source's in-domain units are shared by all of its directions
    for R in D.values():
        SRC.setdefault(L.DS_LAB[R["source"]], R)
    for R in D.values():
        ref = SRC[L.DS_LAB[R["source"]]]["a"]["cells"]
        assert all(c["indomain"] == ref[k]["indomain"] for k, c in R["a"]["cells"].items()), "in-domain units differ"

    def rng(f, over=None):              # entries where f is undefined (None) are left out and counted
        v = OrderedDict((k, float(x)) for k, x in ((k, f(R)) for k, R in (over or D).items()) if x is not None)
        x = np.array(list(v.values()))
        return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()), "argmin": min(v, key=v.get),
                "argmax": max(v, key=v.get), ("n_sources" if over is SRC else "n_directions"): len(v)}

    def count(f, over=None):
        return int(sum(bool(f(R)) for R in (over or D).values()))

    def io(f, pos=False):
        """f(R, part): in-domain over the sources, transfer over the directions"""
        out = OrderedDict([("indomain", rng(lambda R: f(R, "indomain"), SRC)), ("transfer", rng(lambda R: f(R, "transfer")))])
        if pos:
            out["indomain"]["n_sources_pos"] = count(lambda R: f(R, "indomain") > 0, SRC)
            out["transfer"]["n_directions_pos"] = count(lambda R: f(R, "transfer") > 0)
        return out

    def recall(pick):
        return {"rec_neg_drop": rng(lambda R: pick(R)["rec_neg"]["drop"]["mean"]),
                "rec_pos_drop": rng(lambda R: pick(R)["rec_pos"]["drop"]["mean"]),
                "n_directions_pos_drop_larger": count(lambda R: pick(R)["rec_pos"]["drop"]["mean"] >
                                                      pick(R)["rec_neg"]["drop"]["mean"])}
    S = OrderedDict(n_directions=len(D), n_sources=len(SRC))
    S["note"] = ("transfer entries are summarised over the directions and in-domain entries over the sources; "
                 "n_directions_* counts compare each direction's transfer value with its own source's in-domain value")
    S["uniform_transfer_f1"] = rng(lambda R: R["a"]["cells"][UNI]["transfer"]["mean"])
    S["uniform_indomain_f1"] = rng(lambda R: R["a"]["cells"][UNI]["indomain"]["mean"], SRC)
    S["uniform_drop"] = rng(lambda R: R["a"]["cells"][UNI]["drop"]["mean"])
    S["n_directions_uniform_above_indomain"] = count(lambda R: R["a"]["cells"][UNI]["drop"]["mean"] < 0)
    S["learner_drop"] = OrderedDict((k, rng(lambda R, k=k: R["a"]["cells"][k]["drop"]["mean"])) for k in LEARNERS)
    for c in ("best_single_rule", "best_val_learner"):      # the submission's comparator; the hindsight baseline
        g = io(lambda R, p, c=c: R["a"]["ensemble_gain"][p][f"uniform_minus_{c}"]["mean"], pos=True)
        ch = lambda R, c=c: R["a"]["ensemble_gain_shift_minus_indomain"][f"uniform_minus_{c}"]["mean"]
        g["shift_minus_indomain"] = dict(rng(ch), n_directions_pos=count(lambda R: ch(R) > 0))
        S[f"uniform_minus_{c}"] = g
    S["uniform_minus_mean_learner_transfer"] = rng(lambda R: R["a"]["ensemble_gain"]["transfer"]["uniform_minus_mean_learner"]["mean"])
    S["spearman_learner_order"] = dict(rng(lambda R: R["b"]["spearman_cell_means"]),
                                       n_directions_pos=count(lambda R: R["b"]["spearman_cell_means"] > 0))
    S["n_directions_best_val_learner_stays_top"] = count(lambda R: R["b"]["best_val_learner_rank_transfer"] == 1)
    S["n_units_val_selected_learner_top_on_target"] = {
        "n": sum(R["b"]["n_units_val_selected_learner_top_on_target"] for R in D.values()),
        "of": sum(R["b"]["n_units"] for R in D.values())}
    S["top_learner_on_target"] = dict(Counter(R["b"]["top_learner_on_target"] for R in D.values()).most_common())
    S["mono_minus_multi_mean"] = io(lambda R, p: R["b"]["mono_minus_multi_mean"][p]["mean"])
    S["mono_minus_multi_mean"]["n_directions_gap_grows"] = count(
        lambda R: R["b"]["mono_minus_multi_mean"]["transfer"]["mean"] > R["b"]["mono_minus_multi_mean"]["indomain"]["mean"])
    for m in MULTI:
        gap = lambda R, p, m=m: R["b"]["mean_mono_minus"][m][p]["mean"]
        g = io(gap)
        g["n_directions_multilingual_ahead"] = {"transfer": count(lambda R: gap(R, "transfer") < 0)}
        g["n_sources_multilingual_ahead"] = {"indomain": count(lambda R: gap(R, "indomain") < 0, SRC)}
        S[f"mean_mono_minus_{m}"] = g
    S["recall_uniform"] = recall(lambda R: R["c"]["rules"][UNI])
    S["recall_best_single_rule"] = recall(lambda R: R["c"]["rules"][BEST1])
    S["recall_best_val_learner"] = recall(lambda R: R["c"]["rules"][R["c"]["best_val_learner"]])
    S["soft_rule_spread"] = io(lambda R, p: R["d"]["spread"][p]["max_minus_min"])
    S["soft_rule_spread"]["n_directions_larger_under_shift"] = count(
        lambda R: R["d"]["spread"]["transfer"]["max_minus_min"] > R["d"]["spread"]["indomain"]["max_minus_min"])
    S["rule_spread_by_set"] = OrderedDict()
    for name in next(iter(D.values()))["d"]["spread_by_set"]:
        sp = lambda R, p, name=name: R["d"]["spread_by_set"][name][p]
        e = OrderedDict(n_rules=next(iter(D.values()))["d"]["spread_by_set"][name]["n_rules"])
        e.update(io(lambda R, p: sp(R, p)["max_minus_min"]))
        e["indomain"]["n_sources_under_1pp"] = count(lambda R: sp(R, "indomain")["max_minus_min"] < 1, SRC)
        e["transfer"]["n_directions_under_1pp"] = count(lambda R: sp(R, "transfer")["max_minus_min"] < 1)
        e["n_directions_larger_under_shift"] = count(lambda R: sp(R, "transfer")["max_minus_min"] >
                                                     sp(R, "indomain")["max_minus_min"])
        e["max_abs_diff_to_uniform"] = io(lambda R, p: sp(R, p)["max_abs_diff_to_uniform"])
        S["rule_spread_by_set"][name] = e
    S["oracle_gain_over_uniform"] = OrderedDict(
        (k, dict(rng(lambda R, k=k: R["e"]["gain_over_uniform"][k]["mean"]),
                 share_of_uniform_drop=rng(lambda R, k=k: R["e"]["gain_over_uniform"][k]["share_of_uniform_drop"])))
        for k in ("convex_val_f1", "oracle_convex_f1", "oracle_grid_f1", "oracle_best_learner_f1",
                  "oracle_threshold_uniform_f1"))
    S["oracle_convex_minus_val_nll_max"] = max(R["e"]["oracle_convex_minus_val_nll"]["max"] for R in D.values())
    S["diversity"] = OrderedDict((k, io(lambda R, p, k=k: R["f"][k][p]["mean"]))
                                 for k in ("Q", "error_corr", "disagreement", "contested", "all_wrong", "accuracy_mean"))
    S["diversity"]["basis"] = ("Q and error_corr measure how dependent the learners' errors are; disagreement and the "
                               "contested share also rise when accuracy falls (see disagreement_implied_by_accuracy)")
    S["diversity"]["n_directions_Q_lower_under_shift"] = count(lambda R: R["f"]["Q"]["transfer"]["mean"] <
                                                               R["f"]["Q"]["indomain"]["mean"])
    S["diversity"]["n_directions_error_corr_lower_under_shift"] = count(lambda R: R["f"]["error_corr"]["transfer"]["mean"] <
                                                                        R["f"]["error_corr"]["indomain"]["mean"])
    S["diversity"]["n_directions_contested_higher_under_shift"] = count(
        lambda R: R["f"]["contested"]["transfer"]["mean"] > R["f"]["contested"]["indomain"]["mean"])
    S["diversity"]["disagreement_implied_by_accuracy"] = rng(
        lambda R: R["f"]["disagreement"]["transfer_implied_by_accuracy"]["mean"])
    S["diversity"]["share_of_disagreement_rise_implied_by_accuracy"] = rng(
        lambda R: R["f"]["disagreement"]["share_of_rise_implied_by_accuracy"])
    S["auroc"] = OrderedDict((k, io(lambda R, p, k=k: R["f"]["auroc"][k][p]["mean"])) for k in [UNI] + LEARNERS)
    lab = lambda ds: "EV-Norm/EV-Raw posts" if POSTS[ds] == 0 else L.DS_LAB[ds]
    S["dependence"] = {
        "note": "directions of one source share its fold models; directions whose targets are the same posts share "
                "the target instances; EV-Norm and EV-Raw are the same 1,382 posts in two renderings",
        "shared_fold_models": {s: [k for k, R in D.items() if L.DS_LAB[R["source"]] == s] for s in SRC},
        "shared_target_instances": OrderedDict(
            (lab(R["target"]), [k for k, Q in D.items() if POSTS[Q["target"]] == POSTS[R["target"]]]) for R in D.values())}
    return S


# ===================================================================== lexical shift
def _affixed(w):
    """heuristic marker of unstemmed Indonesian: meN- prefix, -kan suffix or -nya clitic"""
    return (len(w) > 5 and (w.startswith(("mem", "men")) or w.endswith("kan"))) or (len(w) > 4 and w.endswith("nya"))


def nearest_jaccard(At, As, chunk=512):
    """largest token-set Jaccard similarity of every target text (rows of At) to any source text (rows of As)"""
    ls, out = np.asarray(As.sum(1)).ravel(), []
    for i in range(0, At.shape[0], chunk):
        A = At[i:i + chunk]
        inter = (A @ As.T).toarray()
        la = np.asarray(A.sum(1)).ravel()
        out.append((inter / np.maximum(la[:, None] + ls[None] - inter, 1)).max(1))
    return np.concatenate(out)


def lexical(directions):
    import run_seeds                    # pulls in torch and transformers; only the parent process gets here
    T = run_seeds.build_targets()
    vocab, prior, corpora, sets, index = {}, {}, OrderedDict(), {}, {}
    for name in ORDER:
        X, y = T[name]
        assert np.array_equal(y, L.target_labels(name)), f"reconstructed labels differ for {name}"
        toks = [TOKEN.findall(str(t).lower()) for t in X]
        flat = list(itertools.chain.from_iterable(toks))
        n_tok = np.array([len(t) for t in toks])
        vocab[name], prior[name] = Counter(flat), 100 * float(y.mean())
        corpora[L.DS_LAB[name]] = {"n_texts": int(len(X)), "pos_prior": prior[name], "mean_tokens": float(n_tok.mean()),
                                   "median_tokens": float(np.median(n_tok)), "p95_tokens": float(np.percentile(n_tok, 95)),
                                   "mean_ws_words": float(np.mean([len(str(t).split()) for t in X])),
                                   "n_tokens": int(n_tok.sum()), "n_types": len(vocab[name]),
                                   "function_word_share": 100 * sum(vocab[name][w] for w in FUNCTION_WORDS) / len(flat),
                                   "affix_share": 100 * sum(c for w, c in vocab[name].items() if _affixed(w)) / len(flat)}
        rc = [(i, index.setdefault(w, len(index))) for i, t in enumerate(toks) for w in set(t)]
        sets[name] = (len(toks), rc)
    S = {k: sparse.csr_matrix((np.ones(len(rc)), tuple(np.array(rc).T)), shape=(n, len(index)))
         for k, (n, rc) in sets.items()}
    pairs = OrderedDict()
    for s in L.IN_DOMAIN:
        Vs = set(vocab[s])
        for t in ORDER:
            if t == s:
                continue
            Ct = vocab[t]; Vt = set(Ct)
            nj = nearest_jaccard(S[t], S[s])
            pairs[dkey(s, t)] = {"source": s, "target": t, "transfer_direction": dkey(s, t) in directions,
                                 "jaccard": 100 * len(Vs & Vt) / len(Vs | Vt),
                                 "oov_token": 100 * sum(c for w, c in Ct.items() if w not in Vs) / sum(Ct.values()),
                                 "oov_type": 100 * len(Vt - Vs) / len(Vt), "prior_shift": prior[t] - prior[s],
                                 "near_dup_share_j50": 100 * float(np.mean(nj >= 0.5)),
                                 "near_dup_share_j80": 100 * float(np.mean(nj >= 0.8))}
    return {"tokenisation": "lower-cased [a-z0-9]+ tokens; source vocabulary = all texts of the source corpus",
            "function_words": sorted(FUNCTION_WORDS),
            "affix_rule": "token longer than 5 with prefix mem-/men- or suffix -kan, or longer than 4 with suffix -nya",
            "near_dup_rule": "share of target texts whose token set has Jaccard >= 0.5 (0.8) with some source text",
            "corpora": corpora, "pairs": pairs}


def lexical_vs_drop(D, G):
    """descriptive rank correlations across the transfer directions (not independent: shared sources and targets)"""
    ks = list(D)
    drop = [D[k]["a"]["cells"][UNI]["drop"]["mean"] for k in ks]
    f1 = [D[k]["a"]["cells"][UNI]["transfer"]["mean"] for k in ks]
    out = {"n_directions": len(ks)}
    if len(ks) >= 3:
        for m in ("oov_token", "jaccard"):
            x = [G["pairs"][k][m] for k in ks]
            out[f"spearman_{m}_vs_uniform_drop"] = float(spearmanr(x, drop)[0])
            out[f"spearman_{m}_vs_uniform_transfer_f1"] = float(spearmanr(x, f1)[0])
        out["spearman_abs_prior_shift_vs_uniform_drop"] = float(spearmanr([abs(G["pairs"][k]["prior_shift"]) for k in ks],
                                                                          drop)[0])
    return out


def regression_seed42(U):
    """seed-42 EV-Norm -> EV-Public must reproduce the submitted transfer numbers (final.json) exactly"""
    p = os.path.join(L.ROOT, "experiments", "results", "final.json")
    U42 = [u for u in U if u["seed"] == 42]
    if not U42 or not os.path.exists(p):
        return None
    worst, n = 0.0, 0
    for k, v in json.load(open(p))["transfer"].items():
        x = _col(U42, "transfer", "SA-WSV (binomial)" if k == "SA-WSV (proposed)" else k)
        worst = max(worst, abs(x.mean() - 100 * v["f1"]["mean"]), abs(x.std(ddof=1) - 100 * v["f1"]["std"])); n += 1
    return {"n_compared": n, "max_abs_diff_pp": worst, "passed": bool(worst < 1e-9)}


def _round(x, k=6):
    if isinstance(x, dict):
        return OrderedDict((kk, _round(v, k)) for kk, v in x.items())
    if isinstance(x, (list, tuple)):
        return [_round(v, k) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), k) if np.isfinite(x) else None
    return x


# ========================================================================= printing
def _ms(c):
    return f"{c['mean']:5.2f}+-{c['sd']:4.2f}" if c["sd"] is not None else f"{c['mean']:5.2f}"


def _ci(c):
    return f"{c['diff']:+.2f} [{c['ci95'][0]:+.2f},{c['ci95'][1]:+.2f}]"


def report(res):
    D, X = res["directions"], res["seed42_row"]
    rows = list(D.items()) + ([(f"{X['direction']} s42", X)] if X else [])     # the seed-42 row is printed apart
    print(f"\ncommon seed set {res['common_seeds']} for every direction (units = seeds x 5 folds):")
    for k, R in D.items():
        print(f"  {k:24s} seeds={R['seeds']} units={R['n_units']}")
    if X:
        print(f"  {X['direction']:24s} seeds={X['seeds']} units={X['n_units']}  (separate row: the submitted runs)")
    nc = {k: v for k, v in res["seeds_complete_not_common"].items() if v}
    if nc:
        print(f"  complete seeds left out because they do not hold every direction yet: {nc}")
    print("\n(a) Uniform soft voting: target macro-F1, mean+-SD over units; [in-domain CV over the source's units]")
    print("  " + "source \\ target".ljust(16) + "".join(L.DS_LAB[t].rjust(15) for t in ORDER))
    for s in L.IN_DOMAIN:
        ref = res["a_indomain_reference"].get(L.DS_LAB[s])
        row = [(f"[{ref['f1'][UNI]['mean']:.2f}]" if ref else "--") if t == s else
               (_ms(D[dkey(s, t)]["a"]["cells"][UNI]["transfer"]) if dkey(s, t) in D else "--") for t in ORDER]
        print("  " + L.DS_LAB[s].ljust(16) + "".join(r.rjust(15) for r in row))
    if X:
        c = X["a"]["cells"][UNI]
        print(f"  seed 42 (submitted runs): {X['direction']} {_ms(c['transfer'])} [in-domain {c['indomain']['mean']:.2f}]")
    print("\n  transfer macro-F1 (change vs in-domain CV on the same units), pp; between-seed SD of the uniform cell")
    print("  " + "direction".ljust(28) + "".join(SHORT[n].rjust(16) for n in MAIN) + "  sd_seed")
    for k, R in rows:
        c = R["a"]["cells"]
        sds = c[UNI]["transfer"]["sd_seed"]
        print("  " + k.ljust(28) + "".join(f"{c[n]['transfer']['mean']:6.2f} ({-c[n]['drop']['mean']:+6.2f})".rjust(16)
                                           for n in MAIN) + (f"  {sds:.2f}" if sds is not None else "  --"))
    print("\n  learners: transfer F1 (change); gain = uniform - Best single (val-selected per fold), in-domain / transfer"
          " [hindsight: uniform - global validation-best learner, transfer]")
    for k, R in rows:
        c, g = R["a"]["cells"], R["a"]["ensemble_gain"]
        print("  " + k.ljust(28) + " ".join(f"{SHORT[n]}={c[n]['transfer']['mean']:.2f}({-c[n]['drop']['mean']:+.1f})"
                                           for n in LEARNERS) +
              f"  gain {g['indomain']['uniform_minus_best_single_rule']['mean']:+.2f} / "
              f"{g['transfer']['uniform_minus_best_single_rule']['mean']:+.2f}"
              f" [{g['transfer']['uniform_minus_best_val_learner']['mean']:+.2f}]")
    print("  mean validation macro-F1 of the two best learners over each source's units, and the per-fold choices of"
          " Best single (val-selected):")
    seen = set()
    for k, R in rows:
        lab = f"{L.DS_LAB[R['source']]} {R['seeds']}"
        if lab in seen:
            continue
        seen.add(lab)
        v = R["a"]["val_f1_mean"]
        a, b = sorted(v, key=v.get, reverse=True)[:2]
        print(f"    {lab:24s} {a} {v[a]:.2f} vs {b} {v[b]:.2f}; per-fold choices "
              + ", ".join(f"{SHORT[m]} {n}" for m, n in R["a"]["best_single_rule_choices"].items()))
    print("\n(b) learner order (Spearman in-domain vs transfer F1 of the 5 learners) and monolingual - multilingual gaps, pp")
    print("  " + "direction".ljust(24) + "rho  rho/unit  val best -> rank  top on target   3mono-2multi    mono-XLM-R"
          "     mono-mBERT   per-fold choice top on target")
    for k, R in D.items():
        b = R["b"]
        io = lambda x: f"{x['indomain']['mean']:+5.2f}->{x['transfer']['mean']:+5.2f}"
        print(f"  {k:24s}{b['spearman_cell_means']:+.1f}  {b['spearman_per_unit']['mean']:+5.2f}"
              f"     {SHORT[b['best_val_learner']]:>7s} -> {b['best_val_learner_rank_transfer']:.0f}"
              f"     {SHORT[b['top_learner_on_target']]:>8s}   {io(b['mono_minus_multi_mean'])}  "
              f"{io(b['mean_mono_minus']['XLM-R'])}  {io(b['mean_mono_minus']['mBERT'])}"
              f"   {b['n_units_val_selected_learner_top_on_target']}/{b['n_units']} units")
    print("\n(c) per-class recall in-domain -> transfer, predicted positive rate on the target; prior = % positive")
    for k, R in rows:
        c = R["c"]
        line = f"  {k:28s} prior {c['source_prior_pos']:.1f}->{c['target_prior_pos']:.1f} |"
        for n in (UNI, BEST1):
            r = c["rules"][n]
            line += (f" {SHORT[n]}: rec- {r['rec_neg']['indomain']:.1f}->{r['rec_neg']['transfer']:.1f}"
                     f" rec+ {r['rec_pos']['indomain']:.1f}->{r['rec_pos']['transfer']:.1f}"
                     f" pos-rate {r['pos_rate']['transfer']:.1f} |")
        print(line)
    sizes = {s: len(v) for s, v in res["rule_sets"].items()}
    print("\n(d) rule spread (max - min of cell means), transfer / in-domain, pp: "
          + " | ".join(f"{sizes[s]} {SET_LABEL[s]}" for s in sizes) + "; max |rule - uniform| of the first set")
    for k, R in rows:
        sb = R["d"]["spread_by_set"]
        print(f"  {k:28s} " + " | ".join(f"{sb[s]['transfer']['max_minus_min']:.2f} / {sb[s]['indomain']['max_minus_min']:.2f}"
                                         for s in sb)
              + f"   max |rule - uniform| {sb['submission_soft']['transfer']['max_abs_diff_to_uniform']:.2f}"
                f" / {sb['submission_soft']['indomain']['max_abs_diff_to_uniform']:.2f}")
    print(f"\n  main rules minus uniform under shift, pp: mean over units [95% CI; two-stage bootstrap of target instances"
          f" and seeds, B={res['bootstrap']['n_boot']}]")
    print("  " + "direction".ljust(28) + "".join(SHORT[n].rjust(22) for n in MAIN[1:]))
    for k, R in rows:
        ci = R["d"]["ci_vs_uniform"]
        print("  " + k.ljust(28) + "".join(_ci(ci[n]).rjust(22) for n in MAIN[1:]))
    A = res["d_across_directions"]
    print("  " + "mean over the directions".ljust(28) + "".join(_ci(A[n]["mean_over_directions"]).rjust(22) for n in MAIN[1:]))
    print("  directions whose 95% CI lies above / below zero: "
          + ", ".join(f"{SHORT[n]} {A[n]['n_directions_ci95_above_zero']}/{A[n]['n_directions_ci95_below_zero']}"
                      for n in MAIN[1:]))
    print("  across directions, difference to uniform: mean / worst / best of the direction means, band = best - worst"
          " (directions > 0; units > 0 / < 0), tightest band first:")
    for n in res["d_across_directions_band_order"]:
        a = A[n]
        print(f"    {n:30s} {a['mean']:+.2f} / {a['worst']:+.2f} / {a['best']:+.2f}  band {a['band']:.2f}"
              f"  ({a['n_directions_pos']}/{a['n_directions']})  units +{a['n_units_pos']} -{a['n_units_neg']} of {a['n_units']}")
    print("\n(e) ORACLE headroom (fitted in-sample on the TARGET labels, analysis only): gain over uniform, pp")
    for k, R in rows:
        e, g = R["e"], R["e"]["gain_over_uniform"]
        print(f"  {k:28s} uniform {e['uniform_f1']['mean']:.2f} | convex(val) {g['convex_val_f1']['mean']:+.2f}"
              f" | oracle convex-NLL {g['oracle_convex_f1']['mean']:+.2f} | oracle grid {g['oracle_grid_f1']['mean']:+.2f}"
              f" | oracle best learner {g['oracle_best_learner_f1']['mean']:+.2f}"
              f" | oracle threshold {g['oracle_threshold_uniform_f1']['mean']:+.2f} (t={e['oracle_threshold_uniform']['mean']:.2f})")
    print("\n(f) diversity of the 5 learners, in-domain -> transfer. Q and the error correlation measure dependence;"
          " disagreement also rises when accuracy falls\n    (implied = disagreement of the transfer accuracies at the"
          " in-domain error correlation)")
    for k, R in rows:
        f = R["f"]

        def s(m, fmt):
            return f"{f[m]['indomain']['mean']:{fmt}}->{f[m]['transfer']['mean']:{fmt}}"
        sh = f["disagreement"]["share_of_rise_implied_by_accuracy"]
        print(f"  {k:28s} Q {s('Q', '.3f')}  err-corr {s('error_corr', '.3f')}  acc% {s('accuracy_mean', '.1f')}"
              f"  dis {s('disagreement', '.3f')} (implied {f['disagreement']['transfer_implied_by_accuracy']['mean']:.3f}"
              f"{f'; {100 * sh:.0f}% of the rise' if sh is not None else ''})  contested% {s('contested', '.1f')}"
              f"  AUROC(uniform) {f['auroc'][UNI]['indomain']['mean']:.3f}->{f['auroc'][UNI]['transfer']['mean']:.3f}")
    G = res["g_lexical"]
    print("\n(g) lexical shift (function-word / affix share: % of tokens; low values indicate stopword removal / stemming)")
    for n, c in G["corpora"].items():
        print(f"  {n:11s} n={c['n_texts']:5d} pos%={c['pos_prior']:5.1f} mean tokens={c['mean_tokens']:5.1f}"
              f" types={c['n_types']:6d} function words={c['function_word_share']:5.1f}% affixed={c['affix_share']:4.1f}%")
    for k, p in G["pairs"].items():
        tag = "" if p["transfer_direction"] else "  (shares instances; not a transfer direction)"
        print(f"  {k:24s} Jaccard={p['jaccard']:5.1f}%  OOV tokens={p['oov_token']:5.1f}%  OOV types={p['oov_type']:5.1f}%"
              f"  prior shift={p['prior_shift']:+5.1f}  near-dup(J>=0.8)={p['near_dup_share_j80']:4.1f}%{tag}")
    print("  " + ", ".join(f"{k}={v:+.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in res["g_lexical_vs_drop"].items()))
    S = res["summary"]
    nd, ns = S["n_directions"], S["n_sources"]
    rr = lambda x: f"{x['min']:+.2f} to {x['max']:+.2f} (mean {x['mean']:+.2f})"
    print(f"\nkey numbers on the common seeds {res['common_seeds']} (transfer: over the {nd} directions; in-domain: over the"
          f" {ns} sources):")
    print(f"  uniform drop {rr(S['uniform_drop'])}; directions above in-domain: {S['n_directions_uniform_above_indomain']}")
    print("  mean drop per learner: " + ", ".join(f"{SHORT[k]} {v['mean']:+.2f}" for k, v in S["learner_drop"].items()))
    print(f"  learner order Spearman {rr(S['spearman_learner_order'])}, > 0 in {S['spearman_learner_order']['n_directions_pos']}"
          f"/{nd}; global validation-best learner stays top in {S['n_directions_best_val_learner_stays_top']}/{nd}; per-fold"
          f" choice top on target in {S['n_units_val_selected_learner_top_on_target']['n']}/"
          f"{S['n_units_val_selected_learner_top_on_target']['of']} units; top on target: "
          + ", ".join(f"{SHORT[k]} {v}" for k, v in S["top_learner_on_target"].items()))
    for c, lab in (("best_single_rule", "Best single (val-selected per fold)"),
                   ("best_val_learner", "[hindsight] global validation-best learner")):
        g = S[f"uniform_minus_{c}"]
        print(f"  uniform - {lab}: in-domain {rr(g['indomain'])} (> 0 in {g['indomain']['n_sources_pos']}/{ns} sources),"
              f" transfer {rr(g['transfer'])} (> 0 in {g['transfer']['n_directions_pos']}/{nd}); shift - in-domain"
              f" {rr(g['shift_minus_indomain'])} (> 0 in {g['shift_minus_indomain']['n_directions_pos']}/{nd})")
    print(f"  positive-class recall drops more than negative-class recall: uniform "
          f"{S['recall_uniform']['n_directions_pos_drop_larger']}/{nd}, Best single "
          f"{S['recall_best_single_rule']['n_directions_pos_drop_larger']}/{nd}")
    for s, e in S["rule_spread_by_set"].items():
        print(f"  spread of the {e['n_rules']} {SET_LABEL[s]}: transfer {rr(e['transfer'])}, < 1 pp in "
              f"{e['transfer']['n_directions_under_1pp']}/{nd}; in-domain {rr(e['indomain'])}; larger under shift in "
              f"{e['n_directions_larger_under_shift']}/{nd}; max |rule - uniform| under shift "
              f"{e['max_abs_diff_to_uniform']['transfer']['max']:.2f}")
    for k, v in S["oracle_gain_over_uniform"].items():
        sd = v["share_of_uniform_drop"]
        print(f"  {k:28s} gain over uniform {rr(v)}; share of the uniform drop {sd['min']:.2f} to {sd['max']:.2f}"
              f" ({sd['n_directions']} directions with a drop)")
    print(f"  oracle convex NLL minus validation-fitted convex NLL on the target, max over units "
          f"{S['oracle_convex_minus_val_nll_max']:+.2e} (<= 0: the oracle is optimal)")
    dv = S["diversity"]
    print(f"  Q in-domain {dv['Q']['indomain']['min']:.3f}-{dv['Q']['indomain']['max']:.3f} -> transfer "
          f"{dv['Q']['transfer']['min']:.3f}-{dv['Q']['transfer']['max']:.3f} (lower in {dv['n_directions_Q_lower_under_shift']}"
          f"/{nd}); error correlation {dv['error_corr']['indomain']['min']:.3f}-{dv['error_corr']['indomain']['max']:.3f} ->"
          f" {dv['error_corr']['transfer']['min']:.3f}-{dv['error_corr']['transfer']['max']:.3f} (lower in "
          f"{dv['n_directions_error_corr_lower_under_shift']}/{nd})")
    sh = dv["share_of_disagreement_rise_implied_by_accuracy"]
    print(f"  disagreement {dv['disagreement']['indomain']['min']:.3f}-{dv['disagreement']['indomain']['max']:.3f} -> "
          f"{dv['disagreement']['transfer']['min']:.3f}-{dv['disagreement']['transfer']['max']:.3f}; share of the rise implied by"
          f" the lower accuracy alone {100 * sh['min']:.0f}-{100 * sh['max']:.0f}% (mean {100 * sh['mean']:.0f}%, "
          f"{sh['n_directions']} directions with a rise; > 100%: the error correlation rose); contested % "
          f"{dv['contested']['indomain']['min']:.1f}-"
          f"{dv['contested']['indomain']['max']:.1f} -> {dv['contested']['transfer']['min']:.1f}-{dv['contested']['transfer']['max']:.1f}")
    print("  dependence: " + S["dependence"]["note"])
    rg = res["regression_seed42"]
    if rg:
        print(f"\nseed-42 EV-Norm -> EV-Public vs final.json: {'PASSED' if rg['passed'] else 'FAILED'} "
              f"({rg['n_compared']} entries, max |diff| {rg['max_abs_diff_pp']:.1e} pp)")


# ============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    seeds = {ds: L.complete_seeds(ds) for ds in L.IN_DOMAIN}
    tg = {(ds, s): xfer_targets(ds, s) for ds in L.IN_DOMAIN for s in seeds[ds]}
    full = {ds: [t for t in ORDER if any(t in tg[(ds, s)] for s in seeds[ds] if s in COMMON_CANDIDATES)]
            for ds in L.IN_DOMAIN}
    common = [s for s in COMMON_CANDIDATES if all(full[ds] and s in seeds[ds] and tg[(ds, s)] == full[ds]
                                                  for ds in L.IN_DOMAIN)]
    assert common, "no seed of 1-4 holds every transfer direction yet"
    jobs = [(ds, s, full[ds]) for ds in L.IN_DOMAIN for s in common]
    s42, t42 = SEED42_ROW
    if 42 in seeds[s42] and t42 in tg[(s42, 42)]:
        jobs.append((s42, 42, [t42]))
    directions = {dkey(ds, t) for ds, s, ts in jobs if s in common for t in ts}
    with Pool(max(1, min(args.workers, len(jobs)))) as pool:
        pending = pool.map_async(run_source_seed, jobs)
        lex = lexical(directions)                       # overlaps with the workers
        units = [u for part in pending.get() for u in part]

    D, UD = OrderedDict(), {}
    for s in L.IN_DOMAIN:
        for t in ORDER:
            U = [u for u in units if u["src"] == s and u["tgt"] == t and u["seed"] in common]
            if U:
                D[dkey(s, t)], UD[dkey(s, t)] = analyse(U, s, t), U
    ci, pooled = shift_intervals(D, UD, N_BOOT)
    for k, R in D.items():
        R["d"]["ci_vs_uniform"] = ci[k]
    U42 = [u for u in units if u["seed"] == 42]
    X = None
    if U42:                                             # the submitted runs, analysed apart from the common seeds
        X = OrderedDict(direction=dkey(s42, t42))
        X.update(analyse(U42, s42, t42))
        X["d"]["ci_vs_uniform"] = shift_intervals({X["direction"]: X}, {X["direction"]: U42}, N_BOOT)[0][X["direction"]]
    ref = OrderedDict()                                 # in-domain CV over every unit of the source used here
    for s in L.IN_DOMAIN:
        V = list({(u["seed"], u["fold"]): u for u in units if u["src"] == s and u["seed"] in common}.values())
        if V:
            ref[L.DS_LAB[s]] = {"seeds": list(dict.fromkeys(u["seed"] for u in V)), "n_units": len(V),
                                "f1": OrderedDict((k, _agg(_col(V, "indomain", k))) for k in V[0]["indomain"])}
    res = OrderedDict(
        script="experiments/rev_transfer.py",
        units_note="macro-F1, recall, accuracy, rates, priors, Jaccard, OOV and shares in percent (differences in pp); Q, "
                   "error correlation, disagreement, AUROC, NLL, thresholds and weights raw. Cells: mean/SD (ddof=1) over "
                   "seeds x folds, sd_seed = SD of the per-seed means.",
        oracle_note="every key starting with oracle_ is fitted in-sample on the TARGET labels (analysis only)",
        comparator_note="'Best single (val-selected)' (best_single_rule) picks the learner per (seed, fold) on that fold's "
                        "validation split: the submission's comparator, used in domain and under shift. best_val_learner "
                        "is one global learner per direction chosen on the pooled validation splits of all its units: a "
                        "hindsight baseline for transfer-side rankings; its in-domain values are partly selected with test "
                        "labels (other folds' validation splits overlap each fold's test portion) and are reference only.",
        seeds_complete={ds: seeds[ds] for ds in L.IN_DOMAIN},
        common_seeds=common,
        seeds_used={k: R["seeds"] for k, R in D.items()},
        seeds_complete_not_common={ds: [s for s in seeds[ds] if s not in common and s != 42 and tg[(ds, s)]]
                                   for ds in L.IN_DOMAIN},
        seeds_without_transfer_posteriors={ds: [s for s in seeds[ds] if not tg[(ds, s)]] for ds in L.IN_DOMAIN},
        learners=LEARNERS, main_rules=MAIN, rule_sets=rule_sets(list(units[0]["transfer"])), grid_steps=GRID_STEPS,
        bootstrap={"n_boot": N_BOOT, "note": "replicates resample the target instances (shared by every fold model, rule "
                   "and seed, and by the directions whose targets are the same posts) and the seeds (shared by the "
                   "directions of one source); a seed's value is the mean over its fold models; percentile CIs, "
                   "two-sided p_boot without multiplicity correction"},
        a_indomain_reference=ref, directions=D)
    res["summary"] = cross_summary(D)
    res["d_across_directions"] = across_directions(D, pooled)
    res["d_across_directions_band_order"] = sorted((k for k in res["d_across_directions"] if k != UNI),
                                                   key=lambda k: (res["d_across_directions"][k]["band"], k))
    res["g_lexical"] = lex
    res["g_lexical_vs_drop"] = lexical_vs_drop(D, lex)
    res["regression_seed42"] = regression_seed42(U42)
    if X:
        X["regression"] = res["regression_seed42"]
    res["seed42_row"] = X
    res = _round(res)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    report(res)
    print(f"\nwrote {os.path.abspath(args.out)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
