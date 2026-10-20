"""Annotation ambiguity and re-annotation reliability on the primary corpus.

Labelling protocol (confirmed by the authors): every post was labelled by a first annotator
(label_old) and then reviewed by a second annotator, who confirmed or changed it (label, used
everywhere). The review saw the first label, so the agreement between passes compares first-pass with
adjudicated labels and is likely an upper bound on independent inter-annotator agreement. 40 posts were
left unlabelled in the first pass; they cannot have been revised, so every revision rate below is
computed on the 1,342 posts labelled in both passes, except the explicitly labelled replication of the
submitted all-posts figures.

Out-of-fold test posteriors map to spreadsheet rows through test_idx: the training loader keeps all
1,382 rows in file order, which is asserted by reproducing every stored y_test from the file. Both
renderings use the same set of seeds (those complete for both), so K, the number of predictions per
post (5 learners x seeds), is equal across them. Pooled all-wrong / all-correct groups become stricter
as K grows, so like-for-like comparisons with the submission use per-seed K = 5 groups, and a K-free
summary (AUROC of the share of wrong predictions for predicting revision) is added. The ensemble
posterior is the uniform average over learners and seeds. The confident-learning thresholds use the
final labels, as Northcutt et al. (2021) prescribe; the ambiguity score 1 - p(final label) depends on the
adjudicated label, the ensemble entropy does not.
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rev_lib as L

PRIMARY = ["evprimary-formal", "evprimary-informal"]
TEXT_COL = {"evprimary-formal": "clean_text_formal", "evprimary-informal": "clean_text"}
B = 5000


def load_sheet():
    df = pd.read_excel(os.path.join(L.ROOT, "hf_space", "data.xlsx"))
    y = df["label"].map({0: 0, 2: 1}).to_numpy()
    old = df["label_old"].map({0.0: 0, 2.0: 1})
    both = old.notna().to_numpy()
    old = old.fillna(-1).astype(int).to_numpy()
    revised = both & (old != y)
    return df, y, old, both, revised


def kappa(a, b):
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in (0, 1)))
    return (po - pe) / (1 - pe), po, pe


def oof(ds, seeds, n):
    """(S, n, 5, 2) out-of-fold posteriors of the five learners, rows in spreadsheet order"""
    out = []
    for s in seeds:
        P = np.full((n, len(L.MO), 2), np.nan)
        for fold in range(L.N_FOLDS):
            d = L.load_fold(ds, fold, s)
            P[d["test_idx"]] = d["Pt"].transpose(1, 0, 2)
        assert not np.isnan(P).any()
        out.append(P)
    return np.stack(out)


def check_alignment(ds, seeds, y):
    for s in seeds:
        for fold in range(L.N_FOLDS):
            d = L.load_fold(ds, fold, s)
            assert np.array_equal(y[d["test_idx"]], d["yt"]), "stored y_test does not match the spreadsheet rows"


def katz(a, n1, c, n2):
    rr = (a / n1) / (c / n2)
    se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
    return float(rr), [float(rr * math.exp(-1.96 * se)), float(rr * math.exp(1.96 * se))]


def groups(k, K, revised, mask):
    """all-wrong / all-correct revision rates and Katz risk ratio on the posts in mask"""
    aw, ar = (k == 0) & mask, (k == K) & mask
    a, n1, c, n2 = int(revised[aw].sum()), int(aw.sum()), int(revised[ar].sum()), int(ar.sum())
    rr, ci = katz(a, n1, c, n2) if a and c else (float("nan"), [float("nan")] * 2)
    return {"K": int(K), "n_allwrong": n1, "n_allcorrect": n2, "rev_allwrong": a / n1 if n1 else float("nan"),
            "rev_allcorrect": c / n2 if n2 else float("nan"), "rr": rr, "rr_ci95_katz": ci}


def confident_learning(p, y):
    """Northcutt et al. (2021): per-class thresholds, confident joint, flagged label issues"""
    t = np.array([p[y == j, j].mean() for j in (0, 1)])
    above = p >= t[None]
    yhat = np.where(above.any(1), np.where(above, p, -1).argmax(1), -1)
    flagged = (yhat >= 0) & (yhat != y)
    C = np.array([[np.sum((y == i) & (yhat == j)) for j in (0, 1)] for i in (0, 1)], float)
    n_i = np.bincount(y, minlength=2)
    Cc = C / np.maximum(C.sum(1, keepdims=True), 1) * n_i[:, None]
    return flagged, t, float((Cc.sum() - np.trace(Cc)) / len(y))


def paired_auroc(scores, label, rng, n_boot=2000):
    """AUROCs of several scores with CIs from shared resamples, plus CIs of their differences to the first"""
    names = list(scores)
    point = {k: L.auroc(scores[k], label) for k in names}
    reps = {k: [] for k in names}
    n = len(label)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if label[i].any() and (~label[i]).any():
            for k in names:
                reps[k].append(L.auroc(scores[k][i], label[i]))
    out = {k: {"auroc": point[k], "ci95": [float(np.percentile(reps[k], 2.5)), float(np.percentile(reps[k], 97.5))]}
           for k in names}
    base = names[0]
    for k in names[1:]:
        d = np.asarray(reps[base]) - np.asarray(reps[k])
        out[k][f"diff_{base}_minus_this"] = {"diff": point[base] - point[k],
                                             "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out


def analyse(ds, seeds, y, old, both, revised, df, rng):
    P = oof(ds, seeds, len(y))
    pens = P.mean(axis=(0, 2))
    correct = P.argmax(-1) == y[None, :, None]                # (S, n, 5)
    K = correct.shape[0] * correct.shape[2]
    k = correct.sum(axis=(0, 2))
    words = df[TEXT_COL[ds]].astype(str).str.split().str.len().to_numpy()
    aw, ar = k == 0, k == K
    contested = ~aw & ~ar
    per_seed = {str(s): groups(correct[i].sum(-1), len(L.MO), revised, both) for i, s in enumerate(seeds)}
    rr_vals = [v["rr"] for v in per_seed.values()]
    flagged, thr, noise = confident_learning(pens, y)
    pred = pens.argmax(1)
    lab = revised[both]
    scores = {"ambiguity_1_minus_p_final": (1 - pens[np.arange(len(y)), y])[both],
              "ensemble_entropy": (1 - L.conf_entropy(pens[None])[0])[both],
              "share_wrong_k_over_K": (1 - k / K)[both]}
    seed42 = None
    if 42 in seeds:
        k42 = correct[seeds.index(42)].sum(-1)
        allposts = np.ones(len(y), bool)
        seed42 = {"all_posts_as_submitted": groups(k42, len(L.MO), revised, allposts),
                  "labelled_posts": groups(k42, len(L.MO), revised, both)}
    return {
        "seeds": seeds, "K_pooled": int(K),
        "groups_pooled": {name: {"n": int((g & both).sum()), "share_of_labelled": float((g & both).sum() / both.sum()),
                                 "revised_rate": float(revised[g & both].mean()), "mean_words": float(words[g].mean())}
                          for name, g in (("all_wrong", aw), ("contested", contested), ("all_correct", ar))},
        "rr_pooled": groups(k, K, revised, both),
        "rr_per_seed_K5": per_seed,
        "rr_per_seed_K5_summary": {"mean": float(np.mean(rr_vals)), "min": float(np.min(rr_vals)), "max": float(np.max(rr_vals))},
        "seed42_replication": seed42,
        "ensemble_accuracy": {"revised": float(np.mean(pred[revised] == y[revised])),
                              "unrevised_labelled": float(np.mean(pred[both & ~revised] == y[both & ~revised])),
                              "first_pass_missing": float(np.mean(pred[~both] == y[~both])), "all": float(np.mean(pred == y))},
        "auroc_revision": {**paired_auroc(scores, lab, rng), "n_items": int(both.sum()), "n_revised": int(lab.sum()),
                           "note": "CIs resample posts and condition on the trained ensemble; the ambiguity score uses the adjudicated label"},
        "confident_learning": {
            "thresholds": thr.tolist(), "n_flagged": int(flagged.sum()), "flag_rate": float(flagged.mean()),
            "n_flagged_labelled": int((flagged & both).sum()),
            "revised_rate_flagged": float(revised[flagged & both].mean()),
            "revised_rate_unflagged": float(revised[~flagged & both].mean()),
            "precision_for_revised": float(revised[flagged & both].mean()),
            "recall_of_revised": float(flagged[revised].mean()),
            "estimated_label_issue_rate": noise,
            "note": "the issue rate estimates residual noise in the final labels, not the revision rate"},
        "side_taking_revised": {"n_revised": int(revised.sum()),
                                "agrees_with_first_pass": float(np.mean(pred[revised] == old[revised])),
                                "per_seed_first_pass": [float(np.mean(P[i].mean(1).argmax(-1)[revised] == old[revised]))
                                                        for i in range(len(seeds))]},
        "examples_allwrong_revised": [{"text": str(df[TEXT_COL[ds]].iloc[i])[:180], "label_first": int(old[i]),
                                       "label_final": int(y[i]), "p_ens_pos": float(pens[i, 1])}
                                      for i in np.flatnonzero(aw & revised)[:5]],
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df, y, old, both, revised = load_sheet()
    rng = np.random.default_rng(0)
    yb, ob = y[both], old[both]
    kap, po, pe = kappa(ob, yb)
    kb = []
    for _ in range(B):
        i = rng.integers(0, len(yb), len(yb))
        kb.append(kappa(ob[i], yb[i])[0])
    n_neg, n_pos = int((ob == 0).sum()), int((ob == 1).sum())
    reliab = {"n_both": int(both.sum()), "n_first_missing": int((~both).sum()), "agreement": po, "kappa": kap,
              "kappa_ci95": [float(np.percentile(kb, 2.5)), float(np.percentile(kb, 97.5))], "pe": pe, "pabak": 2 * po - 1,
              "kappa_equals_pabak_because_pe_is_0.5": abs(pe - 0.5) < 1e-3,
              "transition": {f"first {a} -> final {b}": int(np.sum((ob == a) & (yb == b))) for a in (0, 1) for b in (0, 1)},
              "revised_total": int(revised.sum()), "revised_share_labelled": float(revised[both].mean()),
              "revised_share_all_posts_as_submitted": float(revised.mean()),
              "neg_to_pos": int(np.sum((ob == 0) & (yb == 1))), "pos_to_neg": int(np.sum((ob == 1) & (yb == 0))),
              "flip_rate_first_neg": float(np.mean(yb[ob == 0] == 1)), "flip_rate_first_pos": float(np.mean(yb[ob == 1] == 0)),
              "n_first_neg": n_neg, "n_first_pos": n_pos,
              "note": "agreement between first-pass and non-blind adjudicated labels; likely an upper bound on independent agreement"}
    seeds = sorted(set.intersection(*[set(L.complete_seeds(ds)) for ds in PRIMARY]), key=L.SEEDS.index)
    out = {"reliability": reliab, "seeds_common": seeds, "corpora": {}}
    for ds in PRIMARY:
        check_alignment(ds, seeds, y)
        out["corpora"][L.DS_LAB[ds]] = analyse(ds, seeds, y, old, both, revised, df, np.random.default_rng(1))
    os.makedirs(os.path.join(L.ROOT, "experiments", "results", "rev"), exist_ok=True)
    json.dump(out, open(os.path.join(L.ROOT, "experiments", "results", "rev", "rev_annotation.json"), "w",
                        encoding="utf-8"), indent=1, ensure_ascii=False)
    r = reliab
    if len(seeds) < len(L.SEEDS):
        print(f"PRELIMINARY: seeds {seeds} complete for both renderings")
    print(f"inter-pass (first-pass vs non-blind adjudicated labels; likely an upper bound on independent agreement): "
          f"n={r['n_both']} agreement={100*r['agreement']:.1f}% kappa={r['kappa']:.3f} "
          f"[{r['kappa_ci95'][0]:.3f},{r['kappa_ci95'][1]:.3f}] (= PABAK, since pe={r['pe']:.3f}); revised {r['revised_total']} = "
          f"{100*r['revised_share_labelled']:.1f}% of labelled posts ({100*r['revised_share_all_posts_as_submitted']:.1f}% of all); "
          f"flip rate {100*r['flip_rate_first_neg']:.1f}% of first-pass negatives, {100*r['flip_rate_first_pos']:.1f}% of positives")
    for labn, R in out["corpora"].items():
        g = R["groups_pooled"]
        print(f"\n== {labn} seeds={R['seeds']} (pooled K={R['K_pooled']})")
        for name, v in g.items():
            print(f"  {name:12s} n={v['n']:5d} share={100*v['share_of_labelled']:5.1f}% revised={100*v['revised_rate']:5.1f}% words={v['mean_words']:.1f}")
        rp = R["rr_pooled"]
        print(f"  pooled (K={rp['K']}) RR = {rp['rr']:.1f} [{rp['rr_ci95_katz'][0]:.1f}, {rp['rr_ci95_katz'][1]:.1f}]")
        print("  per-seed K=5 RR: " + ", ".join(f"s{s}: {v['rr']:.1f} [{v['rr_ci95_katz'][0]:.1f},{v['rr_ci95_katz'][1]:.1f}]"
                                            for s, v in R["rr_per_seed_K5"].items())
              + f"  (mean {R['rr_per_seed_K5_summary']['mean']:.1f})")
        if R["seed42_replication"]:
            s = R["seed42_replication"]
            print(f"  seed-42 replication: all posts {s['all_posts_as_submitted']['rr']:.2f}, labelled posts {s['labelled_posts']['rr']:.2f}")
        ea = R["ensemble_accuracy"]
        print(f"  ensemble accuracy: revised {100*ea['revised']:.1f}%  unrevised {100*ea['unrevised_labelled']:.1f}%")
        a = R["auroc_revision"]
        for kname in ("ambiguity_1_minus_p_final", "ensemble_entropy", "share_wrong_k_over_K"):
            v = a[kname]
            extra = v.get("diff_ambiguity_1_minus_p_final_minus_this")
            print(f"  AUROC {kname:28s} {v['auroc']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]"
                  + (f"  ambiguity minus this {extra['diff']:+.3f} [{extra['ci95'][0]:+.3f},{extra['ci95'][1]:+.3f}]" if extra else ""))
        c = R["confident_learning"]
        print(f"  confident learning: flagged {c['n_flagged']} ({100*c['flag_rate']:.1f}%); among labelled flagged {c['n_flagged_labelled']}: "
              f"revised {100*c['revised_rate_flagged']:.1f}% vs unflagged {100*c['revised_rate_unflagged']:.1f}%; "
              f"recall of revisions {100*c['recall_of_revised']:.1f}%; estimated residual issue rate {100*c['estimated_label_issue_rate']:.1f}%")
        s = R["side_taking_revised"]
        print(f"  revised posts: ensemble agrees with FIRST-pass label on {100*s['agrees_with_first_pass']:.1f}% "
              f"(per seed {[round(100*x, 1) for x in s['per_seed_first_pass']]})")
        for e in R["examples_allwrong_revised"]:
            print(f"    - [{e['label_first']}->{e['label_final']}, p_pos={e['p_ens_pos']:.2f}] {e['text'][:110]}")


if __name__ == "__main__":
    main()
