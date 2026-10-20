"""Computational overhead of fine-tuning, inference and fusion (reviewer 1, reproducibility).

(a) GPU cost. run_seeds.py stores with every run of seeds 1-4 the wall-clock of fine-tuning (3 epochs,
    including the per-epoch validation pass and the CPU copy of the best checkpoint, hence the per-example
    metric 'finetune_ms_per_example_epoch_incl_val_ckpt'), of scoring the validation and test portions
    (batch 64, length 128, bf16 autocast, CUDA-synchronised), the parameter count and the peak allocated
    GPU memory of the training loop. The laptop was shared with other jobs, so run times are skewed
    upwards by contention ('contention' quantifies it): medians over runs are the headline figures, means
    and SDs are kept for reference. Seed 42 predates this instrumentation, so the GPU time of all 375 runs
    is estimated from the per-(corpus, learner) medians. The ensemble's cost is the sum over the five
    learners, compared with the learner that 'Best single (val-selected)' deploys in the same fold
    (selected on validation, no test labels).
    End-to-end wall-clock per seed comes from the modification times of its 75 posterior files: the gap
    between consecutive files is one run (model loading, tokenisation, training, inference including
    transfer scoring, saving). Gaps longer than 5x the seed's median gap are downtime (a crash followed by
    a relaunch) and are excluded and reported; the runs no kept gap covers (the first run and the run
    closing each excluded gap) count with their own total= from run_seeds.log, or with the mean kept gap
    where the log lacks them, so a merely slow run never turns into downtime. Seeds whose run_seeds.log
    segment holds a 'relaunch' marker are flagged. Seeds
    1-4 also score 3-4 further corpora per run for the revision's transfer study, which seed 42 did only
    for EV-Norm -> EV-Public, and their end-to-end time includes that work. The train=/total= fields of
    run_seeds.log cross-check the stored scalars and the per-seed wall-clock.
(b) CPU cost of every rule in L.ALL_RULES. 'joint' is the per-rule time from L.fit_rules when all rules
    share one context, so the paired bootstrap and the temperatures are charged to the first rule that
    needs them; 'isolated' fits the rule on a fresh context and so includes its own shared quantities;
    context construction (argmax, validation F1 and accuracy per learner) is reported separately.
    Apply time is the fitted callable on the fold's test posteriors. The B = 1000 entry of the B grid
    is the isolated fit of SA-WSV (bootstrap), one measurement reported twice. Fit time grows with the
    validation size, so the citable values are given per validation size (111 posts: EV-Norm and
    EV-Raw; 288: EV-Public).
(c) Bootstrap size B of SA-WSV (bootstrap). Replicate r draws from L.fold_rng(seed, ds, fold, rep=r), so
    replicate 0 is the production stream. Generator.integers fills the (B, n) index array row by row,
    hence the first B resamples of the B = 5000 draw are exactly the B-resample bootstrap: the
    comparison with B = 5000 is nested (common random numbers). Changed test predictions are counted
    with their denominators (fold fits, test predictions, production-stream folds) and every changed
    test post is located. Where the production lambda (B = 1000, replicate 0) is 0 the reliability
    weights are equal, so SA-WSV is entropy-confidence weighting, not uniform soft voting; for those
    folds the test disagreements with both rules and the macro-F1 effect are stored. Test labels only
    score predictions here; every fitted quantity uses the validation split.

Timing protocol: time.perf_counter, garbage collector off, BLAS limited to one thread, the process
restricted to the performance cores of a hybrid CPU while timing, the measurements of a (corpus, seed,
fold) unit interleaved over 5 repetitions and summarised per unit by their median and by their minimum
('best of reps', the statistic least affected by machine load), then by the median and IQR over units;
the Monte-Carlo study runs in worker processes only after all timing is done. Wall-clock timings
(cpu_timing, citable_cpu_timing, runtime_seconds) are the only fields that differ between two runs on
the same seeds; everything else is deterministic.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single-threaded timings; must precede the numpy import
import sys, re, gc, json, time, struct, argparse, platform
from multiprocessing import Pool
import numpy as np, scipy, sklearn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_lib as L

GRID = [100, 200, 500, 1000, 2000, 5000]
REF_B = 5000
N_MC = 20
REPS = 5
MAX_WORKERS = 6
GAP_FACTOR = 5                              # a gap > 5x the seed's median gap between posterior files is downtime
GB = 2 ** 30
GPU = "NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB), bf16 autocast"
FT = "finetune_ms_per_example_epoch_incl_val_ckpt"
ENTROPY = "Entropy confidence"
SEED_LOG = L._p("experiments", "logs", "run_seeds.log")
CV_LOG = L._p("experiments", "logs", "run_cv.stdout.log")
SEED_RE = re.compile(r"^\[seed(\d+)\]\[([\w-]+)\] fold(\d+) (\S+)\s.*\btrain=(\d+)s total=(\d+)s")
SEED_LINE_RE = re.compile(r"^\[seed(\d+)\]\[")
RELAUNCH_RE = re.compile(r"relaunch", re.I)
CV_RE = re.compile(r"^\[(\S+)\] fold(\d+) (\S+)\s.*\btime=(\d+)s")


# ======================================================================== helpers
def _q(v):
    """median and IQR; NaN marks a rule that could not be fitted in a unit (e.g. no training history)"""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not v.size:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"), "n": 0}
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return {"median": float(q50), "q25": float(q25), "q75": float(q75), "n": int(v.size)}


def _ms(v):
    v = np.asarray(v, float)
    q25, q75 = np.percentile(v, [25, 75])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "median": float(np.median(v)),
            "q25": float(q25), "q75": float(q75), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def _rate(n, of):
    """a count with its denominator"""
    return {"n": int(n), "of": int(of), "pct": 100.0 * n / of if of else float("nan")}


def _sig(o, n=6):
    """round floats to n significant digits for a readable JSON (NaN -> null)"""
    if isinstance(o, dict):
        return {str(k): _sig(v, n) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sig(v, n) for v in o]
    if isinstance(o, np.ndarray):
        return _sig(o.tolist(), n)
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return float(f"{o:.{n}g}") if np.isfinite(o) else None
    return o


def system_times():
    """(idle, kernel + user) time summed over all logical processors; Windows only, None elsewhere"""
    try:
        import ctypes
        ft = [ctypes.c_ulonglong() for _ in range(3)]
        ok = ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(f) for f in ft))
        return (ft[0].value, ft[1].value + ft[2].value) if ok else None   # kernel time includes idle time
    except Exception:
        return None


def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()


def performance_cpus():
    """logical processors of the highest efficiency class, i.e. the P-cores of a hybrid CPU (Windows
    CPU-set API); None if all processors are alike or the API is unavailable"""
    try:
        import ctypes
        k32, n = ctypes.windll.kernel32, ctypes.c_ulong(0)
        k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
        buf = (ctypes.c_ubyte * n.value)()
        if not n.value or not k32.GetSystemCpuSetInformation(buf, n.value, ctypes.byref(n), None, 0):
            return None
        raw, off, eff = bytes(buf), 0, {}
        while off + 19 <= len(raw):
            size, kind = struct.unpack_from("<II", raw, off)
            if kind == 0:                       # SYSTEM_CPU_SET_INFORMATION.CpuSet
                group, lp, _, _, _, ec = struct.unpack_from("<HBBBBB", raw, off + 12)
                if group == 0:
                    eff[lp] = ec
            off += size or len(raw)
        return sorted(k for k, v in eff.items() if v == max(eff.values())) if len(set(eff.values())) > 1 else None
    except Exception:
        return None


def set_affinity(cpus):
    """restrict this process to the given logical processors; returns the previous set, None if impossible"""
    try:
        import psutil
        p = psutil.Process(); old = p.cpu_affinity(); p.cpu_affinity(list(cpus))
        return old
    except Exception:
        return None


# ===================================================================== (a) GPU cost
def gpu_runs(cost_seeds):
    """one record per stored fine-tuning run"""
    runs = []
    for ds, seeds in cost_seeds.items():
        N = len(L.target_labels(ds))
        for s in seeds:
            for f in range(L.N_FOLDS):
                d = L.load_fold(ds, f, s)
                best = int(np.argmax(L.macro_f1_batch(d["yv"], d["Pv"].argmax(-1))))   # as r_best_single
                n_xfer = sum(X.shape[1] for X in d["xfer"].values())                  # transfer posts scored per run
                for m, name in enumerate(L.MO):
                    c = d["cost"][name]
                    assert int(c["infer_n_test"]) == len(d["yt"]), (ds, s, f, name)
                    runs.append(dict(ds=ds, seed=s, fold=f, model=name, best=m == best, n_val=len(d["yv"]),
                                     n_train=N - len(d["yv"]) - len(d["yt"]), train_s=c["train_seconds"],
                                     gpu_s=c["train_seconds"] + c["infer_seconds_val"] + c["infer_seconds_test"],
                                     infer_ms=1e3 * c["infer_seconds_test"] / c["infer_n_test"],
                                     infer_val_ms=1e3 * c["infer_seconds_val"] / len(d["yv"]),
                                     params_M=c["n_params"] / 1e6, peak_GB=c["peak_mem_bytes"] / GB, n_xfer=n_xfer))
    return runs


def gpu_summary(runs, cost_seeds, epochs):
    per, ens = {}, {}
    for ds in cost_seeds:
        R = [r for r in runs if r["ds"] == ds]
        if not R:
            per[ds] = ens[ds] = None
            continue
        per[ds] = {}
        for name in L.MO:
            x = [r for r in R if r["model"] == name]
            per[ds][L.MLAB[name]] = {
                "n_runs": len(x), "train_s": _ms([r["train_s"] for r in x]), "gpu_s": _ms([r["gpu_s"] for r in x]),
                "infer_ms_per_inst": _ms([r["infer_ms"] for r in x]),
                "infer_val_ms_per_inst": _ms([r["infer_val_ms"] for r in x]),
                FT: _ms([1e3 * r["train_s"] / (r["n_train"] * epochs) for r in x]),
                "params_M": float(np.mean([r["params_M"] for r in x])),
                "params_constant": bool(np.ptp([r["params_M"] for r in x]) == 0),
                "peak_mem_GB": _ms([r["peak_GB"] for r in x])}
        units = sorted({(r["seed"], r["fold"]) for r in R})
        rat = {"infer": [], "train": [], "params": []}
        tot = {"infer_ms_per_inst": [], "train_s": [], "gpu_s": []}
        sel = {m: 0 for m in L.MO}
        for u in units:
            x = [r for r in R if (r["seed"], r["fold"]) == u]
            b = next(r for r in x if r["best"]); sel[b["model"]] += 1
            ei, et = sum(r["infer_ms"] for r in x), sum(r["train_s"] for r in x)
            tot["infer_ms_per_inst"].append(ei); tot["train_s"].append(et); tot["gpu_s"].append(sum(r["gpu_s"] for r in x))
            rat["infer"].append(ei / b["infer_ms"]); rat["train"].append(et / b["train_s"])
            rat["params"].append(sum(r["params_M"] for r in x) / b["params_M"])
        ens[ds] = {"n_units": len(units), **{k: _ms(v) for k, v in tot.items()},
                   "params_M": float(sum(v["params_M"] for v in per[ds].values())),
                   "weights_GB_fp32": float(sum(v["params_M"] for v in per[ds].values())) * 4e6 / GB,
                   "ratio_vs_val_selected_single": {k: _ms(v) for k, v in rat.items()},
                   "val_selected_counts": {L.MLAB[k]: v for k, v in sel.items()}}
    return per, ens


def contention(runs):
    """fine-tuning time of every run relative to the median of its (corpus, learner) cell; the spread is
    load on the shared machine, not learner cost, which is why medians are the headline figures"""
    if not runs:
        return None
    cell = {}
    for r in runs:
        cell.setdefault((r["ds"], r["model"]), []).append(r["train_s"])
    cell = {k: float(np.median(v)) for k, v in cell.items()}
    rat = np.array([r["train_s"] / cell[(r["ds"], r["model"])] for r in runs])
    fold = {}
    for r, x in zip(runs, rat):
        fold.setdefault((r["seed"], r["ds"], r["fold"]), []).append(x)
    fold = {k: float(np.median(v)) for k, v in fold.items()}           # median over the fold's five learners
    w = max(fold, key=fold.get)
    return {"train_s_over_cell_median": {**_q(rat), "max": float(rat.max())},
            "runs_slower_than_1.25x_cell_median": _rate((rat > 1.25).sum(), rat.size),
            "per_seed_median_ratio": {s: float(np.median([x for r, x in zip(runs, rat) if r["seed"] == s]))
                                      for s in sorted({r["seed"] for r in runs})},
            "slowest_fold": {"seed": w[0], "corpus": L.DS_LAB[w[1]], "fold": w[2], "median_ratio_over_learners": fold[w]}}


def gpu_totals(runs, per, xc):
    n_cell = len(L.SEEDS) * L.N_FOLDS
    ft = xc["file_mtime_span"]
    out = {"runs_measured": len(runs), "gpu_hours_measured": sum(r["gpu_s"] for r in runs) / 3600,
           "train_hours_measured": sum(r["train_s"] for r in runs) / 3600,
           "gpu_hours_by_seed": {s: sum(r["gpu_s"] for r in runs if r["seed"] == s) / 3600
                                 for s in sorted({r["seed"] for r in runs})},
           "wallclock_hours_from_file_times": {
               "seeds": sorted(ft), "hours": sum(v["hours_all_runs"] for v in ft.values()),
               "downtime_excluded_hours": sum(v["downtime_hours"] for v in ft.values()),
               "hours_raw_spans": sum(v["span_hours"] for v in ft.values()),
               "method": f"per seed, gaps between consecutive posterior files, gaps > {GAP_FACTOR}x the seed's median gap "
                         "excluded as downtime, uncovered runs counted with their log total= (else the mean kept gap)"}}
    if any(v is None for v in per.values()):
        out["estimate_all_runs"] = None                     # a corpus without any measured run
        return out
    est = {"n_runs": n_cell * len(per) * len(L.MO),
           "gpu_hours_from_means": sum(n_cell * v["gpu_s"]["mean"] for P in per.values() for v in P.values()) / 3600,
           "gpu_hours_from_medians": sum(n_cell * v["gpu_s"]["median"] for P in per.values() for v in P.values()) / 3600}
    ratio = xc["seed_log"].get("wallclock_over_gpu_time")
    est["wallclock_hours_from_means"] = est["gpu_hours_from_means"] * ratio if ratio else None
    multi = [v["hours_all_runs"] for s, v in ft.items() if s != 42]
    miss = [s for s in L.SEEDS if s not in ft]
    if ft and (multi or not miss):
        est["wallclock_hours_from_file_times"] = {
            "hours": sum(v["hours_all_runs"] for v in ft.values()) + (len(miss) * float(np.median(multi)) if miss else 0.0),
            "seeds_measured": sorted(ft), "seeds_imputed": miss,
            "imputation": "a seed without a complete set of posterior files counts as the median of the measured seeds 1-4"}
    est["note"] = ("headline: gpu_hours_from_medians and wallclock_hours_from_file_times (measured per seed; seeds 1-4 include "
                   "the revision's transfer scoring, seed 42 does not). wallclock_hours_from_means applies the seeds 1-4 "
                   "end-to-end/GPU ratio, which transfer scoring dominates, to all runs including seed 42; reference only")
    out["estimate_all_runs"] = est
    return out


def file_times(seed, log):
    """end-to-end wall-clock of one seed from the modification times of its posterior files. A run that no
    kept gap covers (the first run, the run closing an excluded gap) counts with its logged total= (log:
    (seed, corpus, fold, learner) -> (train, total) seconds), else with the mean kept gap"""
    rec = sorted((os.path.getmtime(os.path.join(L.seed_dir(seed), ds, f"fold{f}", f"{m}.npz")), ds, f, m)
                 for ds in L.IN_DOMAIN for f in range(L.N_FOLDS) for m in L.MO)
    t = np.array([r[0] for r in rec]); g = np.diff(t)
    med = float(np.median(g)); keep = g <= GAP_FACTOR * med; mean_gap = float(g[keep].mean())
    lab = lambda r: f"{L.DS_LAB[r[1]]} fold{r[2]} {L.MLAB[r[3]]}"
    run = lambda r: (log[(seed, *r[1:])][1], "log") if (seed, *r[1:]) in log else (mean_gap, "mean kept gap")
    first, ex = run(rec[0]), np.flatnonzero(~keep)
    closing = [run(rec[i + 1]) for i in ex]
    return {"span_hours": float((t[-1] - t[0]) / 3600), "files": int(t.size),     # raw span: runs 2..n incl. downtime
            "first_file": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t[0])),
            "last_file": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t[-1])),
            "median_gap_s": med, "gaps_kept": int(keep.sum()), "kept_gap_hours": float(g[keep].sum() / 3600),
            "first_run": {"s": first[0], "source": first[1]},
            "hours_all_runs": float((g[keep].sum() + first[0] + sum(c[0] for c in closing)) / 3600),
            "excluded_gaps": [{"after": lab(rec[i]), "before": lab(rec[i + 1]), "gap_s": float(g[i]),
                               "from": time.strftime("%H:%M:%S", time.localtime(t[i])),
                               "to": time.strftime("%H:%M:%S", time.localtime(t[i + 1])),
                               "closing_run_s": c[0], "closing_run_source": c[1]} for i, c in zip(ex, closing)],
            "excluded_gap_hours": float(g[~keep].sum() / 3600),
            "downtime_hours": float((g[~keep].sum() - sum(c[0] for c in closing)) / 3600)}


def log_segments(lines):
    """seed -> its run_seeds.log lines, from just after the previous seed's last record to its own last record"""
    last = {}
    for i, line in enumerate(lines):
        m = SEED_LINE_RE.match(line)
        if m:
            last[int(m.group(1))] = i
    seg, start = {}, 0
    for s in sorted(last, key=last.get):
        seg[s] = lines[start:last[s] + 1]; start = last[s] + 1
    return seg


def crosscheck(runs, mtime_seeds):
    """run_seeds.log train=/total= fields, the seed-42 log excerpt and posterior-file modification times"""
    lines = [x.strip() for x in open(SEED_LOG, encoding="utf-8-sig", errors="replace")] if os.path.exists(SEED_LOG) else []
    log = {}
    for line in lines:
        m = SEED_RE.match(line)
        if m:
            s, ds, f, mod, tr, tot = m.groups()
            log[(int(s), ds, int(f), mod)] = (float(tr), float(tot))
    hit = [(r, log[(r["seed"], r["ds"], r["fold"], r["model"])]) for r in runs
           if (r["seed"], r["ds"], r["fold"], r["model"]) in log]
    diff = [abs(t[0] - r["train_s"]) for r, t in hit]
    seed_log = {"runs_measured": len(runs), "runs_in_log": len(hit),
                "train_consistent": int(sum(x <= 0.5 + 1e-9 for x in diff)),
                "max_abs_train_diff_s": float(max(diff)) if diff else None}
    if hit:
        thr = {}                                    # median test throughput per (corpus, learner): robust to a stalled run
        for r in runs:
            thr.setdefault((r["ds"], r["model"]), []).append(r["infer_ms"])
        thr = {k: float(np.median(v)) for k, v in thr.items()}
        xf = [r["n_xfer"] * thr[(r["ds"], r["model"])] / 1e3 for r, _ in hit]   # GPU time of transfer scoring
        tot, gpu = sum(t[1] for _, t in hit), sum(r["gpu_s"] for r, _ in hit)
        seed_log.update(wallclock_hours_logged=tot / 3600, wallclock_over_gpu_time=tot / gpu,
                        overhead_s_per_run=_ms([t[1] - r["gpu_s"] for r, t in hit]),
                        transfer_posts_per_run=_ms([r["n_xfer"] for r, _ in hit]),
                        transfer_gpu_s_per_run_est=_ms(xf),
                        wallclock_over_gpu_time_without_transfer_gpu_est=(tot - sum(xf)) / gpu)
    cv = []
    if os.path.exists(CV_LOG):
        for line in open(CV_LOG, encoding="utf-8-sig", errors="replace"):
            m = CV_RE.match(line.strip())
            if m:
                cv.append({"corpus_column": m.group(1), "fold": int(m.group(2)), "model": m.group(3),
                           "total_s": float(m.group(4))})
    seg, spans = log_segments(lines), {}
    for s in mtime_seeds:
        spans[s] = file_times(s, log)
        sg = seg.get(s)                                 # seed 42 ran before run_seeds.py existed: no segment
        tot = [v[1] for k, v in log.items() if k[0] == s]
        spans[s].update(
            relaunch_in_log=None if sg is None else any(RELAUNCH_RE.search(x) for x in sg),
            relaunch_markers=[x for x in sg if RELAUNCH_RE.search(x)] if sg else [],
            log_runs=len(tot), log_total_hours=sum(tot) / 3600 if tot else None,
            log_total_hours_all_runs=sum(tot) / 3600 * spans[s]["files"] / len(tot) if tot else None,
            transfer_posts_per_run=float(np.mean([sum(X.shape[1] for X in L.load_fold(ds, f, s)["xfer"].values())
                                                  for ds in L.IN_DOMAIN for f in range(L.N_FOLDS)])))
    return {"seed_log": seed_log, "seed42_log_excerpt": cv, "file_mtime_span": spans,
            "note": "log total = end-to-end per run (model loading, tokenisation, training, inference incl. transfer "
                    "scoring, saving). file_mtime_span: hours_all_runs is the per-seed wall-clock from the gaps between "
                    f"consecutive posterior files, gaps > {GAP_FACTOR}x the median gap excluded as downtime, the runs no "
                    "kept gap covers counted with their log total= (else the mean kept gap); span_hours is the raw "
                    "first-to-last span (runs 2..n, including downtime). transfer_gpu_s_per_run_est = transfer posts x "
                    "the median test ms per post of the run's (corpus, learner) cell (GPU only, tokenisation excluded)"}


# ================================================================== (b) CPU timing
def time_unit(ds, seed, fold, reps, grid):
    d = L.load_fold(ds, fold, seed)
    Pv, yv, Pt, hist = d["Pv"], d["yv"], d["Pt"], d["hist"]
    rng = lambda: L.fold_rng(seed, ds, fold)
    T = {k: {"joint": [], "iso": [], "apply": []} for k in L.ALL_RULES}
    init, total, bfit, shared = [], [], {B: [] for B in grid}, {}
    gc.collect(); gc.disable()
    try:
        for _ in range(reps):
            t, t0 = {}, time.perf_counter()
            fitted, _ = L.fit_rules(Pv, yv, rng(), hist=hist, timings=t)
            total.append(time.perf_counter() - t0)
            for k in L.ALL_RULES:
                T[k]["joint"].append(t[k])
            for k, fit in L.ALL_RULES.items():
                g = rng(); t0 = time.perf_counter()
                c = L.Ctx(Pv, yv, g, hist=hist); t1 = time.perf_counter()
                fit(c); t2 = time.perf_counter()
                init.append(t1 - t0); T[k]["iso"].append(t2 - t1)
                shared[k] = [n for n, a in (("bootstrap", "_idx"), ("temperatures", "_T")) if getattr(c, a) is not None]
            for k, f in fitted.items():
                t0 = time.perf_counter(); f(Pt); T[k]["apply"].append(time.perf_counter() - t0)
            for B in grid:
                if B == L.N_BOOT:               # the isolated fit of the primary rule is this very computation
                    bfit[B].append(T[L.PRIMARY]["iso"][-1]); continue
                c = L.Ctx(Pv, yv, rng(), n_boot=B); t0 = time.perf_counter()
                L.r_sawsv_boot(c); bfit[B].append(time.perf_counter() - t0)
    finally:
        gc.enable()
    med = lambda v: 1e3 * float(np.median(v)) if v else float("nan")
    lo = lambda v: 1e3 * float(np.min(v)) if v else float("nan")
    return {"ds": ds, "seed": seed, "fold": fold, "n_val": len(yv), "n_test": len(d["yt"]),
            "ctx_init_ms": med(init), "fit_all_joint_ms": med(total), "fit_all_joint_ms_min": lo(total), "shared": shared,
            "rules": {k: {"fit_joint_ms": med(v["joint"]), "fit_isolated_ms": med(v["iso"]), "fit_isolated_ms_min": lo(v["iso"]),
                          "apply_ms": med(v["apply"]), "apply_ms_min": lo(v["apply"])} for k, v in T.items()},
            "B_fit_ms": {B: med(v) for B, v in bfit.items()}, "B_fit_ms_min": {B: lo(v) for B, v in bfit.items()}}


def _by_nval(U, val):
    """median [IQR] over the units of each validation size; corpora with equal n_val are pooled"""
    return {n: {"corpora": [L.DS_LAB[ds] for ds in L.IN_DOMAIN if any(u["ds"] == ds and u["n_val"] == n for u in U)],
                **_q([val(u) for u in U if u["n_val"] == n])} for n in sorted({u["n_val"] for u in U})}


def timing_summary(U, grid):
    rules = {}
    for k in L.ALL_RULES:
        x = [u["rules"][k] for u in U]
        us = [1e3 * r["apply_ms"] / u["n_test"] for r, u in zip(x, U)]
        rules[k] = {"shared": U[0]["shared"][k],
                    **{m: _q([r[m] for r in x]) for m in ("fit_joint_ms", "fit_isolated_ms", "fit_isolated_ms_min")},
                    "fit_isolated_ms_min_by_n_val": _by_nval(U, lambda u, k=k: u["rules"][k]["fit_isolated_ms_min"]),
                    "apply_ms_per_fold": _q([r["apply_ms"] for r in x]), "apply_us_per_inst": _q(us),
                    "apply_us_per_inst_min": _q([1e3 * r["apply_ms_min"] / u["n_test"] for r, u in zip(x, U)]),
                    "per_corpus_median": {ds: {
                        "fit_isolated_ms": float(np.nanmedian([r["fit_isolated_ms"] for r, u in zip(x, U) if u["ds"] == ds])),
                        "apply_us_per_inst": float(np.nanmedian([v for v, u in zip(us, U) if u["ds"] == ds]))}
                        for ds in L.IN_DOMAIN if any(u["ds"] == ds for u in U)}}
    bfit = {B: {**_q([u["B_fit_ms"][B] for u in U]),
                "per_corpus_median": {ds: float(np.median([u["B_fit_ms"][B] for u in U if u["ds"] == ds]))
                                      for ds in L.IN_DOMAIN if any(u["ds"] == ds for u in U)},
                "best_of_reps": _q([u["B_fit_ms_min"][B] for u in U]),
                "best_of_reps_by_n_val": _by_nval(U, lambda u, B=B: u["B_fit_ms_min"][B])} for B in grid}
    return {"units": [[u["ds"], u["seed"], u["fold"]] for u in U],
            "ctx_init_ms": _q([u["ctx_init_ms"] for u in U]), "fit_all_rules_joint_ms": _q([u["fit_all_joint_ms"] for u in U]),
            "fit_all_rules_joint_ms_min_by_n_val": _by_nval(U, lambda u: u["fit_all_joint_ms_min"]),
            "rules": rules, "bootstrap_B_fit_ms": bfit,
            "per_unit": [{k: u[k] for k in ("ds", "seed", "fold", "n_val", "ctx_init_ms", "fit_all_joint_ms", "fit_all_joint_ms_min",
                                            "rules", "B_fit_ms", "B_fit_ms_min")} for u in U]}


# ========================================================== (c) Monte-Carlo error of B
def mc_unit(args):
    ds, seed, fold, grid, n_mc = args
    d, ref = L.load_fold(ds, fold, seed), grid.index(REF_B)
    lam = np.zeros((len(grid), n_mc)); W = np.zeros((len(grid), n_mc, len(L.MO)))
    pred = np.zeros((len(grid), n_mc, len(d["yt"])), np.int8); prob = np.zeros(pred.shape)
    for r in range(n_mc):
        cs = [L.Ctx(d["Pv"], d["yv"], L.fold_rng(seed, ds, fold, rep=r), n_boot=B) for B in grid]
        for i, c in enumerate(cs):
            f = L.r_sawsv_boot(c)
            lam[i, r], W[i, r] = c.meta["sawsv_boot"]["lam"], c.meta["sawsv_boot"]["w"]
            pred[i, r], prob[i, r] = f(d["Pt"])
        assert all(np.array_equal(c.idx, cs[ref].idx[:c.n_boot]) for c in cs), "bootstrap resamples are not nested"
    dp = np.abs(prob - prob[ref][None])
    flip = pred != pred[ref][None]                                   # (B, bootstrap seed, test post)
    loc = [{"pos": int(j), "test_idx": int(d["test_idx"][j]), "y": int(d["yt"][j]),
            "p_ref": [float(prob[ref, :, j].min()), float(prob[ref, :, j].max())],
            "margin_ref_min": float(np.abs(prob[ref, :, j] - 0.5).min()),
            "flips_by_B": {B: int(flip[i, :, j].sum()) for i, B in enumerate(grid) if i != ref}}
           for j in np.flatnonzero(flip.any((0, 1)))]
    c = L.Ctx(d["Pv"], d["yv"], L.fold_rng(seed, ds, fold))         # the two comparators need no fitted quantity
    uni, ent = L.ALL_RULES[L.BASELINE](c)(d["Pt"])[0], L.ALL_RULES[ENTROPY](c)(d["Pt"])[0]
    return {"ds": ds, "seed": seed, "fold": fold, "yt": d["yt"], "lam": lam, "w": W, "pred": pred,
            "dprob_max": dp.max((1, 2)), "margin_min": float(np.abs(prob[ref] - 0.5).min()), "flip_locations": loc,
            "pred_uniform": uni.astype(np.int8), "pred_entropy": ent.astype(np.int8)}


def _mc_block(U, i, ref):
    if not U:
        return None
    flips = sum(int((u["pred"][i] != u["pred"][ref]).sum()) for u in U)
    cells = sum(u["pred"][i].size for u in U)
    anyf = np.concatenate([(u["pred"][i] != u["pred"][ref]).any(1) for u in U])
    vary = sum(int((u["pred"][i] != u["pred"][i][:1]).any(0).sum()) for u in U)
    posts = sum(len(u["yt"]) for u in U)
    prod = sum(int((u["pred"][i, 0] != u["pred"][ref, 0]).any()) for u in U)
    dfold = [100 * abs(L.macro_f1(u["yt"], u["pred"][i, r]) - L.macro_f1(u["yt"], u["pred"][ref, r]))
             for u in U for r in range(u["pred"].shape[1])]
    dpool = []
    for key in sorted({(u["ds"], u["seed"]) for u in U}):
        G = sorted((u for u in U if (u["ds"], u["seed"]) == key), key=lambda u: u["fold"])
        y = np.concatenate([u["yt"] for u in G])
        for r in range(G[0]["pred"].shape[1]):
            a = np.concatenate([u["pred"][i, r] for u in G]); b = np.concatenate([u["pred"][ref, r] for u in G])
            dpool.append(100 * abs(L.macro_f1(y, a) - L.macro_f1(y, b)))
    sd = [u["lam"][i].std(ddof=1) for u in U]
    act = [u["lam"][i].std(ddof=1) for u in U if u["lam"][ref].mean() > 0]   # shrinkage active at B = REF_B
    return {"lambda_mc_sd": {**_q(sd), "max": float(np.max(sd))},
            "lambda_mc_sd_active": {**_q(act), "max": float(np.max(act))} if act else None,
            "lambda_mean": float(np.mean([u["lam"][i].mean() for u in U])),
            "lambda_zero_share": float(np.mean([(u["lam"][i] == 0).mean() for u in U])),
            "weight_mc_sd_max": _q([u["w"][i].std(0, ddof=1).max() for u in U]),
            "flip_pct_vs_ref": 100 * flips / cells, "fits_with_any_flip_pct": 100 * float(anyf.mean()),
            "inst_varying_across_seeds_pct": 100 * vary / posts,
            # the same quantities as counts with their denominators
            "fits_with_flip": _rate(anyf.sum(), anyf.size),             # fold fits (fold x bootstrap seed)
            "flipped_predictions": _rate(flips, cells),                   # test predictions (fit x test post)
            "production_stream_folds_with_flip": _rate(prod, len(U)),     # replicate 0, the stream of the results
            "test_posts_varying_across_bootstrap_seeds": _rate(vary, posts),
            "abs_dF1_fold_pp": {"mean": float(np.mean(dfold)), "max": float(np.max(dfold)), "n": len(dfold)},
            "abs_dF1_pooled_pp": {"mean": float(np.mean(dpool)), "max": float(np.max(dpool)), "n": len(dpool)},
            "max_abs_dprob_vs_ref": {"median_over_folds": float(np.median([u["dprob_max"][i] for u in U])),
                                     "max": float(np.max([u["dprob_max"][i] for u in U]))}}


def lambda_zero(U, grid):
    """folds whose production lambda (B = N_BOOT, replicate 0) is 0: equal reliability weights, so SA-WSV is
    entropy-confidence weighting; disagreements with uniform soft voting and entropy confidence, F1 effect"""
    if L.N_BOOT not in grid:
        return None
    ip = grid.index(L.N_BOOT)
    Z = [u for u in U if u["lam"][ip, 0] == 0]
    rows, pooled = [], []
    for u in Z:
        y, p = u["yt"], u["pred"][ip, 0]
        row = {"corpus": L.DS_LAB[u["ds"]], "seed": u["seed"], "fold": u["fold"], "n_test": len(y),
               "lambda_zero_share_over_B_and_bootstrap_seeds": float((u["lam"] == 0).mean())}
        for name, q in ((L.BASELINE, u["pred_uniform"]), (ENTROPY, u["pred_entropy"])):
            row[name] = {"disagreements": int((p != q).sum()),
                         "sawsv_right_other_wrong": int(((p == y) & (q != y)).sum()),
                         "other_right_sawsv_wrong": int(((p != y) & (q == y)).sum()),
                         "dF1_fold_pp": 100 * (L.macro_f1(y, p) - L.macro_f1(y, q))}
        rows.append(row)
    for key in sorted({(u["ds"], u["seed"]) for u in Z}):
        G = sorted((u for u in U if (u["ds"], u["seed"]) == key), key=lambda u: u["fold"])
        y = np.concatenate([u["yt"] for u in G])
        a = np.concatenate([u["pred"][ip, 0] for u in G])
        b = np.concatenate([u["pred_uniform"] if u["lam"][ip, 0] == 0 else u["pred"][ip, 0] for u in G])
        pooled.append({"corpus": L.DS_LAB[key[0]], "seed": key[1], "folds": [u["fold"] for u in G],
                       "lambda_zero_folds": [u["fold"] for u in G if u["lam"][ip, 0] == 0],
                       "dF1_pooled_pp_vs_uniform_on_lambda_zero_folds": 100 * (L.macro_f1(y, a) - L.macro_f1(y, b))})
    return {"B": L.N_BOOT, "replicate": 0, "folds_with_lambda_zero": _rate(len(Z), len(U)), "folds": rows, "pooled": pooled,
            "note": "dF1 = SA-WSV minus the other rule (fold macro-F1, pp); pooled: SA-WSV as run minus SA-WSV with its "
                    "lambda = 0 folds replaced by uniform soft voting, over the folds of that corpus x seed"}


def mc_summary(U, grid, n_mc):
    ref = grid.index(REF_B)
    out = {"reference_B": REF_B, "n_bootstrap_seeds": n_mc, "n_units": len(U),
           "per_B": {B: _mc_block(U, i, ref) for i, B in enumerate(grid)},
           "per_corpus": {ds: {B: _mc_block([u for u in U if u["ds"] == ds], i, ref) for i, B in enumerate(grid)}
                          for ds in L.IN_DOMAIN if any(u["ds"] == ds for u in U)}}
    # context: genuine fold-to-fold variation of lambda (converged, B = REF_B, mean over bootstrap seeds)
    groups = [[u["lam"][ref].mean() for u in U if (u["ds"], u["seed"]) == k] for k in sorted({(u["ds"], u["seed"]) for u in U})]
    btw = [np.std(g, ddof=1) for g in groups if len(g) > 1]
    out["lambda_between_fold_sd"] = _q(btw) if btw else None
    # a prediction can only flip if the change of P(positive) exceeds the instance's distance from 0.5
    out["decision_margin_min_at_ref"] = {"median_over_folds": float(np.median([u["margin_min"] for u in U])),
                                         "min": float(np.min([u["margin_min"] for u in U]))}
    out["flip_locations"] = [{"corpus": L.DS_LAB[u["ds"]], "seed": u["seed"], "fold": u["fold"], **x}
                             for u in U for x in u["flip_locations"]]
    out["production_lambda_B1000"] = [[u["ds"], u["seed"], u["fold"], float(u["lam"][grid.index(1000), 0])] for u in U] \
        if 1000 in grid else None
    out["lambda_zero_folds"] = lambda_zero(U, grid)
    return out


# ========================================================================= report
def _fusion_overhead(t, g):
    """SA-WSV apply cost against the ensemble's GPU forward passes (medians)"""
    R = t["rules"]
    ens = [g["ensemble"][ds]["infer_ms_per_inst"]["median"] for ds in g["ensemble"] if g["ensemble"][ds]]
    if not ens:
        return None
    a, u, e = R[L.PRIMARY]["apply_us_per_inst"]["median"], R[L.BASELINE]["apply_us_per_inst"]["median"], float(np.mean(ens))
    return {"primary_apply_us_per_inst": a, "uniform_apply_us_per_inst": u, "increment_over_uniform_us_per_inst": a - u,
            "ensemble_forward_ms_per_inst": e, "primary_share_of_ensemble_forward_pct": 100 * 1e-3 * a / e,
            "increment_share_of_ensemble_forward_pct": 100 * 1e-3 * (a - u) / e,
            "wording": f"fusing with SA-WSV costs about {a:.1f} us per post, about {100 * 1e-3 * a / e:.2f}% of the "
                       f"ensemble's forward passes",
            "basis": "median single-thread CPU time of the fitted rule applied to a fold's test posteriors, against the "
                     "five learners' GPU forward passes (bf16, batch 64, throughput); not an increment over uniform "
                     "soft voting, which is given separately"}


def report(J):
    g, t, b = J["gpu_cost"], J["cpu_timing"], J["bootstrap_B"]
    print(f"\n(a) GPU cost per fine-tuning run, median [IQR] over runs (mean +- SD, inflated by contention); {GPU}")
    print(f"    seeds with cost scalars: {J['seeds_used']['gpu_cost']}")
    print(f"{'corpus':10s} {'learner':17s} {'train s median [IQR]':>22s} {'(mean +- SD)':>14s} {'test ms/inst':>12s}"
          f" {'ms/ex/ep*':>9s} {'params M':>9s} {'peak GB':>8s}")
    for ds, P in g["per_learner"].items():
        if P is None:
            print(f"{L.DS_LAB[ds]:10s} (no measured runs yet)"); continue
        for m in L.MO:
            r = P[L.MLAB[m]]; s = r["train_s"]
            print(f"{L.DS_LAB[ds]:10s} {L.MLAB[m]:17s} {s['median']:7.1f} [{s['q25']:5.1f}-{s['q75']:5.1f}]"
                  f" ({s['mean']:5.1f} +-{s['sd']:5.1f}) {r['infer_ms_per_inst']['median']:12.3f} {r[FT]['median']:9.2f}"
                  f" {r['params_M']:9.1f} {r['peak_mem_GB']['max']:8.2f}")
        e = g["ensemble"][ds]; q = e["ratio_vs_val_selected_single"]; s = e["train_s"]
        print(f"{'':10s} {'ensemble (sum)':17s} {s['median']:7.1f} [{s['q25']:5.1f}-{s['q75']:5.1f}]"
              f" ({s['mean']:5.1f} +-{s['sd']:5.1f}) {e['infer_ms_per_inst']['median']:12.3f} {'':9s} {e['params_M']:9.1f}"
              f"   x{q['infer']['median']:.2f} inference, x{q['train']['median']:.2f} training,"
              f" x{q['params']['median']:.2f} parameters vs val-selected single (medians)")
    print("    * fine-tuning wall-clock per training example and epoch, including the per-epoch validation pass and "
          "the checkpoint copy; peak GB = maximum over runs")
    T = g["totals"]
    print(f"    measured: {T['runs_measured']} runs, {T['gpu_hours_measured']:.2f} GPU h (training {T['train_hours_measured']:.2f} h); "
          f"by seed: " + ", ".join(f"{s}: {h:.2f} h" for s, h in T["gpu_hours_by_seed"].items()))
    E = T["estimate_all_runs"]
    if E:
        print(f"    GPU time of all {E['n_runs']} runs: {E['gpu_hours_from_medians']:.2f} h from the per-(corpus, learner) "
              f"medians (headline; {E['gpu_hours_from_means']:.2f} h from means)")
        w = E.get("wallclock_hours_from_file_times")
        if w:
            imp = f", seed(s) {w['seeds_imputed']} imputed" if w["seeds_imputed"] else ""
            print(f"    end-to-end wall-clock of all {E['n_runs']} runs: {w['hours']:.2f} h (seeds {w['seeds_measured']} measured"
                  f"{imp}; seeds 1-4 include transfer scoring)")
    c = J["log_crosscheck"]; sl = c["seed_log"]
    print(f"    per-seed end-to-end wall-clock from posterior-file gaps (gaps > {GAP_FACTOR}x the median excluded as downtime):")
    for s, v in c["file_mtime_span"].items():
        rl = {None: "no run_seeds.log segment", True: "RELAUNCH in log", False: "no relaunch in log"}[v["relaunch_in_log"]]
        lt = f"; log total= {v['log_total_hours_all_runs']:.3f} h ({v['log_runs']} runs, scaled)" if v["log_total_hours_all_runs"] else ""
        print(f"      seed {s}: {v['hours_all_runs']:.3f} h for {v['files']} runs ({v['gaps_kept']} gaps kept, median gap "
              f"{v['median_gap_s']:.0f} s, raw span {v['span_hours']:.3f} h); downtime excluded {v['downtime_hours']:.3f} h; {rl}{lt}; "
              f"{v['transfer_posts_per_run']:.0f} transfer posts scored per run")
        for x in v["excluded_gaps"]:
            print(f"        excluded gap {x['from']}-{x['to']} ({x['gap_s']:.0f} s) between {x['after']} and {x['before']}; "
                  f"the closing run counts {x['closing_run_s']:.0f} s ({x['closing_run_source']})")
    if sl.get("wallclock_over_gpu_time"):
        print(f"    log cross-check: {sl['runs_in_log']}/{sl['runs_measured']} runs in run_seeds.log, train= consistent for "
              f"{sl['train_consistent']}; end-to-end/GPU time = {sl['wallclock_over_gpu_time']:.2f} "
              f"({sl['wallclock_over_gpu_time_without_transfer_gpu_est']:.2f} without the estimated GPU time of transfer "
              f"scoring, median {sl['transfer_gpu_s_per_run_est']['median']:.1f} s per run)")
    k = g.get("contention")
    if k:
        x = k["train_s_over_cell_median"]; w = k["slowest_fold"]
        print(f"    contention: run time / median of its (corpus, learner) cell has IQR {x['q25']:.2f}-{x['q75']:.2f}, max "
              f"{x['max']:.2f}; {k['runs_slower_than_1.25x_cell_median']['n']}/{k['runs_slower_than_1.25x_cell_median']['of']} runs "
              f"> 1.25x; slowest fold: seed {w['seed']} {w['corpus']} fold {w['fold']} ({w['median_ratio_over_learners']:.2f}x)")

    reps, aff = J["settings"]["reps"], t.get("affinity_logical_processors")
    print(f"\n(b) CPU cost of fusion rules, median [IQR] over {len(t['units'])} corpus x seed x fold units "
          f"(seeds {J['seeds_used']['cpu']}); per unit the median of {reps} reps; 1 thread"
          f"{f' on logical processors {aff[0]}-{aff[-1]} (performance cores)' if aff else ''}, {J['hardware']['cpu']}")
    print(f"{'rule':30s} {'fit joint ms':>20s} {'fit isolated ms':>21s} {'apply ms/fold':>14s} {'apply us/inst':>20s}  shared")
    fq = lambda q, f="{:.2f}": f"{f.format(q['median'])} [{f.format(q['q25'])}-{f.format(q['q75'])}]"
    for k, r in t["rules"].items():
        print(f"{k:30s} {fq(r['fit_joint_ms']):>20s} {fq(r['fit_isolated_ms']):>21s} {r['apply_ms_per_fold']['median']:14.3f}"
              f" {fq(r['apply_us_per_inst'], '{:.3f}'):>20s}  {'+'.join(r['shared']) or '-'}")
    print(f"{'(context construction)':30s} {fq(t['ctx_init_ms'], '{:.3f}'):>20s}")
    print(f"{'(all rules, one context)':30s} {fq(t['fit_all_rules_joint_ms']):>20s}")
    if t.get("system_cpu_utilisation_pct") is not None:
        print(f"    mean system-wide CPU utilisation while timing: {t['system_cpu_utilisation_pct']:.0f}% "
              f"of {os.cpu_count()} logical processors (this process: one)")
    key = [k for k in (L.PRIMARY, "SA-WSV (binomial)", "Uniform + TS", "Stacking (LR)") if k in t["rules"]]
    for n, v in t["fit_all_rules_joint_ms_min_by_n_val"].items():
        print(f"    best of {reps} reps, n_val = {n} ({', '.join(v['corpora'])}; {v['n']} units): "
              + ", ".join(f"{k} {t['rules'][k]['fit_isolated_ms_min_by_n_val'][n]['median']:.2f} ms" for k in key)
              + f", all rules on one context {v['median']:.1f} ms")
    fo = _fusion_overhead(t, g)
    if fo:
        print(f"    fusion at deployment: SA-WSV {fo['primary_apply_us_per_inst']:.2f} us per post, uniform soft voting "
              f"{fo['uniform_apply_us_per_inst']:.2f} us (one CPU thread), against {fo['ensemble_forward_ms_per_inst']:.2f} ms "
              f"for the five GPU forward passes: SA-WSV is {fo['primary_share_of_ensemble_forward_pct']:.3f}% of them")

    print(f"\n(c) SA-WSV (bootstrap) vs bootstrap size B: {b['n_bootstrap_seeds']} bootstrap seeds per fold, "
          f"{b['n_units']} folds, reference B = {b['reference_B']} (nested resamples); counts are n/N")
    nv = list(t["fit_all_rules_joint_ms_min_by_n_val"])
    print(f"{'B':>5s} {'fit ms, n_val ' + '/'.join(map(str, nv)):>22s} {'MC SD lambda med/max':>21s} {'mean lambda':>11s}"
          f" {'fits w/ flip':>13s} {'flipped preds':>15s} {'prod. folds':>11s} {'|dF1| fold max':>14s} {'pooled max pp':>13s}")
    fr = lambda x: f"{x['n']}/{x['of']}"
    for B, r in b["per_B"].items():
        tb = t["bootstrap_B_fit_ms"][B]["best_of_reps_by_n_val"]
        ms = "/".join("%.1f" % tb[n]["median"] for n in nv)
        print(f"{B:5d} {ms:>22s}"
              f" {r['lambda_mc_sd']['median']:10.4f} / {r['lambda_mc_sd']['max']:.4f} {r['lambda_mean']:11.3f}"
              f" {fr(r['fits_with_flip']):>13s} {fr(r['flipped_predictions']):>15s} {fr(r['production_stream_folds_with_flip']):>11s}"
              f" {r['abs_dF1_fold_pp']['max']:14.3f} {r['abs_dF1_pooled_pp']['max']:13.3f}")
    print("    fit ms: best of reps, median over units; max |change of fused P(positive)| vs B = %d: " % b["reference_B"] + ", ".join(
        f"B={B}: {r['max_abs_dprob_vs_ref']['max']:.1e}" for B, r in b["per_B"].items() if B != b["reference_B"])
        + f"; smallest |P(positive) - 0.5| on any test post: {b['decision_margin_min_at_ref']['min']:.1e}")
    for x in b["flip_locations"]:
        print(f"    changed post: {x['corpus']} seed {x['seed']} fold {x['fold']} test post {x['pos']} (corpus index {x['test_idx']}, "
              f"label {x['y']}), P(positive) at B = {b['reference_B']} in [{x['p_ref'][0]:.4f}, {x['p_ref'][1]:.4f}]; changed in "
              + ", ".join(f"{n}/{b['n_bootstrap_seeds']} seeds at B={B}" for B, n in x["flips_by_B"].items() if n))
    a = b["per_B"][1000]["lambda_mc_sd_active"] if 1000 in b["per_B"] else None
    if a:
        print(f"    B = 1000, folds with lambda > 0 ({a['n']}): MC SD of lambda median {a['median']:.4f}, max {a['max']:.4f}")
    if b.get("lambda_between_fold_sd"):
        print(f"    for scale: SD of the converged lambda across folds, median over corpus x seed = "
              f"{b['lambda_between_fold_sd']['median']:.3f}")
    z = b.get("lambda_zero_folds")
    if z:
        f = z["folds_with_lambda_zero"]
        print(f"    lambda = 0 at production settings (B = {z['B']}, replicate 0) on {f['n']}/{f['of']} folds; there SA-WSV is "
              f"entropy-confidence weighting:")
        for r in z["folds"]:
            u, e = r[L.BASELINE], r[ENTROPY]
            print(f"      {r['corpus']} seed {r['seed']} fold {r['fold']}: {u['disagreements']}/{r['n_test']} test posts differ from "
                  f"uniform soft voting (SA-WSV right on {u['sawsv_right_other_wrong']}, uniform on {u['other_right_sawsv_wrong']}; "
                  f"fold dF1 {u['dF1_fold_pp']:+.2f} pp), {e['disagreements']} from entropy confidence; lambda = 0 in "
                  f"{100 * r['lambda_zero_share_over_B_and_bootstrap_seeds']:.0f}% of the B x bootstrap-seed fits")
        for p in z["pooled"]:
            print(f"      {p['corpus']} seed {p['seed']} pooled: dF1 {p['dF1_pooled_pp_vs_uniform_on_lambda_zero_folds']:+.3f} pp "
                  f"against uniform soft voting on folds {p['lambda_zero_folds']}")


def citable(J):
    g, b = J["gpu_cost"], J["bootstrap_B"]
    P = {ds: v for ds, v in g["per_learner"].items() if v}
    c = {"headline_statistic": "medians over runs; means and SDs are inflated by contention on the shared machine "
                               "(gpu_cost.contention) and their keys (*_mean*) are kept for reference only"}
    if P:
        lab = list(P[next(iter(P))])
        tr = {(ds, m): P[ds][m]["train_s"]["median"] for ds in P for m in P[ds]}
        c["train_s_per_run_median_range"] = [min(tr.values()), max(tr.values())]
        c["train_s_per_run_median_by_corpus"] = {L.DS_LAB[ds]: [min(v for (d, _), v in tr.items() if d == ds),
                                                               max(v for (d, _), v in tr.items() if d == ds)] for ds in P}
        inf = [P[ds][m]["infer_ms_per_inst"]["median"] for ds in P for m in P[ds]]
        c["infer_ms_per_inst_median_range"] = [min(inf), max(inf)]
        c[f"{FT}_median"] = {m: [min(P[ds][m][FT]["median"] for ds in P), max(P[ds][m][FT]["median"] for ds in P)] for m in lab}
        c["ensemble_infer_ms_per_inst_median"] = {L.DS_LAB[ds]: g["ensemble"][ds]["infer_ms_per_inst"]["median"] for ds in P}
        for k in ("infer", "train", "params"):
            c[f"ensemble_{k}_ratio_vs_val_selected_median"] = {
                L.DS_LAB[ds]: g["ensemble"][ds]["ratio_vs_val_selected_single"][k]["median"] for ds in P}
        tr = {(ds, m): P[ds][m]["train_s"]["mean"] for ds in P for m in P[ds]}
        c["train_s_per_run_mean_range"] = [min(tr.values()), max(tr.values())]
        c["train_s_per_run_mean_by_corpus"] = {L.DS_LAB[ds]: [min(v for (d, _), v in tr.items() if d == ds),
                                                             max(v for (d, _), v in tr.items() if d == ds)] for ds in P}
        inf = [P[ds][m]["infer_ms_per_inst"]["mean"] for ds in P for m in P[ds]]
        c["infer_ms_per_inst_mean_range"] = [min(inf), max(inf)]
        c["params_M"] = {m: v["params_M"] for m, v in P[next(iter(P))].items()}
        c["ensemble_params_M"] = g["ensemble"][next(iter(P))]["params_M"]
        c["ensemble_weights_GB_fp32"] = g["ensemble"][next(iter(P))]["weights_GB_fp32"]
        c["peak_mem_GB_max"] = {m: max(P[ds][m]["peak_mem_GB"]["max"] for ds in P) for m in lab}
        c["ensemble_infer_ms_per_inst"] = {L.DS_LAB[ds]: g["ensemble"][ds]["infer_ms_per_inst"]["mean"] for ds in P}
        for k in ("infer", "train", "params"):
            c[f"ensemble_{k}_ratio_vs_val_selected"] = {L.DS_LAB[ds]: g["ensemble"][ds]["ratio_vs_val_selected_single"][k]["mean"]
                                                        for ds in P}
    if g.get("contention"):
        c["contention"] = g["contention"]
    T = g["totals"]
    c["gpu_hours_measured"] = {"runs": T["runs_measured"], "hours": T["gpu_hours_measured"]}
    if T["estimate_all_runs"]:
        c["gpu_hours_all_runs_estimate"] = T["estimate_all_runs"]
        c["gpu_hours_all_runs_from_medians"] = T["estimate_all_runs"]["gpu_hours_from_medians"]
        if T["estimate_all_runs"].get("wallclock_hours_from_file_times"):
            c["wallclock_hours_all_runs"] = T["estimate_all_runs"]["wallclock_hours_from_file_times"]
    F = J["log_crosscheck"]["file_mtime_span"]
    c["wallclock_hours_per_seed"] = {s: v["hours_all_runs"] for s, v in F.items()}
    c["downtime_excluded_hours_per_seed"] = {s: v["downtime_hours"] for s, v in F.items()}
    c["relaunch_in_log_per_seed"] = {s: v["relaunch_in_log"] for s, v in F.items()}
    c["posterior_file_span_hours_per_seed"] = {s: v["span_hours"] for s, v in F.items()}   # raw: runs 2..n incl. downtime
    c["transfer_posts_per_run_per_seed"] = {s: v["transfer_posts_per_run"] for s, v in F.items()}
    c["wallclock_hours_from_file_times"] = T["wallclock_hours_from_file_times"]
    sl = J["log_crosscheck"]["seed_log"]
    if sl.get("wallclock_over_gpu_time"):
        c["end_to_end_over_gpu_time"] = {"all": sl["wallclock_over_gpu_time"],
                                         "without_transfer_gpu_est": sl["wallclock_over_gpu_time_without_transfer_gpu_est"]}
    for B in (100, 1000, REF_B):
        if B in b["per_B"]:
            r = b["per_B"][B]
            c[f"B{B}_lambda_mc_sd_median"] = r["lambda_mc_sd"]["median"]
            c[f"B{B}_lambda_mc_sd_max"] = r["lambda_mc_sd"]["max"]
            if r["lambda_mc_sd_active"]:
                c[f"B{B}_lambda_mc_sd_median_folds_with_lambda_gt0"] = r["lambda_mc_sd_active"]["median"]
            if B != REF_B:
                c[f"B{B}_flip_pct_vs_B{REF_B}"] = r["flip_pct_vs_ref"]
                c[f"B{B}_abs_dF1_pooled_pp_max"] = r["abs_dF1_pooled_pp"]["max"]
                c[f"B{B}_max_abs_dprob_vs_B{REF_B}"] = r["max_abs_dprob_vs_ref"]["max"]
    c[f"changes_vs_B{REF_B}"] = {B: {k: r[k] for k in ("fits_with_flip", "flipped_predictions", "production_stream_folds_with_flip",
                                                       "test_posts_varying_across_bootstrap_seeds")}
                                 | {"abs_dF1_fold_pp_max": r["abs_dF1_fold_pp"]["max"],
                                    "abs_dF1_pooled_pp_max": r["abs_dF1_pooled_pp"]["max"]}
                                 for B, r in b["per_B"].items() if B != REF_B}
    c["changed_test_posts"] = b["flip_locations"]
    c["decision_margin_min"] = b["decision_margin_min_at_ref"]["min"]
    if b.get("lambda_zero_folds"):
        z = b["lambda_zero_folds"]
        c["lambda_zero_folds"] = {"folds_with_lambda_zero": z["folds_with_lambda_zero"],
                                  "folds": [{k: r[k] for k in ("corpus", "seed", "fold", "n_test")}
                                            | {"vs_uniform": r[L.BASELINE], "vs_entropy_confidence": r[ENTROPY]} for r in z["folds"]],
                                  "pooled": z["pooled"]}
    c["notes"] = [
        "posterior_file_span_hours_per_seed is the raw first-to-last file span (runs 2..n, including downtime); cite "
        "wallclock_hours_per_seed, which excludes downtime_excluded_hours_per_seed",
        f"the B*_pct keys are shares; changes_vs_B{REF_B} gives counts with their denominators: fold fits = folds x "
        "bootstrap seeds, predictions = fits x test posts, production stream = replicate 0 (behind the reported results)",
        "end-to-end wall-clock includes transfer scoring (transfer_posts_per_run_per_seed), which seeds 1-4 did for 3-4 "
        "corpora and seed 42 only from EV-Norm to EV-Public",
        "CPU timings are in citable_cpu_timing (wall-clock, not deterministic)"]
    return c


def citable_timing(J):
    t, g = J["cpu_timing"], J["gpu_cost"]
    R = t["rules"]
    c = {f"per_corpus_median_{k}": R[k]["per_corpus_median"] for k in (L.PRIMARY, "SA-WSV (binomial)", L.BASELINE,
                                                                        "Stacking (LR)", "SA-WSV + TS") if k in R}
    c["fit_isolated_ms_median"] = {k: R[k]["fit_isolated_ms"]["median"] for k in (L.PRIMARY, "SA-WSV (binomial)", "Stacking (LR)")}
    c["apply_us_per_inst_median"] = {k: R[k]["apply_us_per_inst"]["median"] for k in (L.PRIMARY, L.BASELINE, "Stacking (LR)")}
    c["apply_us_per_inst_max_over_rules"] = max(r["apply_us_per_inst"]["median"] for r in R.values())
    c["fit_all_rules_joint_ms_median"] = t["fit_all_rules_joint_ms"]["median"]
    c["B_fit_ms_median"] = {B: v["median"] for B, v in t["bootstrap_B_fit_ms"].items()}
    key = [k for k in (L.PRIMARY, "SA-WSV (binomial)", "Uniform + TS", "Stacking (LR)") if k in R]
    c["fit_ms_best_of_reps_by_n_val"] = {
        n: {"corpora": v["corpora"], "units": v["n"], **{k: R[k]["fit_isolated_ms_min_by_n_val"][n]["median"] for k in key},
            "all rules, one context": v["median"]} for n, v in t["fit_all_rules_joint_ms_min_by_n_val"].items()}
    c["B_fit_ms_best_of_reps_by_n_val"] = {n: {B: v["best_of_reps_by_n_val"][n]["median"] for B, v in t["bootstrap_B_fit_ms"].items()}
                                           for n in t["fit_all_rules_joint_ms_min_by_n_val"]}
    fo = _fusion_overhead(t, g)
    if fo:   # fusion apply cost as a share of the five forward passes it fuses
        c["primary_apply_share_of_ensemble_gpu_inference_pct"] = fo["primary_share_of_ensemble_forward_pct"]
        c["fusion_overhead"] = fo
    c["system_cpu_utilisation_pct"] = t.get("system_cpu_utilisation_pct")
    c["note"] = (f"cite the *_best_of_reps_by_n_val values, rounded: single-thread timings on a shared machine reproduce only "
                 f"to tens of percent; the B = {L.N_BOOT} grid entry and the isolated SA-WSV (bootstrap) fit are one measurement; "
                 "fit time is linear in B")
    return c


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one seed, two folds, 2 reps, 5 bootstrap seeds (development)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--seeds", type=int, nargs="+", default=L.SEEDS, help="restrict to these seeds (default: all complete)")
    ap.add_argument("--out", default=None, help="output JSON (default: experiments/results/rev/rev_cost[_quick].json)")
    args = ap.parse_args()
    t_start = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    assert L.ALL_RULES[L.PRIMARY] is L.r_sawsv_boot and REF_B in GRID and L.N_BOOT in GRID
    reps, n_mc = (2, 5) if args.quick else (REPS, N_MC)
    folds = [0, 1] if args.quick else list(range(L.N_FOLDS))
    cfg = json.load(open(L._p("experiments", "probs_seeds", "config.json")))

    complete, skipped = {}, {}
    for ds in L.IN_DOMAIN:                      # fixed for the whole run
        complete[ds] = []
        for s in L.complete_seeds(ds, args.seeds):
            try:
                for f in range(L.N_FOLDS):
                    L.load_fold(ds, f, s)
                complete[ds].append(s)
            except RuntimeError as e:           # a history file still being written by run_seeds.py
                skipped.setdefault(ds, []).append(s)
                print(f"  seed {s} skipped on {ds}: {e}", flush=True)
    cpu_seeds = {ds: v[:1] if args.quick else v for ds, v in complete.items()}
    cost_seeds = {ds: [s for s in v if s != 42 and L.load_fold(ds, 0, s)["cost"]] for ds, v in complete.items()}
    if args.quick:
        cost_seeds = {ds: v[:1] for ds, v in cost_seeds.items()}
    units = [(ds, s, f) for ds in L.IN_DOMAIN for s in cpu_seeds[ds] for f in folds]
    print(f"complete seeds {complete}; {len(units)} units for (b)/(c); quick={args.quick}", flush=True)

    # (a) stored GPU cost
    runs = gpu_runs(cost_seeds)
    per, ens = gpu_summary(runs, cost_seeds, cfg["epochs"])
    xc = crosscheck(runs, sorted({s for v in complete.values() for s in v if all(s in complete[d] for d in L.IN_DOMAIN)}))

    # (b) + (c) timing, sequential on the performance cores, before any worker process exists
    pcpu = performance_cpus()
    prev = set_affinity(pcpu) if pcpu else None
    try:
        time_unit(*units[0], reps=1, grid=GRID)                  # warm-up (imports, caches); discarded
        U, st0 = [], system_times()
        for i, u in enumerate(units):
            U.append(time_unit(*u, reps=reps, grid=GRID))
            if (i + 1) % 10 == 0 or i + 1 == len(units):
                print(f"  timed {i + 1}/{len(units)} units ({time.perf_counter() - t_start:.0f}s)", flush=True)
        st1 = system_times()
    finally:
        if prev is not None:
            set_affinity(prev)                  # the Monte-Carlo workers inherit the affinity
    load = 100 * (1 - (st1[0] - st0[0]) / (st1[1] - st0[1])) if st0 and st1 and st1[1] > st0[1] else None

    # (c) Monte-Carlo error, deterministic, in parallel
    tasks = [(ds, s, f, GRID, n_mc) for ds, s, f in units]
    with Pool(max(1, min(args.workers, MAX_WORKERS, len(tasks)))) as pool:
        MC = pool.map(mc_unit, tasks, chunksize=1)
    print(f"  Monte-Carlo study done ({time.perf_counter() - t_start:.0f}s)", flush=True)

    J = {"script": "experiments/rev_cost.py", "quick": args.quick,
         "seeds_used": {"gpu_cost": cost_seeds, "cpu": cpu_seeds, "folds": folds, "per_seed_wallclock": sorted(xc["file_mtime_span"]),
                        "skipped_incomplete": skipped},
         "hardware": {"gpu": GPU, "cpu": cpu_name(), "cpu_threads": 1, "python": platform.python_version(),
                      "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
         "settings": {"reps": reps, "B_grid": GRID, "reference_B": REF_B, "n_bootstrap_seeds": n_mc,
                      "n_boot_production": L.N_BOOT, "fine_tuning": {k: cfg[k] for k in ("epochs", "bs", "lr", "max_len")},
                      "inference_batch": 64, "memory_unit": "GB = 2^30 bytes (torch.cuda.max_memory_allocated)",
                      "downtime_gap_factor": GAP_FACTOR},
         "gpu_cost": {"per_learner": per, "ensemble": ens, "contention": contention(runs), "totals": gpu_totals(runs, per, xc)},
         "log_crosscheck": xc,
         "bootstrap_B": mc_summary(MC, GRID, n_mc),
         "cpu_timing": {"system_cpu_utilisation_pct": load, "affinity_logical_processors": pcpu if prev is not None else None,
                        **timing_summary(U, GRID)}}
    report(J)
    J["citable"] = citable(J)
    J["citable_cpu_timing"] = citable_timing(J)
    J["runtime_seconds"] = time.perf_counter() - t_start
    J["nondeterministic_keys"] = ["cpu_timing", "citable_cpu_timing", "runtime_seconds"]
    out = args.out or L._p("experiments", "results", "rev", "rev_cost_quick.json" if args.quick else "rev_cost.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(_sig(J), fh, indent=1)
    try:
        shown = os.path.relpath(out, L.ROOT)
    except ValueError:                          # another drive
        shown = out
    print(f"\nwrote {shown} in {J['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
