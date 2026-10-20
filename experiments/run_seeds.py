"""Additional fine-tuning seeds for the revision (reviewer request: multiple fine-tuning seeds).

The fold structure is held fixed at split seed 42, exactly as in run_cv.py, so every seed scores
the same validation and test instances and only the fine-tuning randomness changes (classifier
head initialisation, dropout masks and data order). The training procedure is the one in
run_cv.py, unchanged. In addition to the in-domain posteriors, every run scores all corpora that
share no instances with its training corpus, which gives the bidirectional transfer study, and
records wall-clock cost for the computational-overhead report.

Output: experiments/probs_seeds/seed{S}/{dataset}/fold{k}/{model}.npz (resumable, atomic writes).
"""
import os, sys, re, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_cv  # sets HF_HOME and tokenizer env vars before transformers is imported
from run_cv import MODELS, LOWERCASE_MODELS, encode, metrics, load_dataset, set_seed
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

SPLIT_SEED = 42
TRAIN_SETS = ["evprimary-formal", "evpublic", "evprimary-informal"]
# corpora that share instances must never score each other
SHARED = {"evprimary-formal": {"evprimary-informal"}, "evprimary-informal": {"evprimary-formal"}}
EXTERNAL = {  # binary subsets of two further public EV corpora, used only as transfer targets
    "evtwitter": ("experiments/data_ext/wynernawati_twitter.csv", "content", "label", {"negatif": 0, "positif": 1}),
    "evyoutube": ("experiments/data_ext/cemerson_mobil_listrik.csv", "text_cleaning", "sentimen", {"negatif": 0, "positif": 1}),
}


def _key(t):
    t = re.sub(r"http\S+|@\w+|#", " ", str(t).lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t).split())


def load_external(name, exclude_keys):
    path, tc, lc, mapping = EXTERNAL[name]
    df = pd.read_csv(path)[[tc, lc]].dropna()
    df = df[df[lc].isin(mapping.keys())].copy()
    df["k"] = df[tc].map(_key)
    df = df[~df["k"].isin(exclude_keys)]                      # no text that also occurs in a training corpus
    conflicted = df.groupby("k")[lc].nunique()
    df = df[~df["k"].isin(conflicted[conflicted > 1].index)]  # duplicates with contradictory labels
    df = df.drop_duplicates("k")
    df = df[df[tc].astype(str).str.split().str.len() >= 2]
    return df[tc].astype(str).to_numpy(), df[lc].map(mapping).astype(int).to_numpy()


def build_targets():
    T = {ds: load_dataset(ds) for ds in TRAIN_SETS}
    seen = set()
    for X, _ in T.values():
        seen |= {_key(t) for t in X}
    for name in EXTERNAL:
        T[name] = load_external(name, seen)
    return T


@torch.no_grad()
def predict_timed(model, ids, mask, bs=64):
    model.eval(); out = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(0, len(ids), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids[i:i + bs].cuda(), attention_mask=mask[i:i + bs].cuda()).logits
        out.append(torch.softmax(logits.float(), -1).cpu())
    torch.cuda.synchronize()
    return torch.cat(out).numpy(), time.perf_counter() - t0


def train_one(model_id, Xtr, ytr, Xva, yva, Xte, targets, args, seed):
    """Identical to run_cv.train_one apart from the extra transfer targets and the timing."""
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id, **({"do_lower_case": True} if model_id in LOWERCASE_MODELS else {}))
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).cuda()
    n_params = int(sum(p.numel() for p in model.parameters()))
    tr_ids, tr_mask = encode(tok, Xtr, args.max_len)
    va_ids, va_mask = encode(tok, Xva, args.max_len)
    te_ids, te_mask = encode(tok, Xte, args.max_len)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(TensorDataset(tr_ids, tr_mask, torch.tensor(ytr, dtype=torch.long)),
                    batch_size=args.bs, shuffle=True, generator=g)
    steps = len(dl) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    best_f1, best_state, history = -1.0, None, []
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t_train = time.perf_counter()
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for ids, mask, yb in dl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=yb.cuda())
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += out.loss.item()
        pv_ep, _ = predict_timed(model, va_ids, va_mask)
        m = metrics(yva, pv_ep.argmax(1))
        m.update(epoch=ep + 1, train_loss=tot / len(dl)); history.append(m)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.cuda.synchronize(); train_s = time.perf_counter() - t_train
    peak_mem = int(torch.cuda.max_memory_allocated())
    model.load_state_dict(best_state)
    pva, t_va = predict_timed(model, va_ids, va_mask)
    pte, t_te = predict_timed(model, te_ids, te_mask)
    xfer = {}
    for name, X in targets.items():
        ex_ids, ex_mask = encode(tok, X, args.max_len)
        xfer[name], _ = predict_timed(model, ex_ids, ex_mask)
    del model, opt, sch; torch.cuda.empty_cache()
    cost = dict(train_seconds=train_s, infer_seconds_test=t_te, infer_n_test=len(Xte),
                infer_seconds_val=t_va, n_params=n_params, peak_mem_bytes=peak_mem)
    return pva, pte, xfer, history, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="+", default=TRAIN_SETS)
    ap.add_argument("--out", default="experiments/probs_seeds")
    ap.add_argument("--ref", default="experiments/probs", help="seed-42 posteriors; splits are checked against them")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--time_budget", type=float, default=1e9)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(vars(args), split_seed=SPLIT_SEED), open(os.path.join(args.out, "config.json"), "w"), indent=2)

    T = build_targets()
    for name, (X, y) in T.items():
        np.savez(os.path.join(args.out, f"labels_{name}.npz"), y=y)
        print(f"corpus {name}: n={len(y)} pos={int(y.sum())}", flush=True)

    t_all = time.time()
    for seed in args.seeds:
        for ds in args.datasets:
            X_all, y_all = T[ds]
            targets = {k: v[0] for k, v in T.items() if k != ds and k not in SHARED.get(ds, set())}
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SPLIT_SEED)
            for fold, (trf_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
                tr_idx, va_idx = train_test_split(trf_idx, test_size=0.1, stratify=y_all[trf_idx], random_state=SPLIT_SEED)
                ref = np.load(os.path.join(args.ref, ds, f"fold{fold}", "indobert-p1.npz"))
                assert np.array_equal(ref["val_idx"], va_idx) and np.array_equal(ref["test_idx"], te_idx), \
                    f"split mismatch with the seed-42 runs for {ds} fold {fold}"
                for name, mid in MODELS.items():
                    odir = os.path.join(args.out, f"seed{seed}", ds, f"fold{fold}"); os.makedirs(odir, exist_ok=True)
                    opath = os.path.join(odir, f"{name}.npz")
                    if os.path.exists(opath):
                        continue
                    if time.time() - t_all > args.time_budget:
                        print(f"TIME BUDGET REACHED after {(time.time() - t_all) / 60:.1f} min; exiting cleanly", flush=True)
                        return
                    t0 = time.time()
                    pva, pte, xfer, hist, cost = train_one(mid, X_all[tr_idx], y_all[tr_idx], X_all[va_idx], y_all[va_idx],
                                                           X_all[te_idx], targets, args, seed)
                    payload = dict(train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx, val_probs=pva, test_probs=pte,
                                   y_val=y_all[va_idx], y_test=y_all[te_idx],
                                   **{f"xfer_{k}": v for k, v in xfer.items()},
                                   **{f"cost_{k}": np.array(v) for k, v in cost.items()})
                    tmp = opath + ".tmp.npz"
                    np.savez(tmp, **payload); os.replace(tmp, opath)
                    json.dump(hist, open(os.path.join(odir, f"{name}_history.json"), "w"), indent=2)
                    mv, mt = metrics(y_all[va_idx], pva.argmax(1)), metrics(y_all[te_idx], pte.argmax(1))
                    xs = " ".join(f"{k}={metrics(T[k][1], v.argmax(1))['f1']:.3f}" for k, v in xfer.items())
                    print(f"[seed{seed}][{ds}] fold{fold} {name:18s} val_f1={mv['f1']:.4f} test_f1={mt['f1']:.4f} "
                          f"xfer[{xs}] train={cost['train_seconds']:.0f}s total={time.time() - t0:.0f}s", flush=True)
    print(f"ALL DONE total_time={(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
