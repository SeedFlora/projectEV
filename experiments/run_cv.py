"""Stratified k-fold fine-tuning of Indonesian/multilingual PLMs for EV sentiment.

Saves per-fold validation/test posteriors so that any ensemble rule can be evaluated offline
without retraining. Resumable: finished (dataset, fold, model) runs are skipped, and --time_budget
makes the process exit cleanly between runs so it can be restarted in chunks.
"""
import os, sys, json, time, random, argparse
os.environ.setdefault("HF_HOME", r"D:\project ev\.hf_cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODELS = {
    "indobert-p1": "indobenchmark/indobert-base-p1",
    "indobert-cahya": "cahya/bert-base-indonesian-1.5G",
    "indoroberta-cahya": "cahya/roberta-base-indonesian-1.5G",
    "xlmr-base": "xlm-roberta-base",
    "mbert-cased": "bert-base-multilingual-cased",
}

# Uncased WordPiece checkpoints: transformers>=5 no longer lowercases by default, which sends ~18%
# of capitalised tokens to [UNK]. Force lowercasing for these two.
LOWERCASE_MODELS = {"indobenchmark/indobert-base-p1", "cahya/bert-base-indonesian-1.5G"}

# name -> (path, text column, label column, mapping from raw label to {0: negative, 1: positive})
DATASETS = {
    "evprimary-formal": ("hf_space/data.xlsx", "clean_text_formal", "label", {0: 0, 2: 1}),
    "evprimary-informal": ("hf_space/data.xlsx", "clean_text", "label", {0: 0, 2: 1}),
    "evpublic": ("experiments/data_ext/ryasakbar_2class.xlsx", "Text", "Label", {"Negatif": 0, "Positif": 1}),
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def encode(tok, texts, max_len):
    enc = tok(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict(model, ids, mask, bs=64):
    model.eval(); out = []
    for i in range(0, len(ids), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids[i:i + bs].cuda(), attention_mask=mask[i:i + bs].cuda()).logits
        out.append(torch.softmax(logits.float(), -1).cpu())
    return torch.cat(out).numpy()


def metrics(y, p):
    acc = accuracy_score(y, p)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"acc": float(acc), "prec": float(pr), "rec": float(rc), "f1": float(f1)}


def train_one(model_id, Xtr, ytr, Xva, yva, Xte, args, seed):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id, **({"do_lower_case": True} if model_id in LOWERCASE_MODELS else {}))
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).cuda()
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
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for ids, mask, yb in dl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=yb.cuda())
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += out.loss.item()
        m = metrics(yva, predict(model, va_ids, va_mask).argmax(1))
        m.update(epoch=ep + 1, train_loss=tot / len(dl)); history.append(m)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pva, pte = predict(model, va_ids, va_mask), predict(model, te_ids, te_mask)
    # zero-shot transfer posteriors on the held-out external corpus, if requested
    pex = None
    if args.transfer_texts is not None:
        ex_ids, ex_mask = encode(tok, args.transfer_texts, args.max_len)
        pex = predict(model, ex_ids, ex_mask)
    del model, opt, sch; torch.cuda.empty_cache()
    return pva, pte, pex, history


def load_dataset(name):
    path, text_col, label_col, mapping = DATASETS[name]
    df = pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df = df[[text_col, label_col]].dropna()
    df = df[df[label_col].isin(mapping.keys())]
    X = df[text_col].astype(str).to_numpy()
    y = df[label_col].map(mapping).astype(int).to_numpy()
    keep = np.array([len(str(t).split()) >= 2 for t in X])
    return X[keep], y[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["evprimary-formal", "evpublic", "evprimary-informal"])
    ap.add_argument("--transfer_from", default="evprimary-formal",
                    help="dataset whose runs also score the transfer corpus zero-shot")
    ap.add_argument("--transfer_to", default="evpublic")
    ap.add_argument("--out", default="experiments/probs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--time_budget", type=float, default=1e9, help="exit cleanly once this many seconds elapsed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json.dump({k: v for k, v in vars(args).items() if k != "transfer_texts"},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    Xtr_ext, ytr_ext = load_dataset(args.transfer_to)
    print(f"transfer corpus {args.transfer_to}: n={len(Xtr_ext)} pos={int(ytr_ext.sum())}", flush=True)
    np.savez(os.path.join(args.out, "transfer_labels.npz"), y=ytr_ext)

    t_all = time.time()
    for ds in args.datasets:
        X_all, y_all = load_dataset(ds)
        print(f"== {ds}: n={len(X_all)} label_counts={np.bincount(y_all).tolist()} "
              f"device={torch.cuda.get_device_name(0)}", flush=True)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for fold, (trf_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
            tr_idx, va_idx = train_test_split(trf_idx, test_size=0.1, stratify=y_all[trf_idx], random_state=args.seed)
            for name, mid in MODELS.items():
                odir = os.path.join(args.out, ds, f"fold{fold}"); os.makedirs(odir, exist_ok=True)
                opath = os.path.join(odir, f"{name}.npz")
                if os.path.exists(opath):
                    continue
                if time.time() - t_all > args.time_budget:
                    print(f"TIME BUDGET REACHED after {(time.time()-t_all)/60:.1f} min; exiting cleanly", flush=True)
                    return
                args.transfer_texts = Xtr_ext if ds == args.transfer_from else None
                t0 = time.time()
                pva, pte, pex, hist = train_one(mid, X_all[tr_idx], y_all[tr_idx], X_all[va_idx], y_all[va_idx],
                                                X_all[te_idx], args, args.seed)
                payload = dict(train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx, val_probs=pva, test_probs=pte,
                               y_val=y_all[va_idx], y_test=y_all[te_idx])
                if pex is not None:
                    payload["transfer_probs"] = pex
                tmp = opath + ".tmp.npz"
                np.savez(tmp, **payload); os.replace(tmp, opath)
                json.dump(hist, open(os.path.join(odir, f"{name}_history.json"), "w"), indent=2)
                mv, mt = metrics(y_all[va_idx], pva.argmax(1)), metrics(y_all[te_idx], pte.argmax(1))
                extra = ""
                if pex is not None:
                    me = metrics(ytr_ext, pex.argmax(1)); extra = f" xfer_f1={me['f1']:.4f}"
                print(f"[{ds}] fold{fold} {name:18s} val_f1={mv['f1']:.4f} test_acc={mt['acc']:.4f} "
                      f"test_f1={mt['f1']:.4f}{extra} best_ep={max(hist, key=lambda h: h['f1'])['epoch']} "
                      f"n_tr={len(tr_idx)} n_va={len(va_idx)} n_te={len(te_idx)} time={time.time()-t0:.0f}s", flush=True)
    print(f"ALL DONE total_time={(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
