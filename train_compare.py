import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODELS = [
    "indobenchmark/indobert-base-p1",
    "cahya/bert-base-indonesian-1.5G",
    "cahya/roberta-base-indonesian-1.5G",
    "xlm-roberta-base",
    "bert-base-multilingual-cased",
    "distilbert-base-multilingual-cased",
    "bert-base-uncased",
    "distilbert-base-uncased",
]


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)


def build_label_map(labels):
    unique = sorted(set(labels))
    if set(unique) == {0, 2}:
        label_id_map = {0: 0, 2: 1}
        id2label = {0: "negatif", 1: "positif"}
    elif set(unique) == {0, 1}:
        label_id_map = {0: 0, 1: 1}
        id2label = {0: "negatif", 1: "positif"}
    else:
        label_id_map = {val: idx for idx, val in enumerate(unique)}
        id2label = {idx: f"label_{idx}" for idx in range(len(unique))}
    label2id = {v: k for k, v in id2label.items()}
    return label_id_map, id2label, label2id


def compute_metrics_builder(num_labels: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        avg = "binary" if num_labels == 2 else "macro"
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=avg, zero_division=0
        )
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="ev_data (without Neutral)_formal_balanced.xlsx",
        help="Path to balanced dataset (xlsx).",
    )
    parser.add_argument("--text-col", default="clean_text_formal")
    parser.add_argument("--label-col", default="label")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=5,
        help="Maximum number of models to train successfully.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_excel(data_path)
    df = df[[args.text_col, args.label_col]].dropna()
    df[args.text_col] = df[args.text_col].astype(str)

    label_id_map, id2label, label2id = build_label_map(df[args.label_col].tolist())
    df["label_id"] = df[args.label_col].map(label_id_map)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df["label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=args.seed,
        stratify=temp_df["label_id"],
    )

    train_ds = Dataset.from_pandas(train_df[[args.text_col, "label_id"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[[args.text_col, "label_id"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[[args.text_col, "label_id"]], preserve_index=False)

    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")
    test_ds = test_ds.rename_column("label_id", "labels")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best_model_name = None
    best_f1 = -1.0

    use_fp16 = torch.cuda.is_available()

    success_count = 0
    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if success_count >= args.max_models:
            break
        run_dir = output_dir / sanitize_model_name(model_name)
        run_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        def tokenize(batch):
            return tokenizer(
                batch[args.text_col],
                padding=False,
                truncation=True,
                max_length=args.max_length,
            )

        tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=[args.text_col])
        tokenized_val = val_ds.map(tokenize, batched=True, remove_columns=[args.text_col])
        tokenized_test = test_ds.map(tokenize, batched=True, remove_columns=[args.text_col])

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        num_labels = len(set(df["label_id"].tolist()))
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, use_safetensors=True
            )
        except Exception as exc:
            print(f"Skipping {model_name}: {exc}")
            continue
        model.config.id2label = id2label
        model.config.label2id = label2id

        training_args = TrainingArguments(
            output_dir=str(run_dir),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            logging_steps=50,
            fp16=use_fp16,
            gradient_accumulation_steps=args.grad_accum,
            seed=args.seed,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_builder(num_labels),
        )

        trainer.train()
        val_metrics = trainer.evaluate()
        test_metrics = trainer.evaluate(tokenized_test)

        trainer.save_model(str(run_dir))
        tokenizer.save_pretrained(str(run_dir))
        label_map_path = run_dir / "label_map.json"
        with label_map_path.open("w", encoding="utf-8") as f:
            json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

        result = {
            "model": model_name,
            "run_dir": str(run_dir),
            "val_accuracy": val_metrics.get("eval_accuracy"),
            "val_precision": val_metrics.get("eval_precision"),
            "val_recall": val_metrics.get("eval_recall"),
            "val_f1": val_metrics.get("eval_f1"),
            "test_accuracy": test_metrics.get("eval_accuracy"),
            "test_precision": test_metrics.get("eval_precision"),
            "test_recall": test_metrics.get("eval_recall"),
            "test_f1": test_metrics.get("eval_f1"),
        }
        results.append(result)
        success_count += 1

        if result["val_f1"] is not None and result["val_f1"] > best_f1:
            best_f1 = result["val_f1"]
            best_model_name = model_name

    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if best_model_name:
        best_path = output_dir / "best_model.txt"
        best_path.write_text(best_model_name, encoding="utf-8")

    label_map_path = output_dir / "label_map.json"
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print("Done. Results saved to", results_path)
    if best_model_name:
        print("Best model:", best_model_name)


if __name__ == "__main__":
    main()
