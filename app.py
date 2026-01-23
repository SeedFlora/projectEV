import json
import os
import re
from collections import Counter
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from wordcloud import WordCloud


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)


def resolve_model_source():
    env_id = os.environ.get("MODEL_ID")
    if env_id:
        return env_id

    env_dir = os.environ.get("MODEL_DIR")
    if env_dir:
        return Path(env_dir)

    outputs_dir = Path("outputs_v4")
    best_file = outputs_dir / "best_model.txt"
    if best_file.exists():
        best_name = best_file.read_text(encoding="utf-8").strip()
        return outputs_dir / sanitize_model_name(best_name)

    outputs_dir = Path("outputs")
    best_file = outputs_dir / "best_model.txt"
    if not best_file.exists():
        raise FileNotFoundError(
            "Best model not found. Set MODEL_ID/MODEL_DIR or run train_compare.py first."
        )

    best_name = best_file.read_text(encoding="utf-8").strip()
    return outputs_dir / sanitize_model_name(best_name)


DATA_PATH = os.environ.get("DATA_PATH", "ev_data_relabel_balanced_v2.xlsx")
RESULTS_PATH = os.environ.get("RESULTS_PATH", "outputs_v4/results.csv")
ANALYTICS_TEXT_COL = os.environ.get("ANALYTICS_TEXT_COL", "clean_text_formal")
ANALYTICS_LABEL_COL = os.environ.get("ANALYTICS_LABEL_COL", "label")

MODEL_SOURCE = resolve_model_source()
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_SOURCE, use_fast=True)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_SOURCE)
MODEL.eval()

ID2LABEL = MODEL.config.id2label

STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "pada",
    "ini",
    "itu",
    "atau",
    "juga",
    "dengan",
    "karena",
    "bahwa",
    "sudah",
    "belum",
    "tidak",
    "bukan",
    "jadi",
    "agar",
    "sebagai",
    "lebih",
    "paling",
    "seperti",
    "saja",
    "masih",
    "bisa",
    "dapat",
    "akan",
    "kami",
    "kita",
    "saya",
    "anda",
    "mereka",
    "aku",
    "dia",
    "kamu",
    "nya",
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "to",
    "in",
    "for",
    "on",
    "it",
    "mobil",
    "wuling",
    "binguo",
    "air",
    "ev",
    "listrik",
    "kendaraan",
    "hyundai",
    "ioniq",
    "cloud",
    "confero",
    "cortez",
    "almaz",
    "byd",
    "scbd",
    "gocar",
    "go",
    "car",
    "kalau",
    "sama",
    "menurut",
    "ada",
    "demikian",
    "tapi",
    "sangat",
    "beli",
    "punya",
    "benar",
    "secara",
    "harga",
    "begitu",
}

BUSINESS_KEYWORDS = {
    "harga",
    "murah",
    "mahal",
    "diskon",
    "promo",
    "subsidi",
    "pajak",
    "biaya",
    "operasional",
    "hemat",
    "irit",
    "boros",
    "baterai",
    "daya",
    "kwh",
    "kilowatt",
    "range",
    "jarak",
    "tempuh",
    "pengisian",
    "cas",
    "charge",
    "charging",
    "charger",
    "waktu",
    "lama",
    "cepat",
    "port",
    "infrastruktur",
    "stasiun",
    "pln",
    "garansi",
    "warranty",
    "after",
    "sales",
    "purna",
    "servis",
    "service",
    "bengkel",
    "perawatan",
    "sparepart",
    "suku",
    "cadang",
    "kualitas",
    "awet",
    "tahan",
    "durable",
    "aman",
    "keamanan",
    "keselamatan",
    "airbag",
    "abs",
    "kamera",
    "sensor",
    "fitur",
    "teknologi",
    "performa",
    "akselerasi",
    "kecepatan",
    "torsi",
    "tenaga",
    "motor",
    "handling",
    "suspensi",
    "kabin",
    "kapasitas",
    "luas",
    "sempit",
    "nyaman",
    "kursi",
    "bagasi",
    "modis",
    "stylish",
    "desain",
    "retro",
    "futuristik",
    "value",
    "worth",
    "resale",
    "jual",
    "nilai",
    "seken",
    "bekas",
    "kredit",
    "cicil",
    "uang",
    "dp",
    "asuransi",
    "keluhan",
    "komplain",
    "rusak",
    "error",
    "mogok",
    "overheat",
    "panas",
    "lemot",
    "lambat",
    "ribet",
    "sulit",
    "mudah",
    "praktis",
}


def read_table(path: str) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_excel(p)


def load_dataset(path: str, text_col: str, label_col: str):
    df = read_table(path)
    if df is None:
        return None, {}
    if text_col not in df.columns or label_col not in df.columns:
        return None, {}
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    labels = sorted(df[label_col].unique().tolist())
    if set(labels) == {0, 2}:
        label_name = {0: "Negatif", 2: "Positif"}
    elif set(labels) == {0, 1}:
        label_name = {0: "Negatif", 1: "Positif"}
    else:
        label_name = {val: f"Label {val}" for val in labels}
    return df, label_name


def load_results():
    df = read_table(RESULTS_PATH)
    if df is None:
        return None
    return df


RESULTS_DF = load_results()


def predict(text):
    if not text or not text.strip():
        return {}

    inputs = TOKENIZER(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = MODEL(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    scores = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}
    return scores


def _label_from_scores(scores: dict) -> tuple[str, float]:
    if not scores:
        return "", 0.0
    best_label = max(scores, key=scores.get)
    return best_label, float(scores[best_label])


def batch_predict(upload_file, text_col):
    if not upload_file:
        return pd.DataFrame(), None
    df = read_table(upload_file)
    if df is None or text_col not in df.columns:
        return pd.DataFrame(), None
    df = df.copy()
    df[text_col] = df[text_col].astype(str)

    score_labels = list(ID2LABEL.values())
    pred_labels = []
    pred_scores = []
    score_cols = {f"score_{lbl}": [] for lbl in score_labels}

    for text in df[text_col].tolist():
        scores = predict(text)
        label, score = _label_from_scores(scores)
        pred_labels.append(label)
        pred_scores.append(score)
        for lbl in score_labels:
            score_cols[f"score_{lbl}"].append(float(scores.get(lbl, 0.0)))

    df["pred_label"] = pred_labels
    df["pred_score"] = pred_scores
    for col, vals in score_cols.items():
        df[col] = vals

    out_path = Path("batch_predictions.csv")
    df.to_csv(out_path, index=False)
    return df, str(out_path)


def _tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 2]
    return tokens


def _business_filter(tokens: list[str], min_keep: int = 30):
    if not tokens:
        return tokens
    business = [t for t in tokens if t in BUSINESS_KEYWORDS]
    if len(business) >= min_keep:
        return business
    return tokens


def _filter_df(df: pd.DataFrame, label_name: dict, label_choice: str, label_col: str):
    if df is None:
        return None
    if label_choice == "Semua":
        return df
    label_val = None
    for val, name in label_name.items():
        if name == label_choice:
            label_val = val
            break
    if label_val is None:
        return df
    return df[df[label_col] == label_val]


def build_distribution_plot(df: pd.DataFrame, label_name: dict, label_col: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    if df is None:
        ax.text(0.5, 0.5, "Dataset tidak ditemukan", ha="center", va="center")
        return fig
    counts = df[label_col].value_counts().sort_index()
    labels = [label_name.get(val, str(val)) for val in counts.index.tolist()]
    bars = ax.bar(labels, counts.values, color=["#ef4444", "#22c55e"])
    ax.set_title("Distribusi Label")
    ax.set_ylabel("Jumlah")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    total = counts.sum()
    if total > 0:
        for bar, val in zip(bars, counts.values):
            pct = val / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(val)} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylim(0, max(counts.values) * 1.15)
    return fig


def build_top_words_plot(df: pd.DataFrame, label_name: dict, label_choice: str, text_col: str, label_col: str, top_n: int = 20):
    fig, ax = plt.subplots(figsize=(6, 5))
    if df is None or df.empty:
        ax.text(0.5, 0.5, "Data kosong", ha="center", va="center")
        return fig
    sub_df = _filter_df(df, label_name, label_choice, label_col)
    tokens = []
    for text in sub_df[text_col].tolist():
        tokens.extend(_tokenize(text))
    tokens = _business_filter(tokens)
    if not tokens:
        ax.text(0.5, 0.5, "Token kosong", ha="center", va="center")
        return fig
    common = Counter(tokens).most_common(top_n)
    words = [w for w, _ in common][::-1]
    freqs = [c for _, c in common][::-1]
    bars = ax.barh(words, freqs, color="#3b82f6")
    ax.set_title(f"Top {top_n} Kata - {label_choice}")
    max_val = max(freqs) if freqs else 0
    if max_val > 0:
        for bar, val in zip(bars, freqs):
            ax.text(
                bar.get_width() + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(val)}",
                va="center",
                ha="left",
                fontsize=8,
            )
        ax.set_xlim(0, max_val * 1.15)
    return fig


def build_wordcloud(df: pd.DataFrame, label_name: dict, label_choice: str, text_col: str, label_col: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if df is None or df.empty:
        ax.text(0.5, 0.5, "Data kosong", ha="center", va="center")
        ax.axis("off")
        return fig
    sub_df = _filter_df(df, label_name, label_choice, label_col)
    tokens = []
    for text in sub_df[text_col].tolist():
        tokens.extend(_tokenize(text))
    tokens = _business_filter(tokens)
    if not tokens:
        ax.text(0.5, 0.5, "Token kosong", ha="center", va="center")
        ax.axis("off")
        return fig
    wc = WordCloud(width=900, height=500, background_color="white", collocations=False)
    wc.generate(" ".join(tokens))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud - {label_choice}")
    return fig


def build_model_comparison_plot():
    fig, ax = plt.subplots(figsize=(6, 4))
    if RESULTS_DF is None or RESULTS_DF.empty:
        ax.text(0.5, 0.5, "results.csv tidak ditemukan", ha="center", va="center")
        return fig
    data = RESULTS_DF.copy()
    if "val_f1" not in data.columns:
        ax.text(0.5, 0.5, "Kolom val_f1 tidak ada", ha="center", va="center")
        return fig
    data = data.sort_values("val_f1", ascending=False)
    models = data["model"].tolist()
    val = data.get("val_f1", []).tolist()
    test = data.get("test_f1", []).tolist() if "test_f1" in data.columns else None
    x = range(len(models))
    bars_val = ax.bar(x, val, width=0.4, label="Val F1", color="#22c55e")
    if test is not None:
        bars_test = ax.bar([i + 0.4 for i in x], test, width=0.4, label="Test F1", color="#3b82f6")
        ax.set_xticks([i + 0.2 for i in x])
    else:
        ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Perbandingan Model (F1)")
    ax.legend()
    for bar in bars_val:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if test is not None:
        for bar in bars_test:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    return fig


def build_insights(df: pd.DataFrame, label_name: dict):
    if df is None or df.empty:
        return "Tidak ada data untuk analitik."

    total = len(df)
    counts = df[ANALYTICS_LABEL_COL].value_counts().to_dict()
    parts = []
    for k, v in counts.items():
        name = label_name.get(k, str(k))
        pct = v / total * 100 if total else 0
        parts.append(f"{name}: {v} ({pct:.1f}%)")
    label_summary = ", ".join(parts)

    tokens = []
    for text in df[ANALYTICS_TEXT_COL].tolist():
        tokens.extend(_tokenize(text))
    tokens = _business_filter(tokens)
    top_words = [w for w, _ in Counter(tokens).most_common(3)]
    top_words_text = ", ".join(top_words) if top_words else "tidak ada"

    best_model = "tidak tersedia"
    if RESULTS_DF is not None and not RESULTS_DF.empty and "val_f1" in RESULTS_DF.columns:
        best_row = RESULTS_DF.sort_values("val_f1", ascending=False).iloc[0]
        model_name = best_row.get("model", "model")
        val_f1 = best_row.get("val_f1")
        test_f1 = best_row.get("test_f1")
        if pd.notna(val_f1):
            best_model = f"{model_name} (val_f1 {val_f1:.3f}" + (f", test_f1 {test_f1:.3f}" if pd.notna(test_f1) else "") + ")"

    return (
        f"Dataset berisi {total} baris dengan komposisi label: {label_summary}. "
        f"Kata paling sering muncul: {top_words_text}. "
        f"Model terbaik saat ini: {best_model}."
    )


def _safe_pct(part: int, total: int) -> float:
    return (part / total * 100.0) if total else 0.0


def _summarize_label_counts(df: pd.DataFrame, label_name: dict, label_col: str):
    if df is None or df.empty:
        return "tidak ada data"
    counts = df[label_col].value_counts().to_dict()
    total = len(df)
    parts = []
    for k, v in counts.items():
        name = label_name.get(k, str(k))
        parts.append(f"{name}: {v} ({_safe_pct(v, total):.1f}%)")
    return ", ".join(parts)


def _top_words_text(df: pd.DataFrame, text_col: str, top_k: int = 3):
    if df is None or df.empty:
        return "tidak ada"
    tokens = []
    for text in df[text_col].tolist():
        tokens.extend(_tokenize(text))
    top_words = [w for w, _ in Counter(tokens).most_common(top_k)]
    return ", ".join(top_words) if top_words else "tidak ada"


def build_insights(df: pd.DataFrame, label_name: dict, label_choice: str):
    if df is None or df.empty:
        return "Tidak ada data untuk analitik."

    sub_df = _filter_df(df, label_name, label_choice, ANALYTICS_LABEL_COL)
    total = len(df)
    sub_total = len(sub_df)
    label_summary_all = _summarize_label_counts(df, label_name, ANALYTICS_LABEL_COL)
    label_summary_sub = _summarize_label_counts(sub_df, label_name, ANALYTICS_LABEL_COL)
    top_words = _top_words_text(sub_df, ANALYTICS_TEXT_COL)

    return (
        f"Filter: {label_choice}. Subset berisi {sub_total} baris "
        f"({ _safe_pct(sub_total, total):.1f}% dari total). "
        f"Komposisi subset: {label_summary_sub}. "
        f"Komposisi total: {label_summary_all}. "
        f"Kata paling sering pada subset: {top_words}."
    )


def build_model_explanation():
    if RESULTS_DF is None or RESULTS_DF.empty or "val_f1" not in RESULTS_DF.columns:
        return "Penjelasan model terbaik belum tersedia karena results.csv tidak ditemukan."
    best_row = RESULTS_DF.sort_values("val_f1", ascending=False).iloc[0]
    model_name = best_row.get("model", "model terbaik")
    val_f1 = best_row.get("val_f1")
    test_f1 = best_row.get("test_f1")
    score_text = f"val_f1 {val_f1:.3f}" if pd.notna(val_f1) else "val_f1 tidak tersedia"
    if pd.notna(test_f1):
        score_text += f", test_f1 {test_f1:.3f}"

    explanation = (
        f"Model terbaik saat ini adalah **{model_name}** ({score_text}). "
        "Untuk tugas klasifikasi pada satu bahasa, model monolingual sering unggul "
        "dibanding model multilingual karena representasi dan tokenisasi lebih selaras "
        "dengan bahasa target. Temuan ini konsisten dengan studi yang menunjukkan model "
        "monolingual mengungguli atau setidaknya kompetitif terhadap model multilingual "
        "pada tugas klasifikasi dan sentiment di bahasa yang sama. [1][2]"
    )

    literature = (
        "Literatur 5 tahun terakhir menekankan dua faktor besar: objective pretraining yang "
        "lebih efektif dan regularisasi saat fine-tuning. DeBERTaV3 menunjukkan bahwa "
        "replaced-token detection (RTD) yang lebih efisien meningkatkan kualitas representasi "
        "untuk NLU, sementara R-Drop meningkatkan generalisasi dengan memaksa konsistensi "
        "output antar dropout masks saat fine-tuning. Faktor-faktor ini membantu menjelaskan "
        "kenapa model dengan pretraining kuat dan fine-tuning stabil cenderung mencapai F1 lebih tinggi. [3][4]"
    )

    references = (
        "**Referensi:**\n"
        "[1] Micallef et al., 2022, *Pre-training Data Quality and Quantity for a Low-Resource Language: New Corpus and BERT Models for Maltese* (ACL). "
        "https://aclanthology.org/2022.deeplo-1.10/\n"
        "[2] Nguyen et al., 2022, *SMTCE: A Social Media Text Classification Evaluation Benchmark and BERTology Models for Vietnamese* (arXiv). "
        "https://arxiv.org/abs/2209.10482\n"
        "[3] He et al., 2021, *DeBERTaV3* (arXiv). https://arxiv.org/abs/2111.09543\n"
        "[4] Liang et al., 2021, *R-Drop* (arXiv). https://arxiv.org/abs/2106.14448"
    )
    return f"{explanation}\n\n{literature}\n\n{references}"


def analytics(label_choice):
    df, label_name = load_dataset(DATA_PATH, ANALYTICS_TEXT_COL, ANALYTICS_LABEL_COL)
    label_options = ["Semua"] + list(label_name.values()) if label_name else ["Semua"]
    if label_choice not in label_options:
        label_choice = "Semua"

    sub_df = _filter_df(df, label_name, label_choice, ANALYTICS_LABEL_COL)
    dist_fig = build_distribution_plot(sub_df, label_name, ANALYTICS_LABEL_COL)
    top_fig = build_top_words_plot(
        sub_df, label_name, label_choice, ANALYTICS_TEXT_COL, ANALYTICS_LABEL_COL
    )
    wc_fig = build_wordcloud(
        sub_df, label_name, label_choice, ANALYTICS_TEXT_COL, ANALYTICS_LABEL_COL
    )
    model_fig = build_model_comparison_plot()

    if df is None:
        summary = pd.DataFrame([{"metric": "rows", "value": 0}])
    else:
        summary = pd.DataFrame(
            [{"metric": "rows", "value": len(df)}]
            + [
                {"metric": f"label_{label_name.get(k, k)}", "value": v}
                for k, v in df[ANALYTICS_LABEL_COL].value_counts().to_dict().items()
            ]
        )

    insights = build_insights(df, label_name, label_choice)
    model_explain = build_model_explanation()
    dropdown_update = gr.update(choices=label_options, value=label_choice)
    return dist_fig, top_fig, wc_fig, model_fig, summary, insights, model_explain, dropdown_update


with gr.Blocks(title="Klasifikasi Sentimen EV") as app:
    gr.Markdown("# Klasifikasi Sentimen EV")
    gr.Markdown("Prediksi sentimen + dashboard analitik (word cloud & distribusi label).")

    with gr.Tab("Prediksi"):
        inp = gr.Textbox(lines=4, label="Teks")
        out = gr.Label(num_top_classes=2, label="Prediksi")
        btn = gr.Button("Prediksi")
        btn.click(predict, inputs=inp, outputs=out)

    with gr.Tab("Analitik"):
        label_choice = gr.Dropdown(["Semua"], value="Semua", label="Filter Label")
        dist_plot = gr.Plot(label="Distribusi Label")
        top_plot = gr.Plot(label="Top Kata")
        wc_plot = gr.Plot(label="Word Cloud")
        model_plot = gr.Plot(label="Perbandingan Model")
        summary_tbl = gr.Dataframe(label="Ringkasan Dataset", interactive=False)
        insight_text = gr.Markdown()
        model_text = gr.Markdown()

        run_btn = gr.Button("Generate")
        run_btn.click(
            analytics,
            inputs=label_choice,
            outputs=[dist_plot, top_plot, wc_plot, model_plot, summary_tbl, insight_text, model_text, label_choice],
        )
        label_choice.change(
            analytics,
            inputs=label_choice,
            outputs=[dist_plot, top_plot, wc_plot, model_plot, summary_tbl, insight_text, model_text, label_choice],
        )

    with gr.Tab("Batch"):
        batch_file = gr.File(label="Upload data (xlsx/csv)", type="filepath")
        batch_text_col = gr.Textbox(value="clean_text_formal", label="Nama kolom teks")
        batch_btn = gr.Button("Proses Batch")
        batch_out = gr.Dataframe(label="Hasil Prediksi", interactive=False)
        batch_download = gr.File(label="Download CSV")
        batch_btn.click(
            batch_predict,
            inputs=[batch_file, batch_text_col],
            outputs=[batch_out, batch_download],
        )

    app.load(
        analytics,
        inputs=label_choice,
        outputs=[dist_plot, top_plot, wc_plot, model_plot, summary_tbl, insight_text, model_text, label_choice],
    )


if __name__ == "__main__":
    app.launch()
