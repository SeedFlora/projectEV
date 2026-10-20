# ICDEES 2026 reproducibility package

Paper: **Shrinkage-Adaptive Weighted Soft Voting for Ensembles of Indonesian Pre-Trained Language Models in Electric Vehicle Sentiment Classification**. Budi Juarto and Kania Alma Tiara.

This package contains the actual local experiment artifacts used for the revised manuscript; it is not a newly trained or recomputed experiment.

## Included materials

- `experiments/probs_seeds/labels_primary_passes.npz`: numerical final/first-pass labels for 1,382 primary posts, with -1 marking 40 missing first-pass labels. No post text, handles, author identities, or worksheet metadata are included. The primary text/worksheets are available on request subject to privacy and redistribution review.
- `experiments/probs/`: seed-42 fold indices, validation/test probabilities and original training histories.
- `experiments/probs_seeds/`: seeds 1-4, matching fold indices/probabilities, training histories, transfer probabilities and numeric target labels. Each `.npz` can be read using `numpy.load(..., allow_pickle=False)`.
- `experiments/*.py`: original training and analysis scripts. `rev_lib.py` implements the fusion rules; `rev_*.py` reproduce the revised analyses from stored posteriors.
- `experiments/results/`: stored analysis output, including the revised multi-seed results in `rev/`. Older exploratory/single-seed outputs are retained separately by their original names.
- `paper/`: current accepted-title LaTeX, bibliography, original figure, number-checking scripts, and the current PDF. Only the Data Availability paragraph was updated for this release.

The current public tree excludes primary post text/worksheets and row-level textual examples. No author account credentials, private identifiers, account tokens, model caches, trained checkpoints, institutional payment forms, or unrelated project files are included. Numeric row positions serve only to align probabilities and labels.

## Privacy review

A follow-up commit removes the primary workbook and textual examples conservatively because independent privacy/redistribution clearance is not established. No embedded passwords, API tokens, personal phone numbers, institutional private identifiers, or private account email addresses were found in the published research artifacts; paper authors' public institutional contacts remain in the manuscript. If raw text was obtained from an earlier publication, this release does not grant permission to redistribute it. This sanitized release does not guarantee removal of copies retained by hosting services or third parties. No new licence is assigned to the primary text. Do not redistribute it from an earlier commit without the required clearance.

## External corpus sources

External source datasets are not re-licensed or copied in this release. Obtain them from their original authors, subject to their original terms:

- EV-Public: https://github.com/ryasakbar060/Dataset-Komentar-Kendaraan-Listrik — save the workbook as `experiments/data_ext/ryasakbar_2class.xlsx` (columns `Text`, `Label`).
- EV-Twitter: https://github.com/wynernawati/analisis-sentimen-kendaraan-listrik — save as `experiments/data_ext/wynernawati_twitter.csv` (columns `content`, `label`).
- EV-YouTube: https://www.kaggle.com/datasets/billycemerson/analisis-sentimen-terkait-intensif-mobil-listrik — save as `experiments/data_ext/cemerson_mobil_listrik.csv` (columns `text_cleaning`, `sentimen`).

The original loaders document label filtering, duplicate handling, and order. Stored numeric target labels and posteriors allow most offline fusion analyses without downloading external texts. Transfer text-length and annotation routines may require the original external sources.

## Reproduce offline analyses

Use Python with NumPy, SciPy, scikit-learn, pandas, openpyxl, matplotlib, and psutil. Training additionally requires PyTorch, Transformers, CUDA and the model checkpoints listed in `experiments/run_cv.py`; it was performed on the authors' GPU. The stored five-seed posteriors avoid retraining for fusion analysis.

Analyses that require raw/normalized text or the first-pass worksheet need the restricted primary worksheet and original external datasets at their documented paths. Numerical fusion analyses can use the released posteriors and labels.

From the repository root:

```sh
python experiments/rev_lib.py
python experiments/rev_all.py
```

`rev_all.py` checks that all five seeds are complete. The `run_cv.py` and `run_seeds.py` scripts retain the actual experiment configuration and split seed 42; change only cache/device paths as needed for your environment. They are not run automatically by this release. Original aggregate JSON files are the provenance for the paper's scientific numbers.

From `paper/`, build the manuscript with `pdflatex camera_ready.tex` twice. `make_numbers.py` and `prose_checks.py` compare reported numbers with stored results. No new scientific results are claimed by publishing these files.

`MANIFEST.sha256` records the bytes of all newly published research artifacts. No blanket new licence is assigned to third-party corpora or pretrained models; their original terms continue to apply.
