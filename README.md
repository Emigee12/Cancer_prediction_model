# Preemptive Cancer Risk Prediction — Breast & Prostate

A questionnaire-based machine-learning project that estimates an individual's risk of
**breast** or **prostate** cancer from non-invasive inputs, served through an interactive
**Streamlit** web app.

The goal is *preemptive screening support*: flagging individuals who may benefit from
clinical follow-up, using data a person can realistically provide (lifestyle,
reproductive history, and routine blood-test values) rather than imaging or biopsy.

> ⚠️ **Disclaimer:** This is an educational project. It does **not** provide a medical
> diagnosis and must not be used as a substitute for professional medical advice.

---

## Highlights

- **Two independent models** (breast, prostate), each benchmarking Logistic Regression,
  K-Nearest Neighbours, and SVM.
- **Leakage-safe pipeline:** feature selection, resampling, and scaling all run *inside*
  a cross-validation pipeline, so no information leaks from the validation folds.
- **Custom multi-method feature selector** that ranks features by averaging Mutual
  Information, Chi-squared, and ANOVA F-test ranks.
- **Class imbalance** handled with SMOTE; **decision thresholds tuned** for recall
  (missing a cancer is costlier than a false alarm).
- **Deployment-aware:** the custom transformer lives in an importable module so the
  saved models load standalone in the web app (train/serve parity).

## Results

Metrics are on a held-out 20% test set. **Logistic Regression is the deployed model**
for both (best recall/AUC balance and fully interpretable coefficients).

### Breast cancer
| Model | Accuracy | Recall (sensitivity) | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.93** | **0.99** | **0.95** | **0.96** |
| KNN (k=7) | 0.91 | 0.93 | 0.93 | 0.93 |
| SVM (RBF, C=0.5) | 0.90 | 0.91 | 0.93 | 0.95 |

High recall (0.99) is the headline number: the breast model misses very few true cancers.

### Prostate cancer
| Model | Accuracy | Recall (sensitivity) | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** ✅ | 0.72 | 0.71 | 0.72 | 0.72 |
| KNN (k=27) | 0.72 | 0.72 | 0.72 | 0.73 |
| SVM (RBF) | 0.72 | 0.71 | 0.72 | 0.73 |

The prostate models plateau around 0.72 ROC-AUC. All three algorithms converging on the
same ceiling points to a **data-quality limit** rather than a modelling one — see
[Limitations](#limitations). Collecting real questionnaire responses to retrain on is the
main line of future work.

---

## How it works

```
Questionnaire answers
        │
        ▼
  encoding.py  ──►  translates plain answers into the exact numeric
        │            features each model was trained on (train/serve parity)
        ▼
  Pipeline:  MultiMethodFeatureSelector ─► [MinMaxScaler*] ─► SMOTE ─► Classifier
        │                                   (*prostate only)
        ▼
  Threshold-tuned prediction  ──►  risk % + lower/higher-risk verdict
```

**Feature selection** (`ml_utils.py`) averages three statistical rankings — Mutual
Information, Chi-squared, and ANOVA F-test — and keeps the top *k* features
(breast k=10, prostate k=6). Because it is a proper scikit-learn transformer inside the
pipeline, it only ever sees the training fold during cross-validation.

---

## Project structure

```
├── streamlit_app.py        # the web app (run this)
├── encoding.py             # answer → model-feature translator
├── ml_utils.py             # MultiMethodFeatureSelector (shared, importable for unpickling)
├── app.py                  # breast-cancer training/benchmark script (LR/KNN/SVM + plots)
├── breast_modelv2.ipynb    # breast modelling notebook
├── prostate_modelv2.ipynb  # prostate modelling notebook
├── models/                 # trained, serialized models (.pkl)
├── data/                   # datasets (raw + preprocessed)
└── requirements.txt
```

---

## Running locally

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\activate           # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the app
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. Pick **Breast** or **Prostate** in the sidebar,
fill in the questionnaire, and click **Predict Risk**.

> **Note on versions:** the serialized models were trained with `scikit-learn==1.8.0`.
> Loading them under a different version may warn or fail, so the pinned versions in
> `requirements.txt` are recommended.

---

## Technical details

- **Imbalance:** SMOTE is applied *inside* the pipeline (training folds only), never to
  the test set.
- **Threshold tuning:** `TunedThresholdClassifierCV` optimizes the decision threshold for
  F1/recall instead of using the default 0.5 — the deployed breast model triggers at
  ≈0.32 and prostate at ≈0.49.
- **Scaling:** the breast dataset was pre-scaled during preprocessing; the prostate
  pipeline scales internally with `MinMaxScaler`. `encoding.py` reproduces the breast
  preprocessing at prediction time so live inputs match training.

---

## Limitations

- **Prostate dataset quality.** Performance ceilings near 0.72 ROC-AUC across all three
  algorithms suggest the available dataset has limited signal (it appears partly
  synthetic). The modelling approach is sound; the data is the bottleneck.
- **Not a diagnostic tool.** Outputs are statistical risk estimates for educational and
  screening-support purposes only.
- **Questionnaire vs. model inputs.** The questionnaire intentionally collects more fields
  than the models score on, to support future retraining.

## Future work

- Collect real questionnaire responses (via the companion Google Form) to retrain the
  prostate model on higher-quality data.
- Expand the deployed feature set once more data is available.
- UI/UX polish and optional logging of submissions for dataset growth.

---

## Tech stack

Python · scikit-learn · imbalanced-learn (SMOTE) · pandas · NumPy · Streamlit · Matplotlib / Seaborn

## Author

Final-year project — preemptive cancer prediction using questionnaire-based machine learning.

