<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/XGBoost-Ensemble-orange?style=for-the-badge&logo=python"/>
<img src="https://img.shields.io/badge/SHAP-Explainable_AI-purple?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit"/>
<img src="https://img.shields.io/badge/Tests-30%20Passing-brightgreen?style=for-the-badge"/>

<br/><br/>

# 🧬 COVID-19 Hospitalization Risk Predictor

### *Production-Grade Calibrated Ensemble ML with Clinical Decision Support*

**Calibrated Stacking · SHAP Explainability · Real-Time Clinical Dashboard · 42+ Features · 30-Day Risk Stratification**

<br/>

> *"Transforming raw clinical data into life-saving decisions — one calibrated probability at a time."*

<br/>

[![CI](https://github.com/Aranya2801/COVID19-Hospitalization-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Aranya2801/COVID19-Hospitalization-Prediction/actions)
[![codecov](https://codecov.io/gh/Aranya2801/COVID19-Hospitalization-Prediction/branch/main/graph/badge.svg)](https://codecov.io/gh/Aranya2801/COVID19-Hospitalization-Prediction)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Results](#-results--performance)
- [Features](#-feature-engineering)
- [Quick Start](#-quick-start)
- [Dashboard](#-clinical-dashboard)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Citations](#-citations)
- [License](#-license)

---

## 🎯 Overview

**COVID-19 Hospitalization Risk Predictor** is an MIT-grade, clinically-motivated machine learning system that predicts 30-day hospitalization risk for COVID-19 patients using a calibrated ensemble of gradient-boosted trees. Built for both **daily clinical use** and **research reproducibility**.

### 🔑 Key Innovations

| Feature | Detail |
|---------|--------|
| **Ensemble Architecture** | XGBoost + LightGBM + Random Forest → Stacking with Logistic meta-learner |
| **Probability Calibration** | Isotonic regression calibration (Brier Score: **0.145**) |
| **SHAP Explainability** | Per-patient, per-feature attribution for clinical transparency |
| **Clinical Feature Engineering** | 16 engineered biomarker indices (Cytokine Storm Index, Severity Score, etc.) |
| **4-Tier Risk Stratification** | LOW / MEDIUM / HIGH / CRITICAL with clinical action protocols |
| **Production Dashboard** | Streamlit app with real-time patient assessment + population analytics |

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         RAW CLINICAL DATA                │
                    │  Demographics · Vitals · Labs · Vaccine  │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │        FEATURE ENGINEERING (42+)         │
                    │  Severity Score · Cytokine Storm Index   │
                    │  Coagulopathy Risk · Vaccine Protection  │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
   ┌──────────▼──────────┐  ┌──────────▼──────────┐  ┌─────────▼──────────┐
   │      XGBoost        │  │      LightGBM        │  │   Random Forest    │
   │  (AUC: 0.855)       │  │  (AUC: 0.849)        │  │  (AUC: 0.852)      │
   └──────────┬──────────┘  └──────────┬──────────┘  └─────────┬──────────┘
              └────────────────────────┼────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │         STACKING META-LEARNER            │
                    │       Logistic Regression (5-CV)         │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │       ISOTONIC CALIBRATION               │
                    │    Brier Score: 0.145 | AUC: 0.855       │
                    └──────────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼───────────────────────┐
              │                        │                       │
   ┌──────────▼──────────┐  ┌──────────▼─────────┐ ┌──────────▼──────────┐
   │  Risk Probability   │  │  SHAP Explanation  │ │ Clinical Tier       │
   │  0.00 → 1.00        │  │  Top-k Features    │ │ LOW/MED/HIGH/CRIT   │
   └─────────────────────┘  └────────────────────┘ └─────────────────────┘
```

---

## 📊 Results & Performance

### Model Metrics (Held-Out Test Set — 15%)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | **0.855** | Excellent discrimination |
| **Average Precision** | **0.926** | Strong positive class ranking |
| **Brier Score** | **0.145** | Well-calibrated probabilities |
| **Sensitivity (0.40)** | **0.84** | High recall for hospitalized |
| **Specificity (0.40)** | **0.73** | Acceptable false-positive control |

### 📈 ROC & Precision-Recall Curves

<div align="center">
<img src="docs/images/roc_pr_curves.png" width="85%"/>
</div>

### 🔍 SHAP Feature Importance

<div align="center">
<img src="docs/images/shap_feature_importance.png" width="80%"/>
</div>

### 🎯 Probability Calibration

<div align="center">
<img src="docs/images/calibration_curve.png" width="55%"/>
</div>

### 📊 Risk Distribution & Confusion Matrix

<table>
  <tr>
    <td><img src="docs/images/risk_distribution.png"/></td>
    <td><img src="docs/images/confusion_matrix.png"/></td>
  </tr>
</table>

### 🔬 Exploratory Data Analysis

<div align="center">
<img src="docs/images/eda_overview.png" width="90%"/>
</div>

<div align="center">
<img src="docs/images/correlation_heatmap.png" width="70%"/>
</div>

---

## ⚗️ Feature Engineering

The pipeline engineers **42+ features** from 28 raw inputs, including novel clinical composite indices:

### 🧪 Biomarker Flags
| Feature | Clinical Basis |
|---------|----------------|
| `hypoxia_flag` | O₂ saturation < 94% |
| `tachycardia_flag` | Heart rate > 100 bpm |
| `lymphopenia` | Lymphocytes < 20% |
| `high_crp` | CRP > 50 mg/L |
| `elevated_d_dimer` | D-Dimer > 0.5 mg/L |
| `high_ferritin` | Ferritin > 500 ng/mL |

### 🧬 Composite Indices
| Index | Formula |
|-------|---------|
| **Severity Score** | Weighted sum of 8 clinical flags (max 10) |
| **Cytokine Storm Index** | `(log(CRP) + log(Ferritin) + log(IL-6)) / 3` |
| **Coagulopathy Risk** | `D-Dimer × 2 + elevated_d_dimer × 3` |
| **Vaccine Protection** | `exp(-days/180) × doses × 0.3` |
| **Age-Comorbidity** | `age × comorbidity_count` |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Aranya2801/COVID19-Hospitalization-Prediction.git
cd COVID19-Hospitalization-Prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py --data data/raw/covid19_hospitalization_data.csv
```

**Output:**
```
============================================================
  COVID-19 Hospitalization Predictor — Training
============================================================
[DATA] Train: 10,200 | Val: 2,550
       Positive rate: 66.6%

  Training xgboost...   CV-AUC: 0.8577 ± 0.0094 | Val-AUC: 0.8554
  Training lightgbm...  CV-AUC: 0.8567 ± 0.0074 | Val-AUC: 0.8485
  Training random_forest... CV-AUC: 0.8612 ± 0.0037 | Val-AUC: 0.8562

  Building stacking ensemble...
  Calibrating probabilities...

  FINAL ENSEMBLE (Calibrated Stacking)
  Val-AUC   : 0.8550
  Avg-Prec  : 0.9255
  Brier     : 0.1448
============================================================
```

### 3. Launch Dashboard

```bash
streamlit run app.py
```

> Opens at `http://localhost:8501`

### 4. Python API

```python
import pickle
import pandas as pd
from src.models.predictor import COVID19HospitalizationPredictor

# Load model
with open("models/saved/covid19_predictor.pkl", "rb") as f:
    model = pickle.load(f)

# Predict on a batch
df = pd.read_csv("data/raw/covid19_hospitalization_data.csv")
risk_df = model.predict_risk(df.head(100))
print(risk_df.head())

# Individual patient explanation
patient = df.iloc[0]
explanation = model.explain_patient(patient)
print(f"Risk: {explanation['hospitalization_probability']:.1%}")
print(f"Tier: {explanation['risk_tier']}")
print(f"Action: {explanation['recommendation']}")
for factor in explanation['top_risk_factors'][:5]:
    print(f"  {factor['feature']:30s}  {factor['direction']}")
```

### 5. Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=term
# ===================== 30 passed ✅ =====================
```

---

## 🖥️ Clinical Dashboard

The **Streamlit dashboard** (`app.py`) provides four interactive modules:

| Tab | Features |
|-----|----------|
| 🏥 **Patient Assessment** | Real-time risk scoring, gauge widget, SHAP waterfall, clinical recommendation |
| 📊 **Model Analytics** | ROC/PR curves, calibration, confusion matrix, feature importance |
| 🗺️ **Population Explorer** | Interactive Plotly maps, filters by state/variant/age, trend charts |
| ℹ️ **About** | Architecture details, model card, citations |

<details>
<summary>📸 Dashboard Screenshots (click to expand)</summary>

**Patient Risk Assessment**
- Enter vitals, labs, demographics → instant probability + 4-tier risk badge
- SHAP top-10 contributing factors with direction arrows
- Gauge chart + clinical action protocol

**Population Analytics**
- Age distribution by hospitalization status
- State-level heatmaps
- Variant × vaccine dose interaction plots

</details>

---

## 📁 Project Structure

```
COVID19-Hospitalization-Prediction/
│
├── 📂 src/
│   ├── 📂 models/
│   │   └── predictor.py          # Core ML engine (600+ lines)
│   ├── 📂 utils/                 # Utility helpers
│   └── 📂 visualization/         # Plotting utilities
│
├── 📂 data/
│   ├── 📂 raw/
│   │   └── covid19_hospitalization_data.csv   # 15,000 patients · 42 features
│   └── 📂 processed/
│
├── 📂 models/saved/              # Serialized model artifacts
│   ├── covid19_predictor.pkl
│   └── evaluation_metrics.json
│
├── 📂 notebooks/
│   └── 01_full_pipeline.ipynb    # End-to-end analysis notebook
│
├── 📂 docs/images/               # All generated plots
│   ├── roc_pr_curves.png
│   ├── shap_feature_importance.png
│   ├── calibration_curve.png
│   ├── confusion_matrix.png
│   ├── risk_distribution.png
│   ├── eda_overview.png
│   └── correlation_heatmap.png
│
├── 📂 tests/
│   └── test_predictor.py         # 30 unit tests (pytest)
│
├── 📂 configs/
│   └── config.yaml               # All hyperparameters & thresholds
│
├── 📂 .github/workflows/
│   └── ci.yml                    # GitHub Actions CI (Python 3.9-3.11)
│
├── app.py                        # Streamlit clinical dashboard
├── train.py                      # Full training pipeline
├── requirements.txt
├── LICENSE                       # MIT
└── README.md
```

---

## 📦 Dataset

### Synthetic Clinical Dataset (`covid19_hospitalization_data.csv`)

| Property | Value |
|----------|-------|
| **Patients** | 15,000 |
| **Raw Features** | 42 |
| **Engineered Features** | 42+ |
| **Date Range** | March 2020 – November 2023 |
| **Hospitalization Rate** | 66.6% |
| **States** | 20 US states |
| **Variants** | Original, Alpha, Delta, Omicron, BA.2, XBB, JN.1 |

### Feature Categories

| Category | Features |
|----------|----------|
| **Demographics** | Age, Sex, Race/Ethnicity, BMI, State |
| **Vaccination** | Doses, Days since last dose |
| **Vital Signs** | O₂ sat, Temperature, HR, RR |
| **Laboratory** | CRP, D-Dimer, Ferritin, IL-6, WBC, Lymphocytes |
| **Comorbidities** | Diabetes, Hypertension, Heart disease, Lung disease, Obesity |
| **COVID Details** | Variant, Symptom onset days, Fever, Cough, SOB |
| **Outcomes** | Hospitalized, ICU, Ventilation, LOS, Mortality |

> **Note:** This dataset is synthetically generated for research and educational purposes, statistically calibrated to match published COVID-19 epidemiological literature (CDC, WHO, NEJM).

### Real Datasets (for extended research)
- [CDC COVID-19 Case Surveillance](https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data/vbim-akqf)
- [COVID-19 Open Research Dataset (CORD-19)](https://www.semanticscholar.org/cord19)
- [HealthData.gov — COVID-19 Reported Patient Impact](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh)

---

## 📐 Clinical Risk Tiers

| Tier | Probability | Action Protocol |
|------|------------|-----------------|
| 🟢 **LOW** | 0–20% | Monitor at home; telehealth follow-up in 48h |
| 🟡 **MEDIUM** | 20–40% | Outpatient monitoring; O₂ sat q6h; antiviral eligibility |
| 🟠 **HIGH** | 40–65% | Urgent ED evaluation; Remdesivir / monoclonal Ab consideration |
| 🔴 **CRITICAL** | 65–100% | **IMMEDIATE HOSPITALIZATION**; ICU-level monitoring |

---

## 🔬 Model Card

| | |
|--|--|
| **Model Type** | Calibrated Stacking Ensemble |
| **Base Learners** | XGBoost, LightGBM, Random Forest |
| **Meta-Learner** | Logistic Regression |
| **Calibration** | Isotonic Regression |
| **Explainability** | SHAP TreeExplainer |
| **Cross-Validation** | Stratified 5-Fold |
| **Intended Use** | Research, Education, Clinical Support Tool |
| **Limitations** | Synthetic data; not FDA-cleared |
| **Fairness** | Evaluated across race, sex, and age groups |

---

## 📚 Citations

```bibtex
@software{aranya2801_covid19_2024,
  author       = {Aranya2801},
  title        = {COVID-19 Hospitalization Risk Prediction},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/Aranya2801/COVID19-Hospitalization-Prediction},
  version      = {2.0.0}
}
```

**Key References:**
1. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
2. Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
3. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
4. CDC COVID-19 Response Team. (2020). Severe Outcomes Among Patients with COVID-19. *MMWR*.
5. Richardson, S. et al. (2020). Presenting Characteristics, Comorbidities, and Outcomes Among 5700 Patients. *JAMA*.

---

## ⚠️ Disclaimer

> This tool is developed for **research and educational purposes only**. It is **not FDA-cleared** and must **not** be used as a substitute for professional clinical judgment. Always consult licensed healthcare providers for medical decisions.

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ and rigor by **[Aranya2801](https://github.com/Aranya2801)**

⭐ **Star this repo** if you find it useful for your research!

</div>
