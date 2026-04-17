# 🫀 Heart Disease Prediction — End-to-End ML Pipeline

A complete machine learning project for predicting coronary artery disease (CAD) using non-invasive clinical data. The project covers the full pipeline from exploratory data analysis to an optimized, interpretable Random Forest model.

---

## 📌 Business Problem

Coronary artery disease diagnosis currently relies heavily on coronary angiography — an invasive, costly, and risky procedure. This project explores whether non-invasive clinical data (ECG results, cholesterol levels, exercise test results) can effectively serve as a pre-screening tool, reducing unnecessary referrals to the catheterization lab.

**Goal:** Build a binary classification model that predicts CAD presence from 11 clinical features, optimizing for **Recall** to minimize missed diagnoses — which carry far greater clinical risk than false positives.

---

## 📂 Dataset

| Property | Details |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Patients | 920 patients across 4 international medical institutions |
| Features | 11 clinical attributes (after preprocessing) |
| Target | Binary: `0` = No CAD, `1` = CAD diagnosed |
| Class Balance | 55.3% CAD / 44.7% No CAD |

Institutions: Cleveland Clinic Foundation (USA), Hungarian Institute of Cardiology (Hungary), V.A. Medical Center (Long Beach, USA), University Hospitals (Switzerland).

**Reference:** Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304–310.

---

## 🗂️ Project Structure

```
heart-disease-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── heart_disease_uci.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline_Model.ipynb
│   └── 03_Advanced_Model.ipynb
```

---

## 🔍 Pipeline Overview

The project is structured as three progressive notebooks:

### Notebook 1 — Exploratory Data Analysis (`01_EDA.ipynb`)

| Section | Description |
|---|---|
| Data Familiarization | Data types, shape, data dictionary |
| Preprocessing | Column dropping, renaming, target binarization |
| Missing Values | Detection and strategy planning |
| Numerical Features | Descriptive stats, KDE distributions, outliers, pairplots, regression plots |
| Categorical Features | Count plots by target class, contingency tables |
| Correlation Analysis | Pearson, Spearman, and PhiK matrices |
| PCA | 2D and 3D dimensionality reduction |
| EDA Summary | Key findings and modeling recommendations |

### Notebook 2 — Baseline Model (`02_Baseline_Model.ipynb`)

| Section | Description |
|---|---|
| Preprocessing | Imputation (median/mode), Label Encoding, train/test split |
| Constant Baseline | DummyClassifier — establishes the minimum performance threshold |
| Decision Tree | Training, evaluation, feature importance, tree visualization |
| Overfitting Check | Train vs. test metrics comparison |
| Results Comparison | Metrics table + ROC curve |

### Notebook 3 — Advanced Model (`03_Advanced_Model.ipynb`)

| Section | Description |
|---|---|
| sklearn Pipeline | Automated preprocessing with ColumnTransformer |
| Random Forest (default) | Internal baseline with default parameters |
| GridSearchCV | Hyperparameter tuning with 5-fold cross-validation |
| Final Model Evaluation | Confusion matrix + ROC curve |
| Model Interpretation | MDI, Permutation Importance, SHAP (global + local) |

---

## 📊 Key Findings

### EDA
- **Most predictive features** (by PhiK correlation): `chest_pain_type` (0.75), `num_major_vessels` (0.68), `exercise_angina` (0.66)
- **Least predictive**: `fasting_blood_sugar` and `resting_ecg`
- **Data quality issues**: `num_major_vessels` (66.4% missing), `thalassemia` (52.8% missing), zero values in `cholesterol` and `resting_bp`

### Modeling Results

| Model | Recall | F1-Score | ROC-AUC |
|---|---|---|---|
| DummyClassifier | 1.00* | 0.72 | 0.50 |
| Decision Tree (Notebook 2) | 0.75 | 0.77 | 0.75 |
| Random Forest Default | 0.83 | 0.85 | 0.88 |
| **Random Forest Tuned (Final)** | **0.82** | **0.83** | **0.89** |

*DummyClassifier's Recall=1.00 is an artifact of predicting class 1 for all patients — it has no real diagnostic capability (ROC-AUC=0.50).

**Key improvements from Decision Tree → Random Forest Tuned:**
- Recall: +7 percentage points (0.75 → 0.82)
- ROC-AUC: +14 percentage points (0.75 → 0.89)
- False Negatives (missed CAD): reduced from 39 to 28 patients

### Model Interpretation (SHAP)
Three feature importance methods (MDI, Permutation Importance, SHAP) consistently identified:
- `chest_pain_type`, `exercise_angina`, `st_depression` — primary predictors
- SHAP force plots confirmed model decisions are clinically consistent

---

## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-blue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.43-green)

- **Data manipulation:** `pandas`, `numpy`
- **Visualization:** `seaborn`, `matplotlib`
- **Preprocessing & Pipelines:** `scikit-learn` (Pipeline, ColumnTransformer, SimpleImputer)
- **Modeling:** `scikit-learn` (DummyClassifier, DecisionTreeClassifier, RandomForestClassifier)
- **Hyperparameter Tuning:** `GridSearchCV`
- **Model Interpretation:** `shap`, `permutation_importance`
- **Correlation Analysis:** `phik`
- **Automated EDA:** `sweetviz`

---

## 🚀 How to Run

1. Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the dataset from [UCI Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) and place `heart_disease_uci.csv` in the `data/` folder.

4. Run the notebooks in order:
```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Baseline_Model.ipynb
jupyter notebook notebooks/03_Advanced_Model.ipynb
```

---

## 📋 Requirements

```
pandas
numpy
seaborn
matplotlib
scipy
scikit-learn
phik
sweetviz
shap
jupyter
```
