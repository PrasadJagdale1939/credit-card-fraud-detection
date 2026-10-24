# 🛡️ Credit Card Fraud Detection

End-to-end Machine Learning project for detecting fraudulent credit card transactions —
built during the Practical Machine Learning module (CDAC-AI).

[![Python](https://img.shields.io/badge/Python-3.14-blue)]()
[![sklearn](https://img.shields.io/badge/scikit--learn-latest-orange)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()

---

## 📌 Problem Statement

Credit card companies must detect fraudulent transactions instantly — but frauds are
**extremely rare** (&lt;0.2% of transactions). A model that predicts "legitimate" for
everything achieves 99.8% accuracy while catching **zero fraud**. This project builds
and evaluates models that maximize **fraud recall** on severely imbalanced data.

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Features | V1–V28 (PCA-anonymized), Time, Amount |
| Fraud rate | **0.172%** (492 frauds) |
| Missing values | None |

&gt; ⚠️ The dataset (`creditcard.csv`) is **not committed** to this repo
&gt; (sensitive financial data). Download from Kaggle and place in `data/raw/`.

## 🔍 Key EDA Insights

1. **Extreme imbalance** (~1 legit : 578 fraud) — accuracy is meaningless; we optimize
   **Precision-Recall AUC** and **fraud recall**
2. **Frauds skew small-amount** (median $9.25 vs $22 legit) — card-testing behavior
3. **V14, V12, V10** show the clearest fraud/legit separation
4. **Time** carries weak predictive signal
5. 1,081 duplicate rows dropped during cleaning

## 🛠️ Approach

| Phase | Notebook | Focus |
|---|---|---|
| 1️⃣ | `01_eda.ipynb` | Exploration & insights (this README's findings) |
| 2️⃣ | `02_baseline.ipynb` | Naive Logistic Regression — demonstrate the "accuracy trap" |
| 3️⃣ | `03_imbalance_handling.ipynb` | SMOTE, class_weight, threshold tuning |
| 4️⃣ | `04_model_comparison.ipynb` | LR / DT / RF / XGBoost comparison → best model |

**Handling strategies compared:** `class_weight='balanced'` · SMOTE · threshold tuning

**Primary metric:** PR-AUC (average precision) — the correct metric for imbalanced data.
Supporting: fraud-class Recall, F1, confusion matrix.

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/PrasadJagdale1939/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux/Mac

# 3. Dependencies
pip install -r requirements.txt

# 4. Data → place creditcard.csv in data/raw/

# 5. Explore
jupyter notebook notebooks/01_eda.ipynb

# 🛡️ Credit Card Fraud Detection — Project Structure

A clean, production-style folder architecture for the Practical ML module (CDAC-AI).

---

## 📁 Directory Tree

```text
credit-card-fraud-detection/
│
├── .venv/                          # 🐍 Virtual environment (gitignored — never commit)
│
├── .gitignore                      # 🙈 Ignores venv/, data/, models/, checkpoints
├── README.md                       # 📘 Project overview, results, how to run
├── requirements.txt                # 📦 All dependencies with pinned versions
│
├── app/
│   └── streamlit_app.py            # 🚀 (Bonus, end of project) deploy demo
│
├── config/
│   └── config.yaml                 # ⚙️  Paths, random_state, test_size, model params
│
├── data/
│   ├── raw/                        # 💾 creditcard.csv (gitignored — sensitive + large)
│   └── processed/                  # 🧹 Cleaned / feature-engineered data (if saved)
│
├── models/                         # 💾 Saved .joblib files (gitignored)
│
├── notebooks/
│   ├── 01_eda.ipynb                # 🔍 Exploration & insights
│   ├── 02_baseline.ipynb           # ⚠️  Naive model + the "accuracy trap" demo
│   ├── 03_imbalance_handling.ipynb # ⚖️  SMOTE, class_weight, threshold tuning
│   └── 04_model_comparison.ipynb   # 🏆 Final comparison + best model selection
│
├── reports/
│   ├── figures/                    # 🖼️  All plots (class_dist.png, pr_curve.png ...)
│   └── results.csv                 # 📈 Model comparison table (auto-appended)
│
└── src/
    ├── __init__.py                 # 📦 Makes src/ a Python package
    ├── data_loader.py              # ⬇️  load_data(), split_data()
    ├── preprocess.py               # 🔄 Scaling, pipeline builders
    ├── train.py                    # 🏋️ train_model() function
    ├── evaluate.py                 # 📊 All metrics, plots, confusion matrix
    └── utils.py                    # 🛠️  set_seed(), save/load model helpers

```

