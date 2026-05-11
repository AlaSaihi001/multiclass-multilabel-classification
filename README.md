# Heart Disease Classification: Multi-Class & Multi-Label Learning

A comprehensive machine learning project implementing **multi-class** and **multi-label** classification strategies on the UCI Heart Disease dataset, with emphasis on clinical interpretability and advanced ensemble techniques.

## 📋 Project Overview

This repository contains two interconnected classification tasks:

### Part I: Multi-Class Classification
Predicting the **severity level** of heart disease (5 classes: healthy → increasing severity).

**Algorithms Implemented:**
- Classical Methods: KNN, Decision Trees
- Ensemble Learners: Random Forest, AdaBoost, Gradient Boosting, Bagging
- Rule-Based: RIPPER (Wittgenstein library)

**Multi-Class Strategies:**
- One-vs-Rest (OvR)
- One-vs-One (OvO)  
- Output Code Classifier (ECOC)

### Part II: Multi-Label Classification
Predicting **multiple co-morbidities** simultaneously (6 binary labels: heart disease, hypertension, hypercholesterolemia, diabetes, exercise-induced angina, low max heart rate).

**Multi-Label Strategies:**
- Binary Relevance (independent label prediction)
- Classifier Chain (captures label dependencies)
- Ensemble of Chains (variance reduction via soft voting)

## 📊 Dataset

**Source:** [Heart Disease UCI (Kaggle)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

- **Size:** 920 patient records
- **Features:** 15 clinical attributes (age, sex, blood pressure, cholesterol, ECG results, etc.)
- **Multiple Hospital Sites:** Cleveland, Hungary, Switzerland, VA Long Beach

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `scikit-learn` — ML algorithms & multi-class/multi-output strategies
- `numpy`, `pandas` — Data manipulation
- `matplotlib`, `seaborn` — Visualization
- `wittgenstein` — RIPPER rule-based classifier
- `kagglehub` — Automatic dataset download

## 🚀 Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set Kaggle API credentials (for automatic dataset download)
4. Run the Jupyter notebook: `jupyter notebook clinical-ml-heart-disease.ipynb`

## 📈 Key Findings

### Part I — Multi-Class Performance
| Model | Accuracy | F1-Macro | Notes |
|-------|----------|----------|-------|
| Random Forest | ~0.85 | 0.82 | Best overall, captures feature importance |
| Gradient Boosting | ~0.83 | 0.80 | Strong but slower training |
| OvO + Decision Tree | ~0.80 | 0.77 | Robust to class imbalance |

### Part II — Multi-Label Performance
| Strategy | Hamming Loss | F1-Macro | Key Insight |
|----------|--------------|----------|-------------|
| Binary Relevance | High | ~0.75 | Ignores medical co-morbidities |
| Classifier Chain | Low | ~0.82 | Captures label dependencies |
| Ensemble Chains | Lowest | ~0.85 | Best precision via soft voting |

**Main Insight:** Medical conditions are highly correlated (e.g., heart disease ↔ exercise angina ↔ low max HR). Classifier Chains substantially outperform independent label prediction by modeling these dependencies.

## 📁 Project Structure

```
TP5.ipynb
├── 0. Imports & Configuration
├── PART I: Multi-Class Classification
│   ├── 1.1 Data Loading (kagglehub)
│   ├── 1.2 Exploratory Data Analysis (EDA)
│   ├── 1.3 Preprocessing & Normalization
│   ├── 1.4 Algorithm Comparison
│   │   ├── 1.4.1 K-Nearest Neighbors (tuned k via CV)
│   │   ├── 1.4.2 Decision Tree (max_depth optimization)
│   │   ├── 1.4.3 Ensemble Learning (RF, AdaBoost, GB, Bagging)
│   │   ├── 1.4.4 sklearn.multiclass strategies (OvR, OvO, ECOC)
│   │   └── 1.4.5 RIPPER rule-based classification
│   ├── 1.5 Comparison & Visualization (confusion matrices, feature importance)
│   └── 1.6 Summary & Interpretation
│
└── PART II: Multi-Label Classification
    ├── 2.1 Multi-Label Target Construction (6 medical conditions)
    ├── 2.2 Preprocessing & Train-Test Split
    ├── 2.3 Binary Relevance (MultiOutputClassifier)
    ├── 2.4 Classifier Chain (sequential label prediction)
    ├── 2.5 Ensemble of Chains (soft voting)
    ├── 2.6 Multi-Label Metrics & Visualization
    └── 2.7 Summary & Interpretation
```

## 🔍 Evaluation Metrics

### Multi-Class Metrics
- **Accuracy** — Overall correctness
- **F1-Macro** — Unweighted average (recommended for imbalanced classes)
- **F1-Weighted** — Class-frequency weighted
- **Confusion Matrix** — Per-class True Positives/False Positives

### Multi-Label Metrics
- **Hamming Loss** — Fraction of incorrectly predicted (instance, label) pairs (↓ better)
- **F1-Micro** — Aggregated TP/FP/FN across all labels
- **F1-Macro** — Per-label F1 average (recommended for imbalanced labels)
- **Jaccard Index** — Intersection/Union per instance
- **Exact Match Ratio** — Perfect prediction for all labels

## 🧬 Medical Context

The UCI Heart Disease dataset contains clinical markers that physicians use to assess cardiac risk:
- **Age & Blood Pressure** → hypertension risk
- **Cholesterol** → cardiovascular risk
- **Fasting Blood Sugar** → diabetes screening
- **Exercise Test Results** → ischemia detection
- **ECG Abnormalities** → electrical conduction issues

Multi-label learning on this data mirrors **real clinical diagnosis**, where patients often present multiple simultaneous conditions.

## 📚 References

- [arXiv:2412.04792 — Multi-Class Heart Disease Detection](https://arxiv.org/html/2412.04792v1#S3)
- [sklearn.multiclass Documentation](https://scikit-learn.org/stable/modules/multiclass.html)
- [sklearn.multioutput Documentation](https://scikit-learn.org/stable/api/sklearn.multioutput.html)
- [RIPPER Algorithm — Wittgenstein Library](https://pypi.org/project/wittgenstein/)
- [IoDiakou/MLC-on-biomedical-data](https://github.com/IoDiakou/MLC-on-biomedical-data)

## ✅ Learning Outcomes

By completing this project, you'll understand:
- ✅ Multi-class classification strategies (OvR, OvO, ECOC)
- ✅ Ensemble methods and their variance-bias tradeoffs
- ✅ Multi-label classification and label dependency modeling
- ✅ Evaluation metrics beyond accuracy
- ✅ Clinical ML application best practices
- ✅ Hyperparameter tuning via cross-validation
- ✅ Feature importance analysis
