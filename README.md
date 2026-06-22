# student-dropout-prediction
# Predicting Student Dropout Using Classical Machine Learning and Deep Learning

Summative Assignment — Machine Learning Module

A comparative machine learning pipeline for predicting student dropout, built using the UCI Predict Students' Dropout and Academic Success dataset (4,424 records, 36 features). This project is motivated by a mission to cultivate reading engagement among young people in Africa through technology — specifically, by enabling early identification of learners at risk of disengaging before dropout becomes irreversible.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Model_Training_and_Evaluation_Summative.ipynb` | Main notebook — fully executed with all cell outputs |
| `Summative_Report.pdf` | IEEE-style academic report (3,500–5,000 words) |
| `student_dropout_data.csv` | Dataset cached locally from UCI on first run |

---

## Project Overview

The notebook builds and compares eleven machine learning experiments across two paradigms — classical ML and deep learning — to determine which approach best predicts student dropout from structured tabular data. Each experiment is systematically varied, documented with a stated hypothesis and trade-off analysis, and evaluated on a fully held-out test set. The best model is selected programmatically by test-set AUC rather than assumed in advance.

Key preprocessing decisions include: StandardScaler fitted on the training set only (no data leakage), SMOTE oversampling applied to the training set only, and a stratified 70/15/15 train/validation/test split. Four engineered features were added — grade progression, 1st and 2nd semester approval rates, and a financial stress composite — which proved among the strongest predictors in the final model.

---

## Experiments

### Classical ML (E1–E5)

| Exp | Model | Key Setting |
|-----|-------|-------------|
| E1 | Logistic Regression | C=1.0, L2 regularization |
| E2 | Logistic Regression | C=0.1, stronger regularization |
| E3 | SVM | RBF kernel, C=1.0 |
| E4 | Random Forest | 100 trees, no depth limit |
| E5 | Random Forest | 200 trees, max_depth=10 — best overall model |

### Deep Learning (E6–E11)

| Exp | Model | Key Change |
|-----|-------|------------|
| E6 | FNN Baseline | 2 layers [64, 32], lr=0.001 |
| E7 | FNN Deeper | 4 layers [128, 64, 32, 16] |
| E8 | FNN + Dropout | Dropout(0.3) — best deep learning model |
| E9 | FNN + BatchNorm + Dropout | Batch Normalization added |
| E10 | FNN Fine-Tuned | lr=0.0005, ReduceLROnPlateau |
| E11 | FNN Functional API + tf.data | Keras Functional API rebuild with tf.data pipeline |

---

## Key Results

| Model | Test Accuracy | Test F1 | Test AUC |
|-------|--------------|---------|----------|
| E5: Random Forest (tuned) | 0.8855 | 0.8241 | 0.9392 |
| E8: FNN + Dropout | 0.8720 | 0.8064 | 0.9338 |
| E1: Logistic Regression | 0.8780 | 0.8204 | 0.9341 |
| E3: SVM RBF | 0.8735 | 0.8065 | 0.9236 |

The tuned Random Forest (E5) achieved the best performance across all eleven experiments. No deep learning variant surpassed it on held-out test AUC — consistent with Grinsztajn et al. (2022), who found that tree-based ensembles remain competitive with or superior to deep learning on small-to-medium tabular datasets. The best deep learning model, E8 (FNN + Dropout), came within 0.0054 AUC points of Random Forest but never exceeded it.

---

## How to Run

1. Open `Model_Training_and_Evaluation_Summative.ipynb` in Google Colab
2. Set the runtime to T4 GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells top-to-bottom (Runtime → Run all)
4. The dataset is loaded automatically from UCI on first run and cached as `student_dropout_data.csv`. If UCI is unavailable, the notebook falls back to the cached file automatically.

All experiments use SEED = 42 for full reproducibility. The notebook runs clean from top to bottom with no manual steps required.

---

## Dependencies

All installed automatically in Cell 1 of the notebook:

- tensorflow
- scikit-learn
- imbalanced-learn
- pandas
- numpy
- matplotlib
- seaborn

---

## Dataset

Realinho, V., Machado, J., Baptista, L., and Martins, M. V. (2022). Predicting student dropout and academic success. Data, 7(11), 146. https://doi.org/10.3390/data7110146

Available at: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
