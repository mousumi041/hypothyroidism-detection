# Hypothyroidism Detection

A machine learning pipeline that predicts the likelihood of a patient having hypothyroidism, using a real-world clinical dataset from the Garavan Institute. Multiple classification models are trained, tuned, and compared to identify the best-performing approach.

## Problem

Given a dataset of patient clinical attributes (hormone levels, symptoms, treatment history), classify whether a patient has hypothyroidism (`binaryClass`: P/N). The dataset contains a mix of categorical and continuous features with a substantial amount of missing data, making preprocessing and outlier handling a core part of the problem.

## Dataset

- Source: [UCI Machine Learning Repository — Thyroid Disease](https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease) (Garavan Institute, Sydney)
- 3,772 patient records, 30 attributes (mostly boolean/categorical, plus 6 continuous lab values: age, TSH, T3, TT4, T4U, FTI)
- Target variable: `binaryClass` (P = hypothyroid, N = not hypothyroid)

## Approach

1. **Data exploration** — inspected structure, types, and summary statistics across all 30 columns.
2. **Cleaning** — replaced `?` placeholders with NaN, converted boolean `t`/`f` columns to 1/0, coerced numeric columns to proper dtypes, and encoded the `binaryClass` target as 1/0.
3. **Missing value treatment** — dropped fully-empty columns (e.g. `TBG`), then imputed numerical columns with the mean and categorical columns with the mode.
4. **Exploratory data analysis** — boxplots, histograms, density plots, scatterplots, a pairplot, and a correlation heatmap across the six numerical lab-value columns to understand distributions and relationships.
5. **Outlier detection** — compared four methods (IQR, Z-score, Modified Z-score, DBSCAN) per column, and selected the most appropriate method for each feature based on its distribution shape (e.g. IQR for skewed features like TSH, Z-score for more normally-distributed features like T4U).
6. **Feature engineering** — one-hot encoded categorical variables (`sex`, `referral source`), then standard-scaled numerical features.
7. **Multicollinearity reduction** — detected feature pairs with correlation > 0.7 (e.g. TT4 & FTI) and dropped redundant features.
8. **Model building & evaluation** — trained and compared five classifiers using 5-fold cross-validation: Logistic Regression, Decision Tree, SVM, Neural Network (MLP), and Random Forest, evaluated on accuracy, precision, recall, F1, and ROC AUC.
9. **Hyperparameter tuning** — Grid Search on Random Forest (`n_estimators`, `max_depth`, `min_samples_split`).
10. **Feature importance** — extracted and visualized feature importances from the best model.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9626 | 0.9618 | 0.9626 | 0.9582 | 0.9743 |
| **Decision Tree** | **0.9947** | **0.9948** | **0.9947** | **0.9948** | 0.9846 |
| Support Vector Machine | 0.9477 | 0.9490 | 0.9477 | 0.9358 | 0.9834 |
| Neural Network | 0.9817 | 0.9814 | 0.9817 | 0.9813 | 0.9789 |
| Random Forest | 0.9939 | 0.9941 | 0.9939 | 0.9940 | **0.9993** |

**Decision Tree** was the best-performing model overall on cross-validated accuracy/F1, with **Random Forest** close behind and showing the strongest ROC AUC (0.9993). TSH, TT4, and FTI emerged as the most important features for prediction.

## Tech Stack

Python · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · SciPy

## Project Structure

```
.
├── hypothyroid_classification.csv    # dataset
├── analysis.ipynb                     # full pipeline: EDA, cleaning, outlier treatment, modeling
├── ProjectReport.pdf                  # detailed written report of findings and insights
├── requirements.txt
└── README.md
```

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure `hypothyroid_classification.csv` is in the same directory as the notebook (the code reads it via `pd.read_csv("hypothyroid_classification.csv")`).
3. Run the notebook top to bottom.

## Notes

- Outlier treatment methods were deliberately chosen per-column based on each feature's distribution shape rather than applying one method uniformly — see the notebook's outlier detection section for the full reasoning per feature.
- The full written analysis with detailed insights and inferences is available in `ProjectReport.pdf`.
