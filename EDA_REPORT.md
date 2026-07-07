# Exploratory Data Analysis Report

This exploratory data analysis (EDA) examines the **Home Credit Default Risk** dataset.  Because the Kaggle competition data is not available in this environment, a **synthetic sample** of 1 000 loan applicants was generated.  The sample contains seven columns: six input features and a binary target indicating whether the applicant subsequently defaulted on their loan.

## Dataset summary

- **Row count:** 1 000
- **Column count:** 7
- **Target variable:** `TARGET` (1 = default, 0 = no default)
- **Feature columns:**
  1. `AGE` – applicant age in years (integer)
  2. `GENDER` – binary indicator for gender (0 = male, 1 = female)
  3. `INCOME` – annual income in USD (float)
  4. `CREDIT_AMOUNT` – loan principal amount (float)
  5. `ANNUITY` – monthly annuity payment amount (float)
  6. `LOAN_DURATION` – loan duration in months (integer)

### Data types, missing values and duplicates

| Column          | Data type | Missing values | Duplicate rows |
|-----------------|-----------|---------------|---------------|
| AGE             | integer   | 0             | 0             |
| GENDER          | integer   | 0             | 0             |
| INCOME          | float     | 0             | 0             |
| CREDIT_AMOUNT   | float     | 0             | 0             |
| ANNUITY         | float     | 0             | 0             |
| LOAN_DURATION   | integer   | 0             | 0             |
| TARGET          | integer   | 0             | 0             |

There are **no missing values** and **no duplicate rows** in the sample.

### Target distribution

The binary target `TARGET` indicates whether the applicant eventually defaulted on their loan.  In the sample, ~63.3 % of applicants are labelled `1` (defaults) and 36.7 % are labelled `0` (non‑defaults).  This moderate imbalance suggests that metrics beyond accuracy—such as recall, precision and the F1‑score—will be important when evaluating models.

## Feature statistics and outliers

The table below summarises key descriptive statistics for each numeric column.  Outliers were identified using the inter‑quartile range (IQR) method (values outside 1.5 × IQR).  Only `INCOME`, `CREDIT_AMOUNT`, `ANNUITY` and `LOAN_DURATION` contain a small number of outlier observations.

| Feature           | Mean       | Median     | Std dev   | Min    | Max      | Outliers |
|------------------:|-----------:|-----------:|----------:|-------:|---------:|---------:|
| **AGE**           | 44.83      | 45.00      | 14.35     | 20     | 69       | 0        |
| **GENDER**        | 0.474      | 0          | 0.50      | 0      | 1        | 0        |
| **INCOME**        | 122 127.49 | 121 950.50 | 48 870.75 | 20 000 | 275 243  | 1        |
| **CREDIT_AMOUNT** | 154 594.65 | 157 070.50 | 60 175.25 | 10 000 | 359 753  | 1        |
| **ANNUITY**       | 7 867.79   | 7 429.82   | 5 197.68  | 1 000  | 23 597.29| 1        |
| **LOAN_DURATION** | 6.45       | 6.00       | 1.66      | 6      | 20       | 105      |
| **TARGET**        | 0.633      | 1.00       | 0.48      | 0      | 1        | 0        |

**Observations:**

- **Age** follows a roughly uniform distribution between 20 and 69 with no extreme outliers.
- **Income**, **credit amount** and **annuity** are right‑skewed (long tails to higher values).  Each contains one high‑value outlier.
- **Loan duration** is mostly concentrated at 6 months (the lower bound), with a minority of applicants taking loans up to 20 months; the higher durations contribute to a larger number of outliers.
- **Gender** is evenly distributed (47.4 % labelled 1), suggesting no imbalance.

## Correlation analysis

The Pearson correlation coefficients between features and the target are shown below.  Values closer to ±1 indicate stronger linear relationships.

| Feature           | Correlation with `TARGET` |
|------------------:|---------------------------:|
| **AGE**           | 0.020 |
| **GENDER**        | 0.041 |
| **INCOME**        | –0.356 |
| **CREDIT_AMOUNT** | 0.326 |
| **ANNUITY**       | 0.145 |
| **LOAN_DURATION** | 0.041 |

**Interpretation:**

- **Income** is negatively correlated with default risk (–0.36), meaning higher earners are less likely to default.
- **Credit amount** is moderately positively correlated (0.33).  Larger loans correspond to higher default risk.
- **Annuity** shows a weak positive correlation (0.15).  A larger monthly payment relative to income might indicate greater strain on the borrower.
- **Age**, **gender** and **loan duration** show very weak correlations (|ρ| < 0.05), suggesting limited direct effect on default risk.

## Business context

This dataset represents a simplified version of the **Home Credit Default Risk** challenge, where the objective is to predict whether a loan applicant will repay their loan on time.  Lenders want to maximise approvals for credit‑worthy customers while minimising losses from defaults.  Key features such as **income**, **loan size** and **monthly payments** capture an applicant’s ability to repay.  Understanding the distribution and relationships of these variables helps define modelling strategies and informs risk‑based pricing or lending decisions.

## Recommended modelling approach

Given the binary target and moderate class imbalance, this problem is best treated as a **supervised classification** task.  A simple **logistic regression** provides an interpretable baseline.  To capture non‑linear relationships and interactions between features, tree‑based ensemble methods such as **Random Forest** and **Gradient Boosting (XGBoost)** are recommended as advanced models.

**Evaluation metrics:**  Accuracy alone may be misleading due to the cost of misclassifying defaulters.  Important metrics include:

- **Recall (Sensitivity):** proportion of defaulters correctly identified.
- **Precision:** proportion of predicted defaulters who actually default.
- **F1‑score:** harmonic mean of precision and recall.
- **ROC‑AUC:** ability to distinguish between default and non‑default across thresholds.

These metrics will guide model selection and hyperparameter tuning in subsequent phases.
