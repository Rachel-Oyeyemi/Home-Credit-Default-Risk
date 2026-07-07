# Home Credit Default Risk

This repository contains an end‑to‑end machine learning project for predicting whether a loan applicant will default on a consumer credit product.  The project follows best practices for data science portfolios: modular, well‑documented code; comprehensive exploratory data analysis; baseline and advanced modelling; business interpretation; and an interactive Streamlit application.

> **Note**: The official Kaggle *Home Credit Default Risk* dataset cannot be downloaded in this environment.  To demonstrate the workflow, a synthetic sample of 1 000 records was generated with similar structure.  The code and pipeline are designed to be easily extended to the full dataset when available.

## Business problem

Lenders need to assess the risk that a borrower will default on their loan.  Approving high‑risk applicants increases losses, while rejecting low‑risk applicants reduces potential revenue.  An accurate default‑risk model helps underwriters make data‑driven decisions and manage portfolio risk.

## Dataset

The synthetic sample includes the following fields:

| Feature           | Description |
|------------------:|-------------|
| **AGE**           | Applicant age in years (20–69) |
| **GENDER**        | Binary indicator for gender (0 = male, 1 = female) |
| **INCOME**        | Annual income (USD) |
| **CREDIT_AMOUNT** | Loan principal amount (USD) |
| **ANNUITY**       | Monthly payment amount (USD) |
| **LOAN_DURATION** | Duration of the loan in months (6–20) |
| **TARGET**        | 1 = default, 0 = no default |

There are no missing values or duplicate records in the sample.  Approximately 63 % of applicants default, so evaluation metrics should account for class imbalance.

## Methodology

1. **Exploratory Data Analysis (EDA)** – Key statistics and correlations were computed for each feature.  Income showed the strongest negative relationship with default (r ≈ –0.36), while larger credit amounts and higher annuities correlated positively with default.  The EDA report is available in [`EDA_REPORT.md`](EDA_REPORT.md).
2. **Preprocessing & feature engineering** – New ratio features (credit‑to‑income, annuity‑to‑income, payment rate) were created, missing values imputed (none in this sample) and numeric variables scaled.  The processed data was split into training and test sets and stored in `data/processed`.
3. **Modelling** – A baseline logistic regression and two advanced models (Random Forest and XGBoost) were trained on the processed data.  Model metrics such as accuracy, precision, recall, F1‑score and ROC‑AUC were computed on the test set.  See [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) and [`MODEL_EVALUATION.md`](MODEL_EVALUATION.md) for detailed results.
4. **Business insights** – The random forest model achieved the highest F1‑score (≈0.80) and recall (≈0.84), making it the recommended choice.  Business recommendations include using income‑based credit limits, monitoring high annuity ratios and adopting explainability tools.  The full recommendations can be found in [`BUSINESS_RECOMMENDATIONS.md`](BUSINESS_RECOMMENDATIONS.md).
5. **Streamlit application** – An interactive web app allows underwriters to enter applicant information and view default risk predictions using the trained model.  The app also displays model performance metrics and summarises business insights.

## How to run

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Rachel-Oyeyemi/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the training pipeline (optional if you wish to retrain the models):

```bash
cd Home-Credit-Default-Risk
python -m src.train_model
```

Launch the Streamlit app:

```bash
streamlit run app/app.py
```

Open your browser to the provided local URL to interact with the prediction interface and visualisations.

## Repository structure

```
Home-Credit-Default-Risk/
├── data/
│   ├── raw/             # raw dataset (synthetic sample)
│   ├── processed/       # train/test splits after preprocessing
│   └── sample_data/     # reserved for additional small samples
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_business_insights.ipynb
│
├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
│
├── models/              # saved pickled models
├── visuals/             # charts and images
├── presentation/        # slide deck (TBD)
├── app/
│   └── app.py           # Streamlit application
│
├── PROJECT_CHARTER.md
├── EDA_REPORT.md
├── MODEL_COMPARISON.md
├── MODEL_EVALUATION.md
├── BUSINESS_RECOMMENDATIONS.md
├── requirements.txt
└── .gitignore
```

## Future improvements

- **Use the full Kaggle dataset.**  Download `application_train.csv` and related files from the Kaggle competition to train models on a larger, more representative sample.
- **Integrate additional data sources,** such as credit bureau scores, employment history or customer transaction data to improve predictive power.
- **Hyperparameter tuning** using cross‑validation, grid search or Bayesian optimisation to further improve model performance.
- **Fairness and bias audits** to ensure the model complies with ethical and regulatory standards.
- **Deployment:** wrap the prediction logic in a REST API or integrate with existing underwriting systems.

## Acknowledgements

This project is inspired by the Kaggle *Home Credit Default Risk* competition.  The synthetic dataset and all code in this repository are provided for educational purposes.
