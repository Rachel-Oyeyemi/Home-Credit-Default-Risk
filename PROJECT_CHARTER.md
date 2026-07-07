# Project Charter – Home Credit Default Risk

## Business problem

Financial institutions need to decide whether to approve or reject consumer loan applications.  Approving high‑risk applicants leads to increased defaults and financial loss, while rejecting low‑risk customers reduces revenue and damages the customer relationship.  The **Home Credit Default Risk** challenge seeks to build a model that predicts the probability of a borrower defaulting on a loan so lenders can make more informed, data‑driven decisions.

## Project objectives

1. **Develop predictive models** that estimate the probability of loan default using applicant demographics and loan attributes.
2. **Minimise false negatives** (rejecting credit‑worthy borrowers) and false positives (approving high‑risk borrowers) by selecting appropriate evaluation metrics and thresholds.
3. **Deliver business insights** about which factors most strongly influence default risk, enabling stakeholders to refine credit policies and pricing strategies.
4. **Provide an interactive tool** (Streamlit app) for underwriters to input applicant information and view predicted default probabilities and recommended actions.

## Stakeholders

| Role                | Interest                                                                                           |
|--------------------|----------------------------------------------------------------------------------------------------|
| **Credit risk team**     | Build and validate credit‑scoring models, manage portfolio risk.                                 |
| **Underwriters**        | Assess individual applications and make final lending decisions using model outputs.             |
| **Product managers**    | Balance approval rates with delinquency rates; design pricing tiers for different risk segments. |
| **Compliance**          | Ensure models adhere to fair lending regulations and explainability requirements.                |
| **Customers**           | Receive timely and fair credit decisions.                                                        |

## Success metrics

- **Model performance:** high recall and ROC‑AUC on a held‑out test set (>0.80), with precision maintained above 0.60.
- **Business impact:** reduction in default rate while maintaining or increasing approval rate compared with current rules‑based approach.
- **User adoption:** underwriters actively use the Streamlit app in pilot testing and provide positive feedback on usability and interpretability.
- **Compliance:** models pass fairness and explainability reviews and meet regulatory requirements.

## Expected business impact

Implementing an accurate default risk model allows the lender to:

- Reduce credit losses by avoiding high‑risk loans.
- Increase revenue by approving more credit‑worthy applicants who might otherwise be rejected.
- Optimise interest rates based on predicted risk, improving competitiveness.
- Improve operational efficiency by automating parts of the underwriting process and focusing human review on borderline cases.

## Technical architecture

The project will be implemented in Python and organised in a modular structure:

- **Data ingestion:** A synthetic sample dataset is loaded from the repository; functions to download the full Kaggle dataset can be added for production use.
- **Preprocessing:** Missing value imputation, feature engineering (ratios and payment rate), scaling and train/test splitting.
- **Modelling:** Baseline logistic regression and advanced ensemble models (Random Forest and XGBoost) trained using `scikit‑learn` and `xgboost`.
- **Evaluation:** Metrics such as accuracy, precision, recall, F1‑score and ROC‑AUC computed on a held‑out test set.
- **Model persistence:** Trained models are saved as pickled files in a `models/` directory.
- **Application:** A Streamlit front‑end loads the best model to generate predictions and displays charts and performance metrics.

## End‑to‑end workflow

1. **Data acquisition:** Load the raw dataset from `data/raw`.  If the full Kaggle dataset becomes available, provide a download function.
2. **Exploratory data analysis:** Generate descriptive statistics, visualise distributions, investigate correlations and identify outliers.  Summarise findings in `EDA_REPORT.md`.
3. **Preprocessing & feature engineering:** Create additional ratio features; handle missing values; scale numeric variables; split into training and test sets and save them to `data/processed`.
4. **Model training:** Train baseline and advanced models on the processed training set; tune hyperparameters via cross‑validation; compare performance.
5. **Model evaluation:** Evaluate models on the test set; generate a model comparison report and select the best performer.
6. **Business insights:** Interpret feature importances and model outputs to derive actionable recommendations; document them in `BUSINESS_RECOMMENDATIONS.md`.
7. **Deployment:** Build a Streamlit app that enables users to input applicant data, view default probability predictions and inspect model performance.
8. **Documentation & presentation:** Write a comprehensive README, project charter, EDA report, model comparison and evaluation reports; create an executive slide deck summarising the project.
