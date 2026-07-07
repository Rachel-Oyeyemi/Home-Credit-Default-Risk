# Business Recommendations

## Executive summary

The objective of this project is to predict the risk of loan default for applicants of a consumer credit product.  A synthetic sample dataset of 1 000 borrowers was analysed to understand the drivers of default and to build predictive models.  After exploring the data, creating engineered features and comparing multiple algorithms, a **Random Forest** model emerged as the best performer on a held‑out test set (F1‑score ≈ 0.80, recall ≈ 0.84).  The key insights and recommendations below will help credit decision makers balance profitability with risk and ensure fair lending practices.

## Key findings

1. **Income is the strongest negative predictor of default.**  Applicants with higher annual incomes are less likely to default (correlation ≈ –0.36).
2. **Loan amount and annuity are positively associated with default.**  Larger principal balances and higher monthly payments increase the likelihood of non‑repayment (correlation ≈ 0.33 and 0.15 respectively).
3. **Loan duration has little linear relationship with default,** but a cluster of longer‑duration loans (up to 20 months) may warrant special review.
4. **Gender and age are weak predictors** of default risk in this sample, suggesting demographic factors alone should not drive decisions.
5. **Random Forest provides the best balance of recall and precision,** enabling the lender to capture more defaulters while maintaining an acceptable false‑positive rate.

## Business recommendations

1. **Adopt the random forest model for underwriting.**  Integrate the trained model into the credit decision workflow to provide probability‑of‑default scores for each applicant.  Use a risk‑based threshold to approve or refer applications for manual review.
2. **Implement income‑based credit limits.**  Since income is inversely related to default risk, cap loan amounts relative to verified income or incorporate a debt‑to‑income ratio in the approval criteria.
3. **Monitor high annuity ratios.**  Applicants with high monthly payments relative to income should be flagged for further affordability checks or offered lower loan amounts.
4. **Provide explainability outputs.**  Even though ensemble models are less transparent than logistic regression, feature importance scores and SHAP values can help underwriters and regulators understand why a decision was made.
5. **Regularly retrain the model.**  As economic conditions and borrower behaviour change, retrain the model on up‑to‑date data and monitor performance metrics to prevent drift.

## Risk assessment

* **Model bias:** Although gender and age have low correlations with default, biased patterns can still emerge.  Conduct fairness tests (e.g. disparate impact analysis) and ensure that sensitive attributes are not used to discriminate against protected groups.
* **Data quality:** The synthetic sample lacks many important variables (e.g. credit history, employment length).  Real‑world deployment requires a richer dataset and robust data validation.
* **Regulatory compliance:** Credit scoring models must comply with fair lending regulations (Equal Credit Opportunity Act, GDPR).  Maintain documentation, explainability reports and human oversight to satisfy regulators.

## Future opportunities

1. **Incorporate additional data sources** such as credit bureau records, transaction history, and behavioural data to improve predictive power.
2. **Experiment with advanced models** including gradient‑boosted trees with hyperparameter tuning (LightGBM, CatBoost) and deep learning architectures for tabular data.
3. **Deploy a champion–challenger framework** where the current model runs alongside new models in production, allowing performance comparisons on live data before full adoption.
4. **Develop a self‑service tool** for applicants to estimate their approval probability and receive personalised recommendations to improve eligibility.
