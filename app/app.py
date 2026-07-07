"""Streamlit application for the Home Credit Default Risk project.

The app provides several pages that allow users to explore the
project overview, inspect model performance, make predictions for
new applicants and read business insights.  It loads the best model
from the ``models`` directory and uses the project utilities for
preprocessing and prediction.  To run locally, execute:

```bash
streamlit run app/app.py
```
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd

from src.download_data import load_raw_data
from src.evaluate_model import evaluate_models
from src.predict import load_best_model, predict_default_risk


def load_metrics() -> dict:
    """Load evaluation metrics from the processed test set.

    Returns
    -------
    dict
        Dictionary of model metrics keyed by model name.
    """
    try:
        return evaluate_models()
    except FileNotFoundError:
        return {}


def home_page():
    st.title("Home Credit Default Risk")
    st.write(
        """
        This app demonstrates an end‑to‑end machine learning pipeline for
        predicting whether a loan applicant will default on a consumer
        credit product.  Use the sidebar to navigate between pages.
        """
    )
    st.markdown(
        """
        **Pages available:**
        - *Project Overview* – learn about the data and objectives.
        - *Prediction* – input applicant information to estimate default risk.
        - *Model Performance* – compare the trained models.
        - *Business Insights* – read recommendations and key findings.
        - *About* – project background and acknowledgements.
        """
    )


def project_overview_page():
    st.header("Project Overview")
    st.write(
        "The goal of this project is to build a credit‑risk model that predicts the probability"
        " that a borrower will default on a loan.  A synthetic dataset of 1 000 applicants was"
        " analysed, containing demographic and loan information.  The pipeline includes EDA,"
        " feature engineering, model training and evaluation, and a Streamlit app for predictions."
    )
    st.subheader("Sample data")
    df = load_raw_data()
    st.write(df.head())
    st.markdown(
        "**Dataset features**\n\n"
        "- **AGE:** applicant age in years\n"
        "- **GENDER:** binary indicator (0 = male, 1 = female)\n"
        "- **INCOME:** annual income in USD\n"
        "- **CREDIT_AMOUNT:** loan principal amount\n"
        "- **ANNUITY:** monthly annuity payment amount\n"
        "- **LOAN_DURATION:** loan duration in months\n"
        "- **TARGET:** 1 = default, 0 = no default"
    )
    st.subheader("Key EDA findings")
    st.write(
        "- Income is negatively correlated with default (–0.36), while larger loan amounts and higher annuities increase default risk.\n"
        "- Age and gender show little correlation with default.\n"
        "- Loan duration is mostly 6 months, with a small set of longer‑duration loans up to 20 months.\n"
        "- The target is moderately imbalanced (≈63 % defaults vs 37 % non‑defaults)."
    )


def prediction_page():
    st.header("Prediction")
    st.write(
        "Enter applicant information below to estimate the probability of loan default using the trained Random Forest model."
    )
    # Input widgets
    age = st.number_input("Age (years)", min_value=18, max_value=80, value=45)
    gender = st.selectbox("Gender", options={0: "Male", 1: "Female"})
    income = st.number_input("Annual income (USD)", min_value=0.0, value=120000.0, step=1000.0)
    credit_amount = st.number_input("Loan amount (USD)", min_value=0.0, value=150000.0, step=1000.0)
    annuity = st.number_input("Annuity (monthly payment, USD)", min_value=0.0, value=8000.0, step=100.0)
    loan_duration = st.number_input("Loan duration (months)", min_value=6, max_value=60, value=6)

    if st.button("Predict default risk"):
        # Prepare feature order consistent with training data and engineered ratios
        # We need to compute the engineered features as the model was trained on scaled ratios.
        credit_to_income = credit_amount / income if income else 0
        annuity_to_income = annuity / income if income else 0
        payment_rate = annuity / credit_amount if credit_amount else 0
        # Create the feature vector in the correct order
        features = [
            age,
            gender,
            income,
            credit_amount,
            annuity,
            loan_duration,
            credit_to_income,
            annuity_to_income,
            payment_rate,
        ]
        model = load_best_model("random_forest_model.pkl")
        pred, prob = predict_default_risk(model, features)
        st.write(f"**Prediction:** {'Default' if pred == 1 else 'No Default'}")
        st.write(f"**Probability of default:** {prob:.2f}")


def model_performance_page():
    st.header("Model Performance")
    st.write("Comparison of trained models on the test set:")
    metrics = load_metrics()
    if not metrics:
        st.info("No evaluation metrics found. Train the models first.")
        return
    # Prepare DataFrame for display
    rows = []
    for model_name, metric_dict in metrics.items():
        row = {
            "Model": model_name,
            "Accuracy": metric_dict.get("accuracy"),
            "Precision": metric_dict.get("precision"),
            "Recall": metric_dict.get("recall"),
            "F1": metric_dict.get("f1"),
            "ROC‑AUC": metric_dict.get("roc_auc"),
        }
        rows.append(row)
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics.set_index("Model"))
    st.write("Confusion matrices:")
    for model_name, metric_dict in metrics.items():
        st.write(f"**{model_name}**")
        cm = metric_dict.get("confusion_matrix")
        cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])
        st.table(cm_df)


def business_insights_page():
    st.header("Business Insights")
    st.write(
        "Based on the exploratory analysis and modelling results, the following insights and recommendations are proposed:\n\n"
        "- **Income matters:** higher earners are much less likely to default, so consider income‑based credit limits.\n"
        "- **Loan size and annuity:** larger loans and higher monthly payments increase risk.  Offer smaller credit amounts or longer durations to reduce the monthly payment burden.\n"
        "- **Ensemble models outperform linear models:** the random forest captures non‑linear patterns and provides the best recall and F1‑score.\n"
        "- **Model explainability:** use feature importance and SHAP values to explain decisions to underwriters and regulators.\n"
        "- **Continuous improvement:** retrain the model with additional features (credit history, employment length, etc.) and evaluate fairness metrics."
    )


def about_page():
    st.header("About")
    st.write(
        "This portfolio project was created to demonstrate an end‑to‑end machine learning pipeline for credit risk analysis."
        " It includes data loading, preprocessing, feature engineering, model training, evaluation, business recommendations"
        " and a Streamlit application.  The repository is intended for educational purposes and does not reflect production"
        " credit‑scoring models."
    )
    st.markdown("\nBuilt by **Rachel Oyeyemi** as part of a Data Analytics & AI portfolio.")


def main():
    st.sidebar.title("Navigation")
    pages = {
        "Home": home_page,
        "Project Overview": project_overview_page,
        "Prediction": prediction_page,
        "Model Performance": model_performance_page,
        "Business Insights": business_insights_page,
        "About": about_page,
    }
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page_func = pages[selection]
    page_func()


if __name__ == "__main__":
    main()
