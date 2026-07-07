"""Model training pipeline for the Home Credit default risk project.

This script trains both baseline and advanced machine learning models
to predict the probability of loan default. It handles data loading,
preprocessing, model training, evaluation and model persistence. The
pipeline is designed to be run as a script but exposes its main
functionality in reusable functions to allow unit testing and
integration with notebooks or a Streamlit app.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .download_data import load_raw_data
from .preprocess import preprocess_data, split_and_save
from .utils import (
    classification_metrics,
    configure_logging,
    get_project_root,
    split_features_target,
)


def train_and_evaluate(
    test_size: float = 0.2, random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """Train baseline and advanced models on the Home Credit dataset.

    The function performs the following steps:

      1. Loads the raw dataset.
      2. Applies preprocessing (feature engineering, imputation,
         scaling).
      3. Splits the data into training and test sets and saves them
         under ``data/processed``.
      4. Trains a logistic regression model as a baseline.
      5. Trains a random forest classifier.
      6. Trains an XGBoost classifier.
      7. Evaluates each model on the held-out test set and returns
         performance metrics.
      8. Persists each trained model to the ``models`` directory.

    Parameters
    ----------
    test_size : float, optional
        Fraction of the dataset to reserve for testing. Defaults to
        ``0.2``.
    random_state : int, optional
        Random seed for reproducibility. Defaults to ``42``.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary keyed by model name with nested metric dictionaries.
    """
    configure_logging()
    root = get_project_root()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and preprocess data
    logging.info("Loading and preprocessing raw data")
    df_raw = load_raw_data()
    df_processed = preprocess_data(df_raw)
    train_df, test_df = split_and_save(df_processed, test_size=test_size, random_state=random_state)

    X_train, y_train = split_features_target(train_df, target_col="TARGET")
    X_test, y_test = split_features_target(test_df, target_col="TARGET")

    metrics: Dict[str, Dict[str, float]] = {}

    # Step 4: Baseline model - Logistic Regression
    logging.info("Training Logistic Regression baseline model")
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_score_lr = lr_model.predict_proba(X_test)[:, 1]
    metrics["logistic_regression"] = classification_metrics(y_test, y_pred_lr, y_score_lr)
    # Persist model
    with open(models_dir / "logistic_regression_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    logging.info("Saved Logistic Regression model")

    # Step 5: Advanced model - Random Forest
    logging.info("Training Random Forest model")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=random_state, class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_score_rf = rf_model.predict_proba(X_test)[:, 1]
    metrics["random_forest"] = classification_metrics(y_test, y_pred_rf, y_score_rf)
    with open(models_dir / "random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    logging.info("Saved Random Forest model")

    # Step 6: Advanced model - XGBoost
    logging.info("Training XGBoost model")
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        scale_pos_weight=1.0,
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_score_xgb = xgb_model.predict_proba(X_test)[:, 1]
    metrics["xgboost"] = classification_metrics(y_test, y_pred_xgb, y_score_xgb)
    with open(models_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    logging.info("Saved XGBoost model")

    logging.info("Training complete. Returning metrics.")
    return metrics


if __name__ == "__main__":
    # When executed as a script, train models and print metrics
    results = train_and_evaluate()
    for model_name, metrics_dict in results.items():
        logging.info("%s metrics:", model_name)
        for metric_name, value in metrics_dict.items():
            logging.info("  %s: %s", metric_name, value)
