"""Model evaluation utilities for the Home Credit default risk project.

This module loads processed test data and trained models from disk,
then computes evaluation metrics on the test set. The metrics are
aligned with those computed during training to enable consistent
comparison. Use this script in notebooks or CLI form to inspect
model performance without retraining.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict

import pandas as pd

from .utils import (
    classification_metrics,
    configure_logging,
    get_project_root,
    split_features_target,
)


def load_model(model_filename: str) -> object:
    """Load a pickled model from the ``models`` directory.

    Parameters
    ----------
    model_filename : str
        Name of the pickle file in the ``models`` directory.

    Returns
    -------
    object
        The deserialized model object.
    """
    root = get_project_root()
    model_path = root / "models" / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate_models() -> Dict[str, Dict[str, float]]:
    """Evaluate all saved models on the processed test dataset.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A mapping from model filename (without extension) to its
        evaluation metrics.
    """
    configure_logging()
    root = get_project_root()
    test_path = root / "data" / "processed" / "test.csv"
    logging.info("Loading processed test data from %s", test_path)
    if not test_path.exists():
        raise FileNotFoundError(
            f"Processed test data not found: {test_path}. Run train_model.py first."
        )
    test_df = pd.read_csv(test_path)
    X_test, y_test = split_features_target(test_df, target_col="TARGET")

    # Determine available models based on files present in models dir
    models_dir = root / "models"
    metrics: Dict[str, Dict[str, float]] = {}
    for model_file in models_dir.glob("*_model.pkl"):
        model_name = model_file.stem
        logging.info("Evaluating model: %s", model_name)
        model = load_model(model_file.name)
        y_pred = model.predict(X_test)
        try:
            # Many classifiers support predict_proba; fallback to decision_function
            y_score = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else model.decision_function(X_test)
            )
        except Exception:
            y_score = None
        metrics[model_name] = classification_metrics(y_test, y_pred, y_score)
    return metrics


if __name__ == "__main__":
    results = evaluate_models()
    for model_name, metrics_dict in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics_dict.items():
            print(f"  {metric_name}: {value}")
