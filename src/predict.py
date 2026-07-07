"""Prediction utility for the Home Credit default risk project.

This module exposes a function to load a trained model and predict
default risk for new applicants. It is intended for use by the
Streamlit app and other integration points. The function returns
both the predicted class and the probability of default.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .utils import configure_logging, get_project_root


def load_best_model(model_name: str = "xgboost_model.pkl") -> object:
    """Load the best-performing model from the models directory.

    Parameters
    ----------
    model_name : str, optional
        Name of the pickle file to load. Defaults to
        ``"xgboost_model.pkl"``.

    Returns
    -------
    object
        The deserialized model object.
    """
    root = get_project_root()
    model_path = root / "models" / model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_name}' not found in models directory."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_default_risk(model: object, features: Iterable[float]) -> Tuple[int, float]:
    """Predict default risk for a single applicant.

    Parameters
    ----------
    model : object
        Trained model supporting ``predict`` and ``predict_proba``.
    features : Iterable[float]
        Ordered feature values matching the training feature order.

    Returns
    -------
    Tuple[int, float]
        Predicted class label (0=no default, 1=default) and the
        probability of default.
    """
    configure_logging()
    X = np.array(features).reshape(1, -1)
    pred = model.predict(X)[0]
    try:
        proba = model.predict_proba(X)[0, 1]
    except Exception:
        proba = float("nan")
    logging.debug(
        "Predicted default risk: class=%s probability=%.3f", pred, proba
    )
    return int(pred), float(proba)
