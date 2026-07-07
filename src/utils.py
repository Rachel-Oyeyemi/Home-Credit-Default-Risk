"""Shared utility functions for the Home Credit Default Risk project.

These helpers centralise common functionality such as loading CSV files,
splitting a DataFrame into features/target, configuring logging and
calculating classification metrics. Keeping these utilities in one module
helps ensure consistency across the preprocessing, training and
evaluation scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_project_root() -> Path:
    """Return the absolute path to the project root.

    The project root is defined as the parent directory of the `src`
    package. Using this helper allows other modules to construct
    filepaths relative to the repository without hardcoding locations.
    """
    return Path(__file__).resolve().parents[1]


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a basic logging format for the project.

    Parameters
    ----------
    level : int, optional
        Logging level, by default `logging.INFO`.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=level,
    )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file from disk with basic error handling.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the CSV contents.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the provided path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    logging.info("Loading data from %s", path)
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame, target_col: str = "TARGET"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix and target vector.

    If the specified `target_col` does not exist, the function attempts
    to infer a plausible target column based on common names (e.g.
    'target', 'default', 'label').

    Parameters
    ----------
    df : pd.DataFrame
        The full DataFrame including features and target.
    target_col : str, optional
        Name of the target column, by default "TARGET".

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the features DataFrame and the target
        Series.

    Raises
    ------
    KeyError
        If the target column cannot be found or inferred.
    """
    if target_col not in df.columns:
        # attempt to infer target column using common names
        candidates = [c for c in df.columns if c.lower() in {"target", "default", "label", "y"}]
        if candidates:
            target_col = candidates[-1]
        else:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    return df.drop(columns=[target_col]), df[target_col]


def classification_metrics(
    y_true: Any, y_pred: Any, y_score: Any | None = None
) -> Dict[str, Any]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted class labels.
    y_score : array-like | None
        Optional probability scores for the positive class. When
        provided, ROC-AUC is computed.

    Returns
    -------
    Dict[str, Any]
        Dictionary of computed metrics.
    """
    output: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        output["roc_auc"] = roc_auc_score(y_true, y_score)
    return output
