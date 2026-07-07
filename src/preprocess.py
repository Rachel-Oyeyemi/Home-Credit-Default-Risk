"""Data preprocessing and train/test splitting for the Home Credit project.

This module handles common data cleaning tasks, such as imputing
missing values, scaling numeric features, encoding categorical
variables, and splitting the dataset into training and testing sets.
Preprocessing is encapsulated in reusable functions to make the
pipeline reproducible and maintainable.

Example
-------

>>> from download_data import load_raw_data
>>> from preprocess import preprocess_data, split_and_save
>>> df = load_raw_data()
>>> df_processed = preprocess_data(df)
>>> split_and_save(df_processed, test_size=0.2)

The processed train and test CSVs will be stored in ``data/processed``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import configure_logging, get_project_root, split_features_target
from .feature_engineering import add_feature_columns


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic preprocessing on the raw dataset.

    Steps include:
      * Adding engineered features via ``add_feature_columns``.
      * Imputing missing values (numeric columns with median; categorical
        columns with mode).
      * Scaling numeric features with ``StandardScaler``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as loaded from ``download_data.load_raw_data``.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for modelling.
    """
    configure_logging()
    logging.info("Starting preprocessing on DataFrame with shape %s", df.shape)

    # Add engineered features
    df = add_feature_columns(df.copy())

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Impute missing values
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median = df[col].median()
            logging.info("Imputing numeric column %s with median value %.3f", col, median)
            df[col].fillna(median, inplace=True)
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            mode = df[col].mode().iloc[0]
            logging.info("Imputing categorical column %s with mode value %s", col, mode)
            df[col].fillna(mode, inplace=True)

    # Separate features and target for scaling
    X, y = split_features_target(df, target_col="TARGET")

    # Scale numeric features (including engineered numeric features)
    scaler = StandardScaler()
    X_scaled = X.copy()
    numeric_feature_cols = X_scaled.select_dtypes(include=[np.number]).columns
    logging.info("Scaling numeric feature columns: %s", list(numeric_feature_cols))
    X_scaled.loc[:, numeric_feature_cols] = scaler.fit_transform(X_scaled[numeric_feature_cols])

    # Recombine scaled features with target
    df_processed = pd.concat([X_scaled, y], axis=1)
    logging.info("Completed preprocessing; result shape %s", df_processed.shape)
    return df_processed


def split_and_save(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test subsets and save to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame containing features and target.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Defaults
        to ``0.2``.
    random_state : int, optional
        Random seed for reproducibility. Defaults to ``42``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The training and testing DataFrames.
    """
    configure_logging()
    logging.info("Splitting DataFrame with shape %s into train/test", df.shape)
    X, y = split_features_target(df, target_col="TARGET")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save to data/processed
    root = get_project_root()
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    logging.info("Saving train dataset to %s", train_path)
    train_df.to_csv(train_path, index=False)
    logging.info("Saving test dataset to %s", test_path)
    test_df.to_csv(test_path, index=False)

    logging.info(
        "Train/Test split complete: train shape %s, test shape %s",
        train_df.shape,
        test_df.shape,
    )
    return train_df, test_df
