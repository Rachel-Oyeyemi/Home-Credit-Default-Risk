"""Data download and loading utilities for the Home Credit Default Risk project.

This module centralises the logic for retrieving the raw dataset used
throughout the pipeline. Because direct access to the Kaggle
competition data is restricted, the project includes a synthetic
sample dataset in ``data/raw/home_credit_sample.csv``. The helper
functions here load that CSV into a pandas DataFrame and provide
basic logging and error handling. When working with the full Kaggle
dataset, you can extend these functions to handle API authentication
and downloading.

Example
-------

>>> from download_data import load_raw_data
>>> df = load_raw_data()
>>> print(df.head())

The sample dataset contains anonymised applicant information and a
binary target column ``TARGET`` indicating default risk. See the
project's README for a detailed description of each field.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import get_project_root, configure_logging


def load_raw_data(filename: str = "home_credit_sample.csv") -> pd.DataFrame:
    """Load the raw Home Credit dataset from the data directory.

    Parameters
    ----------
    filename : str, optional
        Name of the CSV file in ``data/raw`` to load. Defaults to
        ``"home_credit_sample.csv"``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw dataset.

    Raises
    ------
    FileNotFoundError
        If the specified file cannot be found in ``data/raw``.
    """
    configure_logging()  # ensure logging is configured for any downstream use
    root = get_project_root()
    raw_path = root / "data" / "raw" / filename
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. Please ensure the file exists or adjust the filename."
        )
    logging.info("Loading raw dataset from %s", raw_path)
    return pd.read_csv(raw_path)


def save_dataframe(df: pd.DataFrame, dest: Path | str) -> None:
    """Save a DataFrame to a CSV file, creating parent directories if needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    dest : Path | str
        Destination path relative to the project root or an absolute
        path. Parent directories are created automatically.
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Saving DataFrame to %s", dest_path)
    df.to_csv(dest_path, index=False)


def download_kaggle_dataset(kaggle_competition: str, dataset_file: str, dest: Optional[Path] = None) -> Path:
    """Placeholder for downloading the Kaggle Home Credit dataset.

    Kaggle's API requires authentication via a JSON token placed in
    ``~/.kaggle/kaggle.json``. In environments where network access
    and authentication are not available (such as this portfolio
    demonstration), this function raises a ``NotImplementedError``. If
    you wish to use the full Kaggle dataset locally, install the
    ``kaggle`` Python package, authenticate with your Kaggle API
    token, and call ``kaggle competitions download``. Then extract
    ``application_train.csv`` into your ``data/raw`` directory.

    Parameters
    ----------
    kaggle_competition : str
        Name of the Kaggle competition (e.g. ``"home-credit-default-risk"``).
    dataset_file : str
        Filename to download (e.g. ``"application_train.csv"``).
    dest : Path, optional
        Destination directory to save the downloaded file. Defaults to
        ``data/raw`` under the project root.

    Returns
    -------
    Path
        Path to the downloaded file.

    Raises
    ------
    NotImplementedError
        Always in this demonstration environment.
    """
    raise NotImplementedError(
        "Downloading the Kaggle dataset is not implemented in this environment."
    )
