"""Feature engineering functions for the Home Credit project.

This module defines reusable transformations for creating new
features from the raw input variables. Good feature engineering
improves model performance by highlighting important relationships
between variables and capturing domain knowledge. If additional
information (e.g. bureau balance, payment history) is available,
extend these functions accordingly.

The synthetic dataset bundled with this project contains basic
demographic and loan information. We derive ratios such as
``credit_to_income`` and ``annuity_to_income`` to help models
capture affordability signals. A ``payment_rate`` feature is also
computed as the annuity divided by credit amount, which is a
common variable in credit risk modelling.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the Home Credit dataset.

    The following new columns are added:

    * ``CREDIT_TO_INCOME``: ratio of credit amount to applicant income
    * ``ANNUITY_TO_INCOME``: ratio of annuity to income
    * ``PAYMENT_RATE``: annuity divided by credit amount

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with columns ``CREDIT_AMOUNT``, ``INCOME``,
        and ``ANNUITY``.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features. The original
        DataFrame is not modified in place.
    """
    df = df.copy()
    # Avoid division by zero by replacing zeros with NaN; these will
    # subsequently be imputed in preprocessing.
    income = df["INCOME"].replace({0: np.nan})
    credit = df["CREDIT_AMOUNT"].replace({0: np.nan})
    annuity = df["ANNUITY"].replace({0: np.nan})

    df["CREDIT_TO_INCOME"] = credit / income
    df["ANNUITY_TO_INCOME"] = annuity / income
    df["PAYMENT_RATE"] = annuity / credit

    logging.debug(
        "Added feature columns CREDIT_TO_INCOME, ANNUITY_TO_INCOME, PAYMENT_RATE"
    )
    return df
