"""Translate numeric clusters into readable customer profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS


def _segment_name(row: pd.Series, medians: pd.Series) -> str:
    high_income = row["annual_income_k"] >= medians["annual_income_k"]
    high_spend = row["spending_score"] >= medians["spending_score"]
    high_visits = row["visit_frequency"] >= medians["visit_frequency"]
    if high_income and high_spend:
        return "Premium Champions"
    if high_spend and high_visits:
        return "Engaged Enthusiasts"
    if high_income and not high_spend:
        return "Cautious Affluent"
    if not high_income and not high_spend:
        return "Value Seekers"
    return "Mainstream Customers"


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Return cluster sizes, feature averages, friendly names, and actions."""
    assigned = df.copy()
    assigned["cluster"] = labels
    assigned = assigned[assigned["cluster"] != -1]
    if assigned.empty:
        return pd.DataFrame()

    profiles = assigned.groupby("cluster")[FEATURE_COLUMNS].mean().round(2)
    profiles.insert(0, "customer_count", assigned.groupby("cluster").size())
    profiles["share_percent"] = (profiles["customer_count"] / len(assigned) * 100).round(1)
    medians = assigned[FEATURE_COLUMNS].median()
    profiles["segment_name"] = profiles.apply(_segment_name, axis=1, medians=medians)
    actions = {
        "Premium Champions": "Offer VIP rewards, early access, and premium bundles.",
        "Engaged Enthusiasts": "Use loyalty challenges, referrals, and frequent new offers.",
        "Cautious Affluent": "Build trust with quality evidence and personalized high-value offers.",
        "Value Seekers": "Promote discounts, practical bundles, and price alerts.",
        "Mainstream Customers": "Use broad seasonal campaigns and cross-sell recommendations.",
    }
    profiles["recommended_action"] = profiles["segment_name"].map(actions)
    return profiles.reset_index()

