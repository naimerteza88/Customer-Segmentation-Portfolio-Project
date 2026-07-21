"""Dataset loading, validation, and reproducible demo-data generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "age",
    "annual_income_k",
    "spending_score",
    "visit_frequency",
    "online_purchase_ratio",
]

REQUIRED_COLUMNS = ["customer_id", *FEATURE_COLUMNS]


def generate_demo_customers(n_customers: int = 600, seed: int = 42) -> pd.DataFrame:
    """Create a realistic, deterministic dataset with five hidden personas."""
    if n_customers < 50:
        raise ValueError("n_customers must be at least 50")

    rng = np.random.default_rng(seed)
    personas = [
        # age, income, spend, visits, online ratio
        (29, 38, 78, 16, 0.82),
        (46, 105, 34, 6, 0.45),
        (39, 78, 66, 12, 0.62),
        (58, 48, 28, 4, 0.24),
        (34, 125, 88, 18, 0.73),
    ]
    proportions = np.array([0.22, 0.18, 0.28, 0.17, 0.15])
    counts = np.floor(proportions * n_customers).astype(int)
    counts[0] += n_customers - counts.sum()
    scales = np.array([5.0, 9.0, 7.0, 2.0, 0.08])

    blocks: list[np.ndarray] = []
    channels: list[str] = []
    channel_options = ["Digital", "Email", "Omnichannel", "In-store", "Premium"]
    for count, center, channel in zip(counts, personas, channel_options, strict=True):
        block = rng.normal(loc=np.array(center), scale=scales, size=(count, 5))
        blocks.append(block)
        channels.extend([channel] * count)

    values = np.vstack(blocks)
    permutation = rng.permutation(len(values))
    values = values[permutation]
    channels = np.asarray(channels)[permutation]
    values[:, 0] = np.clip(values[:, 0], 18, 75)
    values[:, 1] = np.clip(values[:, 1], 15, 180)
    values[:, 2] = np.clip(values[:, 2], 1, 100)
    values[:, 3] = np.clip(values[:, 3], 1, 30)
    values[:, 4] = np.clip(values[:, 4], 0, 1)

    df = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    df.insert(0, "customer_id", [f"CUST-{i:04d}" for i in range(1, n_customers + 1)])
    df["preferred_channel"] = channels
    for column in ["age", "annual_income_k", "spending_score", "visit_frequency"]:
        df[column] = df[column].round().astype(int)
    df["online_purchase_ratio"] = df["online_purchase_ratio"].round(2)
    return df


def validate_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and values, returning a safe copy for modeling."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if len(df) < 20:
        raise ValueError("At least 20 customer rows are required")
    if df["customer_id"].duplicated().any():
        raise ValueError("customer_id values must be unique")

    clean = df.copy()
    for column in FEATURE_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns must contain numeric, non-missing values")
    if not np.isfinite(clean[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Feature columns must contain finite values")
    return clean


def load_customer_data(path: str | Path) -> pd.DataFrame:
    """Load and validate a customer CSV file."""
    return validate_customer_data(pd.read_csv(Path(path)))
