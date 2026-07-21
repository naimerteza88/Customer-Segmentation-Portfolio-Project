"""Predict the segment of one new customer from a trained bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict(bundle_path: Path, values: dict[str, float]) -> int:
    bundle = joblib.load(bundle_path)
    row = pd.DataFrame([values], columns=bundle["feature_columns"])
    scaled = bundle["scaler"].transform(row)
    return int(bundle["model"].predict(scaled)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a customer's K-Means segment")
    parser.add_argument("--model", type=Path, default=Path("artifacts/kmeans_bundle.joblib"))
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--income", type=float, required=True, help="Annual income in thousands")
    parser.add_argument("--spending", type=float, required=True, help="Spending score from 1 to 100")
    parser.add_argument("--visits", type=float, required=True, help="Monthly visit frequency")
    parser.add_argument("--online-ratio", type=float, required=True, help="Share of purchases made online")
    args = parser.parse_args()
    values = {
        "age": args.age,
        "annual_income_k": args.income,
        "spending_score": args.spending,
        "visit_frequency": args.visits,
        "online_purchase_ratio": args.online_ratio,
    }
    print(f"Predicted cluster: {predict(args.model, values)}")


if __name__ == "__main__":
    main()

