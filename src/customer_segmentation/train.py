"""Command-line training workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from .data import load_customer_data
from .modeling import compare_models, create_prediction_bundle, results_to_frame, select_best_model
from .profiling import profile_clusters


def train(data_path: Path, output_dir: Path, n_clusters: int = 5) -> None:
    df = load_customer_data(data_path)
    results, _, _ = compare_models(df, n_clusters=n_clusters)
    best = select_best_model(results)
    comparison = results_to_frame(results)
    profiles = profile_clusters(df, best.labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    assignments = df.copy()
    assignments["cluster"] = best.labels
    assignments.to_csv(output_dir / "customer_assignments.csv", index=False)

    bundle = create_prediction_bundle(df, n_clusters=n_clusters)
    joblib.dump(bundle, output_dir / "kmeans_bundle.joblib")

    print(comparison.to_string(index=False))
    print(f"\nBest analytical model: {best.name}")
    print(f"Artifacts written to: {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and compare customer segmentation models")
    parser.add_argument("--data", type=Path, default=Path("data/customers.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--clusters", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args.data, args.output, args.clusters)


if __name__ == "__main__":
    main()

