"""Clustering model comparison and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLUMNS, validate_customer_data

MODEL_NAMES = ("K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture")


@dataclass
class ModelResult:
    name: str
    estimator: Any
    labels: np.ndarray
    silhouette: float
    davies_bouldin: float
    n_clusters: int
    noise_points: int


def _evaluate(features: np.ndarray, labels: np.ndarray) -> tuple[float, float, int, int]:
    labels = np.asarray(labels)
    mask = labels != -1
    usable_labels = labels[mask]
    unique = np.unique(usable_labels)
    n_clusters = len(unique)
    noise_points = int((~mask).sum())
    if n_clusters < 2 or len(usable_labels) <= n_clusters:
        return float("nan"), float("nan"), n_clusters, noise_points
    return (
        float(silhouette_score(features[mask], usable_labels)),
        float(davies_bouldin_score(features[mask], usable_labels)),
        n_clusters,
        noise_points,
    )


def compare_models(
    df: pd.DataFrame,
    n_clusters: int = 5,
    dbscan_eps: float = 0.8,
    dbscan_min_samples: int = 8,
    random_state: int = 42,
) -> tuple[list[ModelResult], StandardScaler, np.ndarray]:
    """Fit four clustering approaches on the same standardized features."""
    clean = validate_customer_data(df)
    scaler = StandardScaler()
    features = scaler.fit_transform(clean[FEATURE_COLUMNS])

    estimators: list[tuple[str, Any]] = [
        ("K-Means", KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)),
        ("Hierarchical", AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")),
        ("DBSCAN", DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)),
        (
            "Gaussian Mixture",
            GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=random_state),
        ),
    ]

    results: list[ModelResult] = []
    for name, estimator in estimators:
        if hasattr(estimator, "fit_predict"):
            labels = estimator.fit_predict(features)
        else:
            estimator.fit(features)
            labels = estimator.predict(features)
        silhouette, db_index, found_clusters, noise = _evaluate(features, labels)
        results.append(
            ModelResult(
                name=name,
                estimator=estimator,
                labels=np.asarray(labels, dtype=int),
                silhouette=silhouette,
                davies_bouldin=db_index,
                n_clusters=found_clusters,
                noise_points=noise,
            )
        )
    return results, scaler, features


def results_to_frame(results: list[ModelResult]) -> pd.DataFrame:
    """Convert model results into a presentation-friendly comparison table."""
    rows = [
        {
            "model": result.name,
            "silhouette_score": result.silhouette,
            "davies_bouldin_index": result.davies_bouldin,
            "clusters_found": result.n_clusters,
            "noise_points": result.noise_points,
        }
        for result in results
    ]
    return pd.DataFrame(rows).sort_values(
        ["silhouette_score", "davies_bouldin_index"], ascending=[False, True], na_position="last"
    )


def select_best_model(results: list[ModelResult]) -> ModelResult:
    """Choose the valid model with the highest Silhouette score."""
    valid = [result for result in results if np.isfinite(result.silhouette)]
    if not valid:
        raise ValueError("No model produced at least two valid clusters")
    return max(valid, key=lambda result: (result.silhouette, -result.davies_bouldin))


def create_prediction_bundle(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> dict:
    """Train a deployable K-Means model and package its preprocessing state."""
    clean = validate_customer_data(df)
    scaler = StandardScaler()
    features = scaler.fit_transform(clean[FEATURE_COLUMNS])
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = model.fit_predict(features)
    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "training_labels": labels,
        "version": "1.0.0",
    }

