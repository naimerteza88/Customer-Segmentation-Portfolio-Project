"""Interactive Streamlit dashboard for customer segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from customer_segmentation.data import load_customer_data, validate_customer_data  # noqa: E402
from customer_segmentation.modeling import compare_models, results_to_frame, select_best_model  # noqa: E402
from customer_segmentation.profiling import profile_clusters  # noqa: E402

st.set_page_config(page_title="Customer Segmentation Lab", page_icon="🎯", layout="wide")

st.title("Customer Segmentation Lab")
st.caption("Discover customer groups, compare clustering algorithms, and turn patterns into marketing actions.")

with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Upload customer CSV", type="csv")
    n_clusters = st.slider("Target clusters", min_value=2, max_value=8, value=5)
    dbscan_eps = st.slider("DBSCAN radius (eps)", 0.2, 2.0, 0.8, 0.05)
    dbscan_min = st.slider("DBSCAN minimum samples", 3, 20, 8)
    st.markdown("The included demo data is used when no file is uploaded.")

try:
    if uploaded is None:
        customers = load_customer_data(ROOT / "data" / "customers.csv")
    else:
        customers = validate_customer_data(pd.read_csv(uploaded))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

results, _, _ = compare_models(
    customers,
    n_clusters=n_clusters,
    dbscan_eps=dbscan_eps,
    dbscan_min_samples=dbscan_min,
)
best = select_best_model(results)
comparison = results_to_frame(results)
profiles = profile_clusters(customers, best.labels)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{len(customers):,}")
col2.metric("Best model", best.name)
col3.metric("Silhouette score", f"{best.silhouette:.3f}")
col4.metric("Clusters found", best.n_clusters)

st.subheader("Model comparison")
st.dataframe(
    comparison.style.format({"silhouette_score": "{:.3f}", "davies_bouldin_index": "{:.3f}"}),
    width="stretch",
    hide_index=True,
)

assigned = customers.copy()
assigned["cluster"] = best.labels.astype(str)
left, right = st.columns([1.35, 1])
with left:
    st.subheader("Income and spending map")
    st.scatter_chart(
        assigned,
        x="annual_income_k",
        y="spending_score",
        color="cluster",
        size="visit_frequency",
        width="stretch",
    )
with right:
    st.subheader("Cluster distribution")
    counts = assigned[assigned["cluster"] != "-1"]["cluster"].value_counts().sort_index()
    st.bar_chart(counts, x_label="Cluster", y_label="Customers", width="stretch")

st.subheader("Customer profiles and recommended actions")
st.dataframe(profiles, width="stretch", hide_index=True)

st.subheader("Explore assignments")
cluster_filter = st.multiselect(
    "Show clusters", sorted(assigned["cluster"].unique()), default=sorted(assigned["cluster"].unique())
)
filtered = assigned[assigned["cluster"].isin(cluster_filter)]
st.dataframe(filtered, width="stretch", hide_index=True)
st.download_button(
    "Download customer assignments",
    filtered.to_csv(index=False),
    file_name="customer_segments.csv",
    mime="text/csv",
)
