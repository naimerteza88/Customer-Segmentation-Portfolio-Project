# Analysis summary

## Executive finding

K-Means created the clearest separation on the included 600-customer dataset, with a Silhouette score of **0.518** and a Davies–Bouldin index of **0.716**. Hierarchical Clustering and Gaussian Mixture Models produced nearly identical results, while DBSCAN identified four clusters and seven possible noise points.

## Model comparison

| Model | Silhouette score | Davies–Bouldin index | Clusters | Noise points |
| --- | ---: | ---: | ---: | ---: |
| K-Means | 0.517545 | 0.715673 | 5 | 0 |
| Hierarchical | 0.517248 | 0.715662 | 5 | 0 |
| Gaussian Mixture | 0.517248 | 0.715662 | 5 | 0 |
| DBSCAN | 0.463740 | 0.759516 | 4 | 7 |

## Interpretation

- **Silhouette score:** Higher values indicate that customers are closer to their own cluster than to neighboring clusters.
- **Davies–Bouldin index:** Lower values indicate compact clusters that are well separated from one another.
- **Noise points:** DBSCAN may mark isolated customers as `-1` instead of forcing them into a group.

## Business use

The profiles can support targeted campaigns, loyalty design, product bundling, channel selection, and retention experiments. These recommendations should be tested through controlled campaigns before broad use.

## Reproducibility

The demo dataset uses a fixed random seed. Model training also uses a fixed random state where the algorithm supports it. Run `make data`, `make train`, and `make test` to reproduce the analysis.

