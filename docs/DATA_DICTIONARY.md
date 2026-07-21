# Data dictionary

## Required columns

| Column | Type | Allowed values | Modeling role |
| --- | --- | --- | --- |
| `customer_id` | Text | Unique, non-empty value | Identifier only |
| `age` | Numeric | Recommended: 18–100 | Feature |
| `annual_income_k` | Numeric | Non-negative; currency in thousands | Feature |
| `spending_score` | Numeric | Recommended: 1–100 | Feature |
| `visit_frequency` | Numeric | Non-negative monthly count | Feature |
| `online_purchase_ratio` | Numeric | Recommended: 0–1 | Feature |

## Optional columns

Any additional columns are preserved in exported assignments but are not supplied to the clustering models. The demo dataset includes `preferred_channel` as an example.

## Validation rules

- At least 20 customer rows are required.
- Every `customer_id` must be unique.
- Required feature values must be numeric, finite, and non-missing.
- Column names are case-sensitive.

## Privacy guidance

Do not place names, email addresses, phone numbers, or other direct personal identifiers in a public repository. Use anonymous customer IDs and confirm that any real dataset may legally and ethically be used for analysis.

