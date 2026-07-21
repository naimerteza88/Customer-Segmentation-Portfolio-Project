# Customer Segmentation Lab

An end-to-end data science project that groups customers by behavior, compares four unsupervised learning algorithms, explains the resulting segments, and turns them into practical marketing actions.

The project includes a reusable Python package, command-line training workflow, interactive Streamlit dashboard, Docker support, automated tests, and continuous integration.

## Business problem

Treating every customer the same can waste marketing budget and reduce engagement. This project helps a business answer three questions:

1. Which customers behave similarly?
2. Which clustering method produces the clearest groups?
3. What action should the business take for each group?

## What the project does

- Validates and prepares customer data.
- Standardizes features so variables with large numeric ranges do not dominate the analysis.
- Compares K-Means, Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models.
- Evaluates clusters with the Silhouette score and Davies–Bouldin index.
- Profiles each cluster with readable customer names and recommended marketing actions.
- Provides an interactive dashboard for model comparison and exploration.
- Exports customer assignments, cluster profiles, and a deployable K-Means model.

## Results

Using the included reproducible dataset of 600 customers:

| Model | Silhouette score ↑ | Davies–Bouldin index ↓ | Clusters | Noise points |
| --- | ---: | ---: | ---: | ---: |
| K-Means | 0.518 | 0.716 | 5 | 0 |
| Hierarchical | 0.517 | 0.716 | 5 | 0 |
| Gaussian Mixture | 0.517 | 0.716 | 5 | 0 |
| DBSCAN | 0.464 | 0.760 | 4 | 7 |

K-Means produced the highest Silhouette score and is used for the deployable prediction bundle. The comparison remains available because a different algorithm may perform better on another dataset.

## Customer features

| Column | Meaning |
| --- | --- |
| `customer_id` | Unique customer identifier |
| `age` | Customer age in years |
| `annual_income_k` | Annual income in thousands |
| `spending_score` | Relative spending activity from 1 to 100 |
| `visit_frequency` | Average visits per month |
| `online_purchase_ratio` | Share of purchases completed online, from 0 to 1 |
| `preferred_channel` | Demonstration-only descriptive channel |

The model uses the five numeric features. `customer_id` is retained for traceability, while `preferred_channel` is available for interpretation.

## Project structure

```text
customer-segmentation/
├── .github/workflows/tests.yml      # Continuous integration
├── artifacts/                       # Generated model and reports
├── data/customers.csv               # Reproducible demonstration data
├── docs/DATA_DICTIONARY.md
├── reports/analysis_summary.md
├── scripts/generate_data.py
├── src/customer_segmentation/
│   ├── data.py                      # Data generation and validation
│   ├── modeling.py                  # Clustering and evaluation
│   ├── predict.py                   # Prediction CLI
│   ├── profiling.py                 # Segment names and actions
│   └── train.py                     # Training CLI
├── tests/                           # Unit tests
├── app.py                           # Streamlit dashboard
├── Dockerfile
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Quick start

### 1. Clone and enter the project

```bash
git clone https://github.com/naimerteza88/Customer-Segmentation-Portfolio-Project.git
cd Customer-Segmentation-Portfolio-Project
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Train from the command line

```bash
python -m pip install -e .
segment-customers --data data/customers.csv --output artifacts --clusters 5
```

This creates:

- `model_comparison.csv`
- `cluster_profiles.csv`
- `customer_assignments.csv`
- `kmeans_bundle.joblib`

Predict a new customer's cluster:

```bash
predict-segment \
  --age 30 \
  --income 72 \
  --spending 84 \
  --visits 16 \
  --online-ratio 0.75
```

## Use a custom dataset

Upload a CSV in the dashboard or pass its path to the training command. It must include the required columns documented above. Extra columns are preserved and do not affect training.

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

## Run with Docker

```bash
docker build -t customer-segmentation .
docker run --rm -p 8501:8501 customer-segmentation
```

Then open `http://localhost:8501`.

## Methodology

1. **Validation:** Check schema, unique customer IDs, row count, missing values, and numeric types.
2. **Preprocessing:** Standardize numeric features with `StandardScaler`.
3. **Modeling:** Fit four clustering approaches with consistent inputs.
4. **Evaluation:** Prefer a high Silhouette score and a low Davies–Bouldin index.
5. **Profiling:** Calculate cluster averages and assign understandable business names.
6. **Delivery:** Expose the analysis through a dashboard, CSV exports, CLI, and Docker image.

## Important limitations

- The included data is synthetic and intended for learning and demonstration.
- Customer segments are descriptive patterns, not facts about an individual.
- Real business deployment requires representative data, privacy review, fairness checks, monitoring, and periodic retraining.
- Marketing decisions should combine model output with domain knowledge and controlled experiments.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Streamlit, Joblib, Docker, GitHub Actions, and `unittest`.

## License

Released under the [MIT License](LICENSE).
