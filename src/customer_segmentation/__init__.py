"""Customer segmentation package."""

from .data import FEATURE_COLUMNS, load_customer_data, validate_customer_data
from .modeling import compare_models, select_best_model

__all__ = [
    "FEATURE_COLUMNS",
    "load_customer_data",
    "validate_customer_data",
    "compare_models",
    "select_best_model",
]

__version__ = "1.0.0"

