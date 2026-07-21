"""Generate the reproducible portfolio dataset."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_segmentation.data import generate_demo_customers  # noqa: E402


def main() -> None:
    destination = ROOT / "data" / "customers.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_customers().to_csv(destination, index=False)
    print(f"Created {destination}")


if __name__ == "__main__":
    main()

