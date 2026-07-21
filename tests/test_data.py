import unittest

import pandas as pd

from customer_segmentation.data import FEATURE_COLUMNS, generate_demo_customers, validate_customer_data


class DataTests(unittest.TestCase):
    def test_generation_is_reproducible(self):
        first = generate_demo_customers(100, seed=7)
        second = generate_demo_customers(100, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_generated_data_has_expected_schema(self):
        data = generate_demo_customers(100)
        self.assertTrue({"customer_id", *FEATURE_COLUMNS}.issubset(data.columns))
        self.assertEqual(data["customer_id"].nunique(), 100)

    def test_missing_feature_is_rejected(self):
        data = generate_demo_customers(100).drop(columns=[FEATURE_COLUMNS[0]])
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            validate_customer_data(data)


if __name__ == "__main__":
    unittest.main()

