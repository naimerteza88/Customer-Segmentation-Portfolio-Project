import unittest

import numpy as np

from customer_segmentation.data import generate_demo_customers
from customer_segmentation.modeling import compare_models, create_prediction_bundle, select_best_model
from customer_segmentation.profiling import profile_clusters


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_demo_customers(250, seed=42)

    def test_all_models_are_compared(self):
        results, scaler, features = compare_models(self.data, n_clusters=5)
        self.assertEqual(len(results), 4)
        self.assertEqual(features.shape, (250, 5))
        self.assertEqual(len(scaler.mean_), 5)

    def test_best_model_has_valid_score(self):
        results, _, _ = compare_models(self.data, n_clusters=5)
        best = select_best_model(results)
        self.assertTrue(np.isfinite(best.silhouette))
        self.assertGreaterEqual(best.n_clusters, 2)

    def test_prediction_bundle_predicts_one_row(self):
        bundle = create_prediction_bundle(self.data, n_clusters=5)
        row = self.data[bundle["feature_columns"]].iloc[[0]]
        prediction = bundle["model"].predict(bundle["scaler"].transform(row))
        self.assertEqual(prediction.shape, (1,))

    def test_profiles_cover_non_noise_customers(self):
        results, _, _ = compare_models(self.data, n_clusters=5)
        best = select_best_model(results)
        profiles = profile_clusters(self.data, best.labels)
        self.assertEqual(int(profiles["customer_count"].sum()), int((best.labels != -1).sum()))


if __name__ == "__main__":
    unittest.main()

