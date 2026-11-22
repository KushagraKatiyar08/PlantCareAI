import unittest

from prediction_guard import assess_prediction


class PredictionGuardTests(unittest.TestCase):
    def test_empty_result_requires_review(self):
        result = assess_prediction([])
        self.assertTrue(result["needs_review"])
        self.assertIsNone(result["label"])

    def test_low_top_score_requires_review(self):
        result = assess_prediction([
            {"label": "Tomato___Early_blight", "score": 0.41},
            {"label": "Tomato___Late_blight", "score": 0.22},
        ])
        self.assertTrue(result["needs_review"])

    def test_small_margin_requires_review(self):
        result = assess_prediction([
            {"label": "Tomato___Early_blight", "score": 0.72},
            {"label": "Tomato___Late_blight", "score": 0.65},
        ])
        self.assertTrue(result["needs_review"])

    def test_clearer_result_is_still_not_called_confirmed(self):
        result = assess_prediction([
            {"label": "Tomato___Early_blight", "score": 0.83},
            {"label": "Tomato___Late_blight", "score": 0.55},
        ])
        self.assertFalse(result["needs_review"])
        self.assertIn("not a confirmed diagnosis", result["heading"])

    def test_ignores_invalid_rows(self):
        result = assess_prediction([
            {"label": "", "score": 0.90},
            {"label": "bad", "score": "not-a-number"},
            {"label": "Tomato___Early_blight", "score": 0.80},
        ])
        self.assertEqual(result["label"], "Tomato___Early_blight")

    def test_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            assess_prediction([], min_top_score=1.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
