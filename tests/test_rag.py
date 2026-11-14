import unittest
from unittest.mock import patch

from app.rag import assess_evidence


class EvidenceAssessmentTests(unittest.TestCase):
    def test_strong_evidence_uses_best_score_and_reports_average(self):
        hits = [
            {"score": 0.848},
            {"score": 0.525},
            {"score": 0.432},
        ]

        result = assess_evidence(hits, threshold=0.25)

        self.assertEqual(result["level"], "strong")
        self.assertEqual(result["label"], "Strong evidence")
        self.assertEqual(result["score"], 0.848)
        self.assertEqual(result["average_score"], 0.602)
        self.assertEqual(result["supporting_chunks"], 3)

    def test_limited_evidence_below_moderate_band(self):
        result = assess_evidence(
            [{"score": 0.392}, {"score": 0.348}],
            threshold=0.30,
        )

        self.assertEqual(result["level"], "limited")
        self.assertEqual(result["score"], 0.392)
        self.assertEqual(result["supporting_chunks"], 2)

    def test_chunks_below_threshold_are_excluded(self):
        result = assess_evidence(
            [{"score": 0.392}, {"score": 0.20}],
            threshold=0.30,
        )

        self.assertEqual(result["supporting_chunks"], 1)
        self.assertEqual(result["score"], 0.392)

    def test_no_relevant_evidence_returns_insufficient(self):
        result = assess_evidence(
            [{"score": 0.20}],
            threshold=0.25,
        )

        self.assertEqual(result["level"], "insufficient")
        self.assertIsNone(result["score"])
        self.assertEqual(result["supporting_chunks"], 0)

    def test_default_threshold_comes_from_configuration(self):
        with patch("app.rag.MIN_RELEVANCE_SCORE", 0.25):
            result = assess_evidence([{"score": 0.30}])

        self.assertEqual(result["threshold"], 0.25)
        self.assertEqual(result["supporting_chunks"], 1)


if __name__ == "__main__":
    unittest.main()
