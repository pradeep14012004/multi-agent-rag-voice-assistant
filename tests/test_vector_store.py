import unittest
from unittest.mock import Mock, patch

from app.vector_store import add_chunks, _stable_id, search


class VectorStoreTests(unittest.TestCase):
    def test_reindex_deletes_only_matching_filename_and_domain(self):
        chunks = [
            {
                "text": "Updated first chunk.",
                "metadata": {
                    "filename": "guide.txt",
                    "domain": "Healthcare",
                    "page": 1,
                    "chunk_index": 0,
                },
            },
            {
                "text": "Updated second chunk.",
                "metadata": {
                    "filename": "guide.txt",
                    "domain": "Healthcare",
                    "page": 1,
                    "chunk_index": 1,
                },
            },
        ]
        model = Mock()
        model.encode.return_value.tolist.return_value = [[0.1], [0.2]]
        collection = Mock()

        with (
            patch("app.vector_store.get_model", return_value=model),
            patch("app.vector_store.get_collection", return_value=collection),
        ):
            result = add_chunks(chunks)

        self.assertEqual(result, 2)
        collection.delete.assert_called_once_with(where={
            "$and": [
                {"filename": "guide.txt"},
                {"domain": "Healthcare"},
            ],
        })
        collection.upsert.assert_called_once()

    def test_stable_ids_differ_between_domains(self):
        item = {
            "text": "Same content.",
            "metadata": {
                "filename": "guide.txt",
                "domain": "Healthcare",
                "page": 1,
                "chunk_index": 0,
            },
        }
        other_domain_item = {
            **item,
            "metadata": {**item["metadata"], "domain": "Engineering"},
        }

        self.assertNotEqual(_stable_id(item), _stable_id(other_domain_item))

    def test_search_retrieves_all_candidates_and_deduplicates_per_domain(self):
        model = Mock()
        model.encode.return_value.tolist.return_value = [[0.5]]
        collection = Mock()
        collection.count.return_value = 8
        collection.query.return_value = {
            "documents": [[
                " Same chunk ",
                "same   chunk",
                "Same chunk",
                "Below threshold",
            ]],
            "metadatas": [[
                {"domain": "Healthcare"},
                {"domain": "Healthcare"},
                {"domain": "Engineering"},
                {"domain": "Healthcare"},
            ]],
            "distances": [[0.1, 0.2, 0.3, 0.8]],
        }

        with (
            patch("app.vector_store.get_model", return_value=model),
            patch("app.vector_store.get_collection", return_value=collection),
        ):
            hits = search(" query ", top_k=5, domain="Healthcare", min_score=0.25)

        self.assertEqual(len(hits), 2)
        self.assertEqual(
            [hit["metadata"]["domain"] for hit in hits],
            ["Healthcare", "Engineering"],
        )
        self.assertAlmostEqual(hits[0]["score"], 0.9)
        collection.query.assert_called_once_with(
            query_embeddings=[[0.5]],
            n_results=8,
            include=["documents", "metadatas", "distances"],
            where={"domain": "Healthcare"},
        )


if __name__ == "__main__":
    unittest.main()
