import unittest
from app.ingestion import chunk_text, extract_file, prepare_documents

class IngestionTests(unittest.TestCase):
    def test_chunk_overlap_and_content(self):
        text = " ".join([f"sentence{i}." for i in range(250)])
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertIn("sentence0", chunks[0])
        self.assertIn("sentence249", chunks[-1])

    def test_text_extraction_and_metadata(self):
        raw = b"Heap is a tree-based data structure."
        pages = extract_file("heap.txt", raw)
        self.assertEqual(pages[0]["page"], 1)
        chunks = prepare_documents("heap.txt", raw, "Computer Science")
        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["metadata"]["domain"], "Computer Science")
        self.assertEqual(chunks[0]["metadata"]["filename"], "heap.txt")

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("   \n  "), [])

if __name__ == "__main__":
    unittest.main()
