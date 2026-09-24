import unittest

from app.ingestion.chunking.splitter import chunk_text


class ChunkingTests(unittest.TestCase):
    def test_long_text_is_split_within_limit(self):
        text = "abcdefghij"

        chunks = chunk_text(text, chunk_size=5)

        self.assertEqual(chunks, ["abcde", "fghij"])
        self.assertTrue(all(len(chunk) <= 5 for chunk in chunks))
        self.assertEqual("".join(chunks), text)

    def test_whitespace_only_input_returns_no_chunks(self):
        chunks = chunk_text("   \n  ", chunk_size=5)

        self.assertEqual(chunks, [])

    def test_invalid_chunk_size_raises_error(self):
        for size in (0, -1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    chunk_text("hello", chunk_size=size)