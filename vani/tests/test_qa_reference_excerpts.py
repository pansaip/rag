import unittest
from unittest.mock import patch
from uuid import uuid4

from api import qa


DOCUMENT_ID = str(uuid4())


def chunk(index, text, page=1):
    return {
        "document_id": DOCUMENT_ID,
        "filename": "paper.pdf",
        "page_number": page,
        "chunk_index": index,
        "text": text,
    }


class FakeClient:
    class Responses:
        def create(self, **_kwargs):
            return type("Response", (), {"output_text": "Supported statement [S1]."})()

    responses = Responses()


class ReferenceExcerptTests(unittest.TestCase):
    def test_excerpt_prefers_one_to_three_complete_sentences(self):
        source = chunk(0, "The paper defines a benchmark. It evaluates several tasks. Results improve across domains. A fourth sentence is omitted.")
        excerpt = qa._coherent_source_excerpt(source, [source])
        self.assertEqual(excerpt, "The paper defines a benchmark. It evaluates several tasks. Results improve across domains.…")

    def test_mid_sentence_start_is_reconstructed_from_same_page_overlap(self):
        previous = chunk(0, "The authors study a difficult task and report strong")
        source = chunk(1, "and report strong results across several domains. Another complete sentence follows.")
        excerpt = qa._coherent_source_excerpt(source, [previous, source])
        self.assertTrue(excerpt.startswith("The authors study a difficult task and report strong results"))
        self.assertNotIn("and report strong results", excerpt[excerpt.find("results") + len("results"):])
        self.assertIn("Another complete sentence follows.", excerpt)

    def test_mid_sentence_start_without_neighbor_is_marked_as_truncated(self):
        source = chunk(1, "continuation of an earlier sentence with supporting detail. A complete sentence follows.")
        excerpt = qa._coherent_source_excerpt(source, [source])
        self.assertTrue(excerpt.startswith("…"))
        self.assertIn("A complete sentence follows.", excerpt)

    def test_mid_sentence_end_is_reconstructed_from_same_page_overlap(self):
        source = chunk(0, "The method improves retrieval and supports robust")
        following = chunk(1, "supports robust citation results. The system was evaluated.")
        excerpt = qa._coherent_source_excerpt(source, [source, following])
        self.assertEqual(excerpt, "The method improves retrieval and supports robust citation results.…")
        self.assertEqual(excerpt.count("supports robust"), 1)

    def test_mid_sentence_end_without_neighbor_uses_ellipsis(self):
        source = chunk(0, "The method improves retrieval and supports")
        excerpt = qa._coherent_source_excerpt(source, [source])
        self.assertTrue(excerpt.startswith("…"))
        self.assertTrue(excerpt.endswith("…"))

    def test_no_safe_overlap_does_not_join_adjacent_text(self):
        source = chunk(1, "A complete sentence remains in the selected chunk.")
        unrelated_neighbor = chunk(0, "Unrelated words from a distant fragment that do not overlap.")
        excerpt = qa._coherent_source_excerpt(source, [unrelated_neighbor, source])
        self.assertEqual(excerpt, "…A complete sentence remains in the selected chunk.")
        self.assertNotIn("Unrelated", excerpt)

    def test_response_preserves_page_and_uses_coherent_reference_excerpt(self):
        source = chunk(0, "The authors describe the benchmark. It measures several capabilities.")
        with patch.object(qa, "search_chunks", return_value=[source]), \
                patch.object(qa, "_get_openai_client", return_value=FakeClient()), \
                patch.dict("os.environ", {"GROQ_API_KEY": "mock", "GROQ_MODEL": "mock-model"}):
            result = qa.answer_question("deepsea คืออะไร", DOCUMENT_ID)
        citation = result["citations"][0]
        self.assertEqual(citation["page_number"], 1)
        self.assertEqual(citation["filename"], "paper.pdf")
        self.assertEqual(citation["document_id"], DOCUMENT_ID)
        self.assertEqual(citation["excerpt"], "The authors describe the benchmark. It measures several capabilities.")


if __name__ == "__main__":
    unittest.main()
