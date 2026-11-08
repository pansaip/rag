import assert from "node:assert/strict";
import test from "node:test";
import {
  citationNavigationTarget,
  findReliableExcerptMatch,
  normalizeCitationText,
} from "../lib/citations.ts";

const citations = [
  { source_id: "S4", document_id: "doc-4", filename: "four.pdf", page_number: 7, excerpt: "A supporting passage with enough detail." },
  { source_id: "S1", document_id: "doc-1", filename: "one.pdf", page_number: 2, excerpt: "Another passage with enough detail." },
  { source_id: "S4", document_id: "doc-4", filename: "four.pdf", page_number: 7, excerpt: "A supporting passage with enough detail." },
];

test("citation text normalization handles line breaks, hyphenation, and text spanning items", () => {
  const pdfText = "The evidence demon-\nstrates a meaningful result across several experiments.";
  const excerpt = "evidence demonstrates a meaningful\nresult across several experiments.";
  const normalized = normalizeCitationText(pdfText).text;
  assert.equal(normalized, "the evidence demonstrates a meaningful result across several experiments.");
  assert.deepEqual(findReliableExcerptMatch(pdfText, excerpt), { start: 4, end: normalized.length });
  assert.equal(findReliableExcerptMatch(`${pdfText} ${pdfText}`, excerpt), null);
  assert.equal(findReliableExcerptMatch(pdfText, "short text"), null);
});

test("citation navigation targets the cited document and source page", () => {
  assert.deepEqual(citationNavigationTarget(citations[1], "request-1"), {
    requestId: "request-1",
    documentId: "doc-1",
    pageNumber: 2,
    excerpt: "Another passage with enough detail.",
  });
});
