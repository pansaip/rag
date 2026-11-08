import assert from "node:assert/strict";
import test from "node:test";
import { deleteDocument, mergeIndexedDocuments, mergeUploadedDocument, saveOriginalPdf } from "../lib/api.ts";

test("merging indexed documents preserves current selection data and local files", () => {
  const localFile = { name: "session.pdf" };
  const current = [
    { document_id: "existing", filename: "session.pdf", chunk_count: 2, pdf_available: true, file: localFile },
    { document_id: "new-upload", filename: "new.pdf", chunk_count: 1, pdf_available: true, status: "stored" },
  ];
  const indexed = [
    { document_id: "existing", filename: "session.pdf", chunk_count: 2, pdf_available: true },
    { document_id: "restored", filename: "restored.pdf", chunk_count: 4, pdf_available: false },
  ];

  const merged = mergeIndexedDocuments(current, indexed);

  assert.deepEqual(merged.map(({ document_id }) => document_id), ["existing", "new-upload", "restored"]);
  assert.equal(merged[0].file, localFile);
  assert.equal(merged[1].status, "stored");
  assert.equal(merged[2].file, undefined);
});

test("an empty indexed list does not clear session documents", () => {
  const current = [{ document_id: "session", filename: "session.pdf", chunk_count: 1, pdf_available: true }];
  assert.deepEqual(mergeIndexedDocuments(current, []), current);
});

test("duplicate upload reuses the existing document without attaching the selected local File", () => {
  const file = new File(["%PDF-1.7"], "copy.pdf", { type: "application/pdf" });
  const current = [{ document_id: "existing-id", filename: "original.pdf", chunk_count: 3, pdf_available: true, file }];
  const merged = mergeUploadedDocument(current, {
    document_id: "existing-id", filename: "original.pdf", status: "reused", page_count: 0, chunk_count: 3, duplicate: true,
  }, file);
  assert.equal(merged.length, 1);
  assert.equal(merged[0].document_id, "existing-id");
  assert.equal(merged[0].pdf_available, true);
  assert.equal(merged[0].file, undefined);
});

test("saving a legacy original uses PUT and does not use the upload endpoint", async () => {
  const originalFetch = globalThis.fetch;
  let request;
  globalThis.fetch = async (url, options) => {
    request = { url: String(url), ...options };
    return new Response(JSON.stringify({ document_id: "legacy-id", pdf_available: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };
  try {
    await saveOriginalPdf("legacy-id", new File(["%PDF-1.7"], "legacy.pdf", { type: "application/pdf" }));
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(request.method, "PUT");
  assert.match(request.url, /\/documents\/legacy-id\/pdf$/);
  assert.ok(request.body instanceof FormData);
});

test("deletion sends only the exact document ID and requires the expected success response", async () => {
  const originalFetch = globalThis.fetch;
  let request;
  globalThis.fetch = async (url, options) => {
    request = { url: String(url), ...options };
    return new Response(JSON.stringify({ document_id: "exact-id", deleted: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };
  try {
    await deleteDocument("exact-id");
  } finally {
    globalThis.fetch = originalFetch;
  }
  assert.equal(request.method, "DELETE");
  assert.match(request.url, /\/documents\/exact-id$/);
});
