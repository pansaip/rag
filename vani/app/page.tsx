"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState, type ChangeEvent, type FormEvent, type ReactNode } from "react";
import {
  askQuestion,
  documentPdfUrl,
  deleteDocument,
  listDocuments,
  mergeIndexedDocuments,
  mergeUploadedDocument,
  saveOriginalPdf,
  uploadDocument,
  type AnswerCitation,
  type SessionDocument,
} from "@/lib/api";
import { citationNavigationTarget } from "@/lib/citations";
import { AnswerPresentation } from "@/lib/answer_presentation";

const PdfViewer = dynamic(() => import("./PdfViewer"), {
  ssr: false,
  loading: () => <div className="flex min-h-72 flex-1 items-center justify-center rounded-2xl border border-[#e5e8e2] bg-[#eef0ec] text-sm text-[#777a73]">Preparing PDF viewer…</div>,
});

type ChatTurn = {
  id: string;
  question: string;
  answer?: string;
  citations?: AnswerCitation[];
  loading?: boolean;
  error?: string;
};

function Icon({ name, className = "" }: { name: string; className?: string }) {
  const paths: Record<string, ReactNode> = {
    brand: <><path d="M12 3.5c-4.5 0-7.5 3.2-7.5 7.6 0 4 2.5 7.4 6.1 8.9" /><path d="M12 3.5c4.5 0 7.5 3.2 7.5 7.6 0 4-2.5 7.4-6.1 8.9" /><path d="M8.3 11.2c1.7.1 3 1.4 3.7 3.1.7-1.7 2-3 3.7-3.1M12 14.3v6.2" /></>,
    upload: <><path d="M12 16V4" /><path d="m7 9 5-5 5 5" /><path d="M5 15v4h14v-4" /></>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6M8 13h8M8 17h6" /></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></>,
    alert: <><circle cx="12" cy="12" r="9" /><path d="M12 8v4M12 16h.01" /></>,
    sparkle: <><path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z" /><path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z" /></>,
  };
  return <svg aria-hidden="true" className={className} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">{paths[name] ?? paths.file}</svg>;
}

function isPdf(file: File): boolean {
  return file.name.toLowerCase().endsWith(".pdf") && (!file.type || file.type === "application/pdf");
}

export default function Home() {
  const uploadInput = useRef<HTMLInputElement>(null);
  const originalInput = useRef<HTMLInputElement>(null);
  const [documents, setDocuments] = useState<SessionDocument[]>([]);
  const [documentListLoading, setDocumentListLoading] = useState(true);
  const [documentListError, setDocumentListError] = useState("");
  const [documentListAttempt, setDocumentListAttempt] = useState(0);
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [selectionRestored, setSelectionRestored] = useState(false);
  const [hasSavedSelection, setHasSavedSelection] = useState(false);
  const [documentListLoaded, setDocumentListLoaded] = useState(false);
  const [pendingPreview, setPendingPreview] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number | null>(0);
  const [uploadError, setUploadError] = useState("");
  const [uploadNotice, setUploadNotice] = useState("");
  const [previewError, setPreviewError] = useState("");
  const [savingPdfFor, setSavingPdfFor] = useState("");
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState("");
  const [manageDocumentsOpen, setManageDocumentsOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<SessionDocument | null>(null);
  const [deleteConfirmed, setDeleteConfirmed] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState("");
  const [deleteError, setDeleteError] = useState("");
  const [citationTarget, setCitationTarget] = useState<ReturnType<typeof citationNavigationTarget> | null>(null);

  useEffect(() => {
    try {
      const savedSelection = window.localStorage.getItem("vani.selectedDocumentId");
      setHasSavedSelection(savedSelection !== null);
      if (savedSelection) setSelectedDocumentId(savedSelection);
    } catch {
      setHasSavedSelection(false);
    } finally {
      setSelectionRestored(true);
    }
  }, []);

  useEffect(() => {
    if (!selectionRestored) return;
    try {
      window.localStorage.setItem("vani.selectedDocumentId", selectedDocumentId);
    } catch {
      // The workspace still works if browser storage is unavailable.
    }
  }, [selectionRestored, selectedDocumentId]);

  useEffect(() => {
    let active = true;
    setDocumentListLoading(true);
    setDocumentListError("");
    listDocuments()
      .then((indexed) => {
        if (active) setDocuments((current) => mergeIndexedDocuments(current, indexed));
      })
      .catch(() => {
        if (active) setDocumentListError("Could not load indexed documents. Session uploads remain available.");
      })
      .finally(() => {
        if (active) {
          setDocumentListLoading(false);
          setDocumentListLoaded(true);
        }
      });
    return () => { active = false; };
  }, [documentListAttempt]);

  useEffect(() => {
    if (!selectionRestored || !documentListLoaded || documents.length === 0) return;
    const firstAvailable = documents.find((document) => document.pdf_available) ?? documents[0];
    if (!hasSavedSelection && !selectedDocumentId) {
      setSelectedDocumentId(firstAvailable.document_id);
    } else if (hasSavedSelection && selectedDocumentId && !documents.some((document) => document.document_id === selectedDocumentId)) {
      setSelectedDocumentId(firstAvailable.document_id);
    }
  }, [selectionRestored, documentListLoaded, documents, hasSavedSelection, selectedDocumentId]);

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!isPdf(file)) {
      setUploadError("Choose a PDF file to continue.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setUploadError("The PDF must be 10 MB or smaller.");
      return;
    }

    setUploadError("");
    setUploadNotice("");
    setPreviewError("");
    setPendingPreview(file);
    setSelectedDocumentId("");
    setCitationTarget(null);
    setUploading(true);
    setUploadProgress(0);
    try {
      const uploaded = await uploadDocument(file, setUploadProgress);
      if (uploaded.duplicate) {
        setDocuments((current) => mergeUploadedDocument(current, uploaded, file));
        setUploadNotice("Already in library. Opened the existing indexed document without re-indexing it.");
      } else {
        setDocuments((current) => mergeUploadedDocument(current, uploaded, file));
      }
      setSelectedDocumentId(uploaded.document_id);
      setPendingPreview(null);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  }

  async function handleAttachOriginal(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const target = documents.find((document) => document.document_id === selectedDocumentId);
    if (!target) return;
    if (!isPdf(file)) {
      setPreviewError("Choose the original PDF file to preview this document.");
      return;
    }
    if (file.name.trim().toLocaleLowerCase() !== target.filename.trim().toLocaleLowerCase()) {
      setPreviewError(`Choose the original PDF named “${target.filename}”. This file was not uploaded.`);
      return;
    }
    setPreviewError("");
    setDocuments((current) => current.map((document) => document.document_id === target.document_id ? { ...document, file } : document));
    setSavingPdfFor(target.document_id);
    try {
      await saveOriginalPdf(target.document_id, file);
      setDocuments((current) => current.map((document) => document.document_id === target.document_id ? { ...document, file, pdf_available: true } : document));
    } catch (error) {
      setPreviewError(error instanceof Error ? `${error.message} The local preview is still available.` : "Could not save the original PDF. The local preview is still available.");
    } finally {
      setSavingPdfFor("");
    }
  }

  async function handleDeleteDocument() {
    if (!deleteTarget || !deleteConfirmed || deletingDocumentId) return;
    const documentId = deleteTarget.document_id;
    setDeletingDocumentId(documentId);
    setDeleteError("");
    try {
      await deleteDocument(documentId);
      setDocuments((current) => current.filter((document) => document.document_id !== documentId));
      if (selectedDocumentId === documentId) {
        setSelectedDocumentId("");
        setPendingPreview(null);
        setPreviewError("");
      }
      if (citationTarget?.documentId === documentId) setCitationTarget(null);
      setDeleteTarget(null);
      setDeleteConfirmed(false);
      try {
        const indexed = await listDocuments();
        setDocuments((current) => mergeIndexedDocuments(current, indexed).filter((document) => document.document_id !== documentId));
      } catch {
        setDocumentListError("The document was deleted, but the indexed-document list could not refresh. Retry to update it.");
      }
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : "Document deletion failed. Retry the same document ID.");
    } finally {
      setDeletingDocumentId("");
    }
  }

  function navigateToCitation(citation: AnswerCitation) {
    setSelectedDocumentId(citation.document_id);
    setHasSavedSelection(true);
    setPendingPreview(null);
    setPreviewError("");
    setCitationTarget(citationNavigationTarget(citation, crypto.randomUUID()));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = question.trim();
    if (!text || sending) return;

    const id = crypto.randomUUID();
    setQuestion("");
    setChatError("");
    setSending(true);
    setTurns((current) => [...current, { id, question: text, loading: true }]);
    try {
      const result = await askQuestion(text, selectedDocumentId || undefined);
      setTurns((current) => current.map((turn) => turn.id === id ? { ...turn, loading: false, answer: result.answer, citations: result.citations } : turn));
    } catch (error) {
      const message = error instanceof Error ? error.message : "The request failed. Please try again.";
      setChatError(message);
      setTurns((current) => current.map((turn) => turn.id === id ? { ...turn, loading: false, error: message } : turn));
    } finally {
      setSending(false);
    }
  }

  const selectedDocument = documents.find((document) => document.document_id === selectedDocumentId);
  const citationPdfUrl = citationTarget?.documentId === selectedDocumentId && selectedDocument?.pdf_available
    ? documentPdfUrl(selectedDocument.document_id)
    : null;
  const activeFile = citationPdfUrl
    ?? selectedDocument?.file
    ?? (selectedDocument?.pdf_available ? documentPdfUrl(selectedDocument.document_id) : null)
    ?? (!selectedDocument ? pendingPreview : null);

  return (
    <main className="research-workspace">
      <section className="pdf-pane">
        <header className="pdf-pane-header">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-[#1f4b37] text-white"><Icon name="brand" className="size-5" /></div>
            <div className="min-w-0"><p className="text-sm font-semibold tracking-[-0.02em]">vani</p><p className="text-[11px] text-[#858781]">Research workspace</p></div>
          </div>
          <div className="flex min-w-0 flex-1 items-center justify-end gap-2">
            <select
              aria-label="Select document"
              value={selectedDocumentId}
              onChange={(event) => { setSelectedDocumentId(event.target.value); setHasSavedSelection(true); setPendingPreview(null); setPreviewError(""); setCitationTarget(null); }}
              className="min-w-0 max-w-[55%] rounded-lg border border-[#e1e5de] bg-white px-2.5 py-2 text-xs text-[#4a4e47] outline-none focus:border-[#91aa95] sm:max-w-[260px]"
            >
              <option value="">All documents</option>
              {documents.map((document) => <option key={document.document_id} value={document.document_id}>{document.filename}</option>)}
            </select>
            <button type="button" onClick={() => uploadInput.current?.click()} disabled={uploading} className="flex shrink-0 items-center gap-2 rounded-lg bg-[#244a36] px-3 py-2 text-xs font-medium text-white transition hover:bg-[#183b29] disabled:opacity-60">
              <Icon name="upload" className="size-4" /><span className="hidden sm:inline">{uploading ? "Uploading…" : "Upload PDF"}</span><span className="sm:hidden">{uploading ? "…" : "Upload"}</span>
            </button>
            <button type="button" onClick={() => setManageDocumentsOpen(true)} className="shrink-0 rounded-lg border border-[#d6ddd4] bg-white px-2.5 py-2 text-xs font-medium text-[#42604a] hover:bg-[#f5f8f4]">Manage</button>
            <input ref={uploadInput} type="file" accept="application/pdf,.pdf" className="hidden" onChange={handleUpload} />
          </div>
        </header>

        {documentListLoading && documents.length === 0 && <p role="status" className="px-5 pt-2 text-xs text-[#858a82]">Loading indexed documents…</p>}
        {documentListError && <div role="status" className="mx-4 mt-2 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-[#f0e2c9] bg-[#fffaf0] px-3 py-2 text-xs text-[#79633c] sm:mx-5"><span>{documentListError}</span><button type="button" onClick={() => setDocumentListAttempt((attempt) => attempt + 1)} className="font-semibold underline underline-offset-2">Retry</button></div>}

        {(uploading || uploadError || uploadNotice || previewError) && <div className="space-y-2 px-4 pt-3 sm:px-5">
          {uploading && <div aria-live="polite" className="flex items-center gap-3 rounded-lg bg-white px-3 py-2 text-xs text-[#777a73]"><span className="size-3.5 animate-spin rounded-full border-2 border-[#d7e2d7] border-t-[#4c7957]" />Uploading and indexing PDF{uploadProgress !== null ? ` · ${uploadProgress}%` : "…"}</div>}
          {uploadError && <p role="alert" className="flex gap-2 rounded-lg bg-[#fff4f1] px-3 py-2 text-xs leading-5 text-[#a64735]"><Icon name="alert" className="mt-0.5 size-4 shrink-0" />{uploadError}</p>}
          {uploadNotice && <p role="status" className="rounded-lg border border-[#dce8db] bg-[#f3f8f2] px-3 py-2 text-xs leading-5 text-[#3f6748]">{uploadNotice}</p>}
          {previewError && <p role="alert" className="flex gap-2 rounded-lg bg-[#fff9f7] px-3 py-2 text-xs leading-5 text-[#a64735]"><Icon name="alert" className="mt-0.5 size-4 shrink-0" />{previewError}</p>}
        </div>}

        {selectedDocument && !selectedDocument.pdf_available && <div className="mx-4 mt-3 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#dce5dc] bg-white px-4 py-3 sm:mx-5">
          <div><p className="text-sm font-medium text-[#3e483f]">Save original PDF</p><p className="mt-0.5 text-xs text-[#858a82]">Choose “{selectedDocument.filename}” to save it for this indexed document. It will not be re-indexed.</p></div>
          <button type="button" disabled={savingPdfFor === selectedDocument.document_id} onClick={() => originalInput.current?.click()} className="shrink-0 rounded-lg border border-[#cfdacf] px-3 py-2 text-xs font-medium text-[#315a3d] hover:bg-[#f4f8f4] disabled:opacity-60">{savingPdfFor === selectedDocument.document_id ? "Saving PDF…" : "Save original PDF"}</button>
          <input ref={originalInput} type="file" accept="application/pdf,.pdf" className="hidden" onChange={handleAttachOriginal} />
        </div>}

        <div className="pdf-viewer-wrap">
          <PdfViewer file={activeFile} documentId={selectedDocumentId} navigationTarget={citationTarget} />
        </div>
        <p className="px-4 pb-3 text-center text-[10px] text-[#989b94] sm:px-5">Saved PDFs reopen automatically when you return to this workspace.</p>
      </section>

      <section className="chat-pane">
        <header className="chat-pane-header">
          <div className="min-w-0"><p className="text-sm font-semibold text-[#30332f]">AI Chat</p><p className="mt-0.5 truncate text-xs text-[#92948e]">{selectedDocument ? `Using ${selectedDocument.filename}` : "Ask across all uploaded documents"}</p></div>
        </header>

        <section className="chat-conversation" aria-label="AI chat conversation">
          <div className="mx-auto w-full max-w-3xl">
            {turns.length === 0 ? (
              <div className="flex min-h-[36vh] flex-col items-center justify-center py-10 text-center sm:min-h-[48vh]">
                <div className="mb-5 flex size-14 items-center justify-center rounded-2xl border border-[#e6ebe4] bg-white text-[#477354] shadow-sm"><Icon name="sparkle" className="size-6" /></div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[#6b8b70]">Vani · Vanilla RAG</p>
                <h1 className="text-2xl font-semibold tracking-[-0.04em] text-[#282a27] sm:text-3xl">Find answers in your files.</h1>
                <p className="mt-3 max-w-md text-sm leading-6 text-[#858780]">Upload a PDF, ask in Thai or English, and review the passages retrieved from your documents.</p>
                <div className="mt-7 flex flex-wrap justify-center gap-2 text-xs text-[#777b73]"><span className="rounded-full border border-[#e6e7e1] bg-white px-3 py-1.5">PDF up to 10 MB</span><span className="rounded-full border border-[#e6e7e1] bg-white px-3 py-1.5">Page citations</span></div>
              </div>
            ) : <div className="space-y-8">
              {turns.map((turn) => <article key={turn.id} className="space-y-5">
                <div className="flex justify-end"><p className="max-w-[88%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-[#244a36] px-4 py-3 text-sm leading-6 text-white">{turn.question}</p></div>
                <div className="space-y-3">
                <div className="flex items-center gap-2 text-xs font-semibold text-[#4d6652]"><Icon name="sparkle" className="size-4" />AI answer · grounded in retrieved sources</div>
                  {turn.loading && <div className="flex items-center gap-3 rounded-xl border border-[#e8e9e3] bg-white px-4 py-4 text-sm text-[#777a73]"><span className="size-4 animate-spin rounded-full border-2 border-[#d7e2d7] border-t-[#4c7957]" />Retrieving sources and preparing an answer…</div>}
                  {turn.error && <div role="alert" className="flex gap-2 rounded-xl border border-[#f0d9d2] bg-[#fff9f7] px-4 py-3 text-sm leading-6 text-[#a64735]"><Icon name="alert" className="mt-1 size-4 shrink-0" />{turn.error}</div>}
                  {turn.answer && <div className="rounded-xl border border-[#e7e8e2] bg-white p-4 shadow-[0_2px_10px_rgba(24,35,25,0.025)] sm:p-5">
                    <AnswerPresentation answer={turn.answer} citations={turn.citations ?? []} onCitationClick={navigateToCitation} />
                  </div>}
                </div>
              </article>)}
              <div aria-live="polite" className="h-1" />
            </div>}
          </div>
        </section>

        <div className="chat-composer">
          <div className="mx-auto max-w-3xl">
            {chatError && <p role="alert" className="mb-2 flex items-center gap-2 text-xs text-[#a64735]"><Icon name="alert" className="size-4" />{chatError}</p>}
            <form onSubmit={handleSubmit} className="flex items-end gap-2 rounded-2xl border border-[#dedfd8] bg-white p-2 shadow-[0_4px_18px_rgba(30,40,30,0.05)] transition focus-within:border-[#93ad98] focus-within:ring-2 focus-within:ring-[#dce9de]">
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} rows={1} placeholder="Ask a question in Thai or English…" aria-label="Question" className="max-h-28 min-h-10 flex-1 resize-y bg-transparent px-3 py-2.5 text-sm leading-5 text-[#30332f] outline-none placeholder:text-[#a3a59e]" />
              <button type="submit" disabled={!question.trim() || sending} aria-label="Get AI answer" className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-[#244a36] text-white transition hover:bg-[#183b29] disabled:cursor-not-allowed disabled:bg-[#d9ded8]">{sending ? <span className="size-4 animate-spin rounded-full border-2 border-white/40 border-t-white" /> : <Icon name="send" className="size-[18px]" />}</button>
            </form>
            <p className="mt-2 text-center text-[10px] text-[#a0a29c]">Answers use retrieved context and include source citations. Press Enter to send · Shift + Enter for a new line.</p>
          </div>
        </div>
      </section>
      {manageDocumentsOpen && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-3 sm:p-6" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !deleteTarget) setManageDocumentsOpen(false); }}>
        <section role="dialog" aria-modal="true" aria-labelledby="manage-documents-title" className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-2xl bg-white p-4 shadow-2xl sm:p-6">
          <div className="flex items-start justify-between gap-4"><div><h2 id="manage-documents-title" className="text-lg font-semibold text-[#30332f]">Manage Documents</h2><p className="mt-1 text-xs text-[#858a82]">Deleting a document removes its indexed chunks and saved PDF.</p></div><button type="button" aria-label="Close document manager" disabled={Boolean(deleteTarget)} onClick={() => setManageDocumentsOpen(false)} className="rounded-lg px-3 py-1.5 text-sm text-[#666a62] hover:bg-[#f3f4f0] disabled:opacity-40">Close</button></div>
          {documentListLoading && documents.length === 0 && <p role="status" className="py-6 text-sm text-[#777a73]">Loading indexed documents…</p>}
          {!documentListLoading && documents.length === 0 && <p className="py-6 text-sm text-[#777a73]">No indexed documents are available.</p>}
          <div className="mt-4 space-y-3">{documents.map((document) => <article key={document.document_id} className="rounded-xl border border-[#e5e8e2] p-3 sm:p-4"><div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start"><div className="min-w-0 space-y-1.5"><h3 className="break-words text-sm font-semibold text-[#363a35]">{document.filename}</h3><p className="break-all font-mono text-[11px] text-[#777a73]">{document.document_id}</p><p className="text-xs text-[#777a73]">{document.chunk_count} chunks · PDF {document.pdf_available ? "saved" : "not saved"}</p></div><button type="button" disabled={Boolean(deletingDocumentId)} onClick={() => { setDeleteTarget(document); setDeleteConfirmed(false); setDeleteError(""); }} className="self-start rounded-lg border border-[#e6c9c2] px-3 py-2 text-xs font-medium text-[#a64735] hover:bg-[#fff7f5] disabled:opacity-50">Delete</button></div></article>)}</div>
          {deleteTarget && <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4"><section role="alertdialog" aria-modal="true" aria-labelledby="delete-confirm-title" className="w-full max-w-lg rounded-2xl bg-white p-5 shadow-2xl"><h3 id="delete-confirm-title" className="text-base font-semibold text-[#30332f]">Confirm document deletion</h3><p className="mt-3 text-sm text-[#555a52]">Delete <strong className="break-words">{deleteTarget.filename}</strong> and its indexed chunks?</p><p className="mt-2 break-all font-mono text-xs text-[#777a73]">{deleteTarget.document_id}</p><label className="mt-4 flex items-start gap-2 text-sm leading-5 text-[#555a52]"><input type="checkbox" checked={deleteConfirmed} onChange={(event) => setDeleteConfirmed(event.target.checked)} className="mt-1 accent-[#315a3d]" />I confirm deletion of this exact document and its saved PDF.</label>{deleteError && <p role="alert" className="mt-3 rounded-lg bg-[#fff4f1] px-3 py-2 text-xs leading-5 text-[#a64735]">{deleteError}</p>}<div className="mt-5 flex justify-end gap-2"><button type="button" disabled={Boolean(deletingDocumentId)} onClick={() => { setDeleteTarget(null); setDeleteConfirmed(false); setDeleteError(""); }} className="rounded-lg border border-[#d9ddd7] px-3 py-2 text-xs font-medium text-[#555a52] disabled:opacity-50">Cancel</button><button type="button" disabled={!deleteConfirmed || Boolean(deletingDocumentId)} onClick={handleDeleteDocument} className="rounded-lg bg-[#a64735] px-3 py-2 text-xs font-semibold text-white disabled:opacity-40">{deletingDocumentId === deleteTarget.document_id ? "Deleting…" : "Delete document"}</button></div></section></div>}
        </section>
      </div>}
    </main>
  );
}
