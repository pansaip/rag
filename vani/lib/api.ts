export interface UploadedDocument {
  document_id: string;
  filename: string;
  status: string;
  page_count: number;
  chunk_count: number;
  duplicate?: boolean;
}

export interface IndexedDocument {
  document_id: string;
  filename: string;
  chunk_count: number;
  pdf_available: boolean;
}

export interface SessionDocument extends IndexedDocument {
  status?: string;
  page_count?: number;
  file?: File;
}

export interface SearchResult {
  text: string;
  score: number;
  document_id: string;
  filename: string;
  page_number: number;
  chunk_index: number;
}

interface SearchResponse {
  results: SearchResult[];
}

interface DocumentListResponse {
  documents: IndexedDocument[];
}

export interface AnswerCitation {
  source_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  excerpt: string;
}

export interface AIAnswer {
  answer: string;
  citations: AnswerCitation[];
}

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

function responseDetail(value: unknown, fallback: string): string {
  if (value && typeof value === "object" && "detail" in value) {
    const detail = value.detail;
    if (typeof detail === "string") return detail;
  }
  return fallback;
}

function isIndexedDocument(value: unknown): value is IndexedDocument {
  if (!value || typeof value !== "object") return false;
  const document = value as Record<string, unknown>;
  return typeof document.document_id === "string"
    && document.document_id.length > 0
    && typeof document.filename === "string"
    && typeof document.chunk_count === "number"
    && Number.isInteger(document.chunk_count)
    && document.chunk_count >= 0
    && typeof document.pdf_available === "boolean";
}

export function mergeIndexedDocuments(
  current: SessionDocument[],
  indexed: IndexedDocument[],
): SessionDocument[] {
  const merged = new Map(current.map((document) => [document.document_id, document]));
  for (const document of indexed) {
    if (!merged.has(document.document_id)) merged.set(document.document_id, document);
  }
  return [...merged.values()];
}

export function mergeUploadedDocument(
  current: SessionDocument[],
  uploaded: UploadedDocument,
  file: File,
): SessionDocument[] {
  if (uploaded.duplicate) {
    const reused: SessionDocument = { ...uploaded, pdf_available: true };
    if (current.some((document) => document.document_id === uploaded.document_id)) {
      return current.map((document) => document.document_id === uploaded.document_id
        ? { ...document, ...reused, file: undefined }
        : document);
    }
    return [reused, ...current];
  }
  return mergeIndexedDocuments([{ ...uploaded, file, pdf_available: true }, ...current], []);
}

export async function listDocuments(): Promise<IndexedDocument[]> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/documents`, { cache: "no-store" });
  } catch {
    throw new Error("Could not reach the API. Check that FastAPI is running and CORS is enabled.");
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    throw new Error(responseDetail(body, `Document list failed (${response.status})`));
  }
  if (!body || typeof body !== "object" || !("documents" in body) || !Array.isArray(body.documents) || !body.documents.every(isIndexedDocument)) {
    throw new Error("The API returned an unexpected document list response.");
  }
  return body.documents;
}

export function documentPdfUrl(documentId: string): string {
  return `${API_BASE_URL}/documents/${encodeURIComponent(documentId)}/pdf`;
}

export async function saveOriginalPdf(documentId: string, file: File): Promise<void> {
  let response: Response;
  try {
    const form = new FormData();
    form.append("file", file);
    response = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(documentId)}/pdf`, {
      method: "PUT",
      body: form,
      cache: "no-store",
    });
  } catch {
    throw new Error("Could not reach the API. Check that FastAPI is running and CORS is enabled.");
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    throw new Error(responseDetail(body, `Saving the original PDF failed (${response.status})`));
  }
  if (!body || typeof body !== "object" || !("document_id" in body) || body.document_id !== documentId || !("pdf_available" in body) || body.pdf_available !== true) {
    throw new Error("The API returned an unexpected PDF attachment response.");
  }
}

export async function deleteDocument(documentId: string): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
      cache: "no-store",
    });
  } catch {
    throw new Error("Could not reach the API. Check that FastAPI is running and CORS is enabled.");
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    throw new Error(responseDetail(body, `Document deletion failed (${response.status})`));
  }
  if (!body || typeof body !== "object" || !("document_id" in body) || body.document_id !== documentId || !("deleted" in body) || body.deleted !== true) {
    throw new Error("The API returned an unexpected document deletion response.");
  }
}

export function uploadDocument(
  file: File,
  onProgress: (percent: number | null) => void,
): Promise<UploadedDocument> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);

    request.open("POST", `${API_BASE_URL}/documents`);
    request.upload.addEventListener("progress", (event) => {
      onProgress(event.lengthComputable ? Math.round((event.loaded / event.total) * 100) : null);
    });
    request.addEventListener("load", () => {
      let body: unknown;
      try {
        body = JSON.parse(request.responseText);
      } catch {
        body = null;
      }

      if (request.status < 200 || request.status >= 300) {
        reject(new Error(responseDetail(body, `Upload failed (${request.status})`)));
        return;
      }
      if (!body || typeof body !== "object" || !("document_id" in body)) {
        reject(new Error("The API returned an unexpected upload response."));
        return;
      }
      resolve(body as UploadedDocument);
    });
    request.addEventListener("error", () => {
      reject(new Error("Could not reach the API. Check that FastAPI is running and CORS is enabled."));
    });
    request.send(form);
  });
}

export async function searchDocuments(
  question: string,
  documentId?: string,
): Promise<SearchResult[]> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: 5,
        ...(documentId ? { document_id: documentId } : {}),
      }),
      cache: "no-store",
    });
  } catch {
    throw new Error("Could not reach the API. Check that FastAPI is running and CORS is enabled.");
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    throw new Error(responseDetail(body, `Search failed (${response.status})`));
  }
  if (!body || typeof body !== "object" || !("results" in body) || !Array.isArray(body.results)) {
    throw new Error("The API returned an unexpected search response.");
  }
  return body.results as SearchResult[];
}

export async function askQuestion(
  question: string,
  documentId?: string,
): Promise<AIAnswer> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        ...(documentId ? { document_id: documentId } : {}),
      }),
      cache: "no-store",
    });
  } catch {
    throw new Error("Could not reach the API. Check that FastAPI is running and CORS is enabled.");
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    throw new Error(responseDetail(body, `AI answer failed (${response.status})`));
  }
  if (!body || typeof body !== "object" || !("answer" in body) || typeof body.answer !== "string" || !("citations" in body) || !Array.isArray(body.citations)) {
    throw new Error("The API returned an unexpected answer response.");
  }
  return body as AIAnswer;
}
