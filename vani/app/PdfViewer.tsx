"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { findReliableExcerptMatch, normalizeCitationText, type CitationNavigationTarget } from "@/lib/citations";
import { estimatedPdfPageHeight, PDF_HIGHLIGHT_UNAVAILABLE_NOTICE, PDF_LAZY_RENDER_ROOT_MARGIN, pdfPageNumbers, resolveCitationPage, separatorBetweenPdfTextItems, shouldRenderPdfPage } from "@/lib/pdf_reader";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

type PdfViewerProps = {
  file: File | string | null;
  documentId?: string;
  navigationTarget?: CitationNavigationTarget | null;
};

type HighlightRect = { left: number; top: number; width: number; height: number };

export default function PdfViewer({ file, documentId, navigationTarget }: PdfViewerProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [pageCount, setPageCount] = useState(0);
  const [visiblePages, setVisiblePages] = useState<Set<number>>(() => new Set([1]));
  const [activePage, setActivePage] = useState(1);
  const [zoom, setZoom] = useState(100);
  const [error, setError] = useState("");
  const [pageHeightRatio, setPageHeightRatio] = useState(1.294);
  const [highlight, setHighlight] = useState<{ page: number; rects: HighlightRect[] } | null>(null);
  const [highlightNotice, setHighlightNotice] = useState("");

  useEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(entry.contentRect.width);
    });
    observer.observe(panel);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setPageCount(0);
    setVisiblePages(new Set([1]));
    setActivePage(1);
    setError("");
    setHighlight(null);
    setHighlightNotice("");
    if (panelRef.current) panelRef.current.scrollTop = 0;
  }, [file]);

  useEffect(() => {
    const panel = panelRef.current;
    if (!panel || pageCount < 1) return;
    const observer = new IntersectionObserver((entries) => {
      setVisiblePages((current) => {
        const next = new Set(current);
        for (const entry of entries) {
          const pageNumber = Number((entry.target as HTMLElement).dataset.pageNumber);
          if (!Number.isInteger(pageNumber)) continue;
          if (entry.isIntersecting) next.add(pageNumber);
          else next.delete(pageNumber);
        }
        return next;
      });
    }, { root: panel, rootMargin: PDF_LAZY_RENDER_ROOT_MARGIN, threshold: 0 });
    const activeRatios = new Map<number, number>();
    const activeObserver = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const pageNumber = Number((entry.target as HTMLElement).dataset.pageNumber);
        if (Number.isInteger(pageNumber)) activeRatios.set(pageNumber, entry.isIntersecting ? entry.intersectionRatio : 0);
      }
      const bestPage = [...activeRatios.entries()].sort((left, right) => right[1] - left[1])[0];
      if (bestPage && bestPage[1] > 0) setActivePage(bestPage[0]);
    }, { root: panel, threshold: [0, 0.15, 0.35, 0.65, 0.9] });
    panel.querySelectorAll<HTMLElement>("[data-page-number]").forEach((page) => {
      observer.observe(page);
      activeObserver.observe(page);
    });
    return () => { observer.disconnect(); activeObserver.disconnect(); };
  }, [pageCount, file]);

  const scrollPanelToRect = useCallback((rect: DOMRect) => {
    const panel = panelRef.current;
    if (!panel) return;
    const panelBounds = panel.getBoundingClientRect();
    const top = panel.scrollTop + rect.top - panelBounds.top - 24;
    if (top >= panel.scrollTop && rect.bottom <= panelBounds.bottom - 12) return;
    panel.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  }, []);

  const highlightCitationPassage = useCallback((pageNumber: number) => {
    if (!navigationTarget || navigationTarget.documentId !== documentId || navigationTarget.pageNumber !== pageNumber) return;
    const pageElement = panelRef.current?.querySelector<HTMLElement>(`[data-page-number="${pageNumber}"] .textLayer`);
    if (!pageElement) return;
    if (!navigationTarget.excerpt.trim()) {
      setHighlight(null);
      setHighlightNotice(PDF_HIGHLIGHT_UNAVAILABLE_NOTICE);
      return;
    }

    const walker = window.document.createTreeWalker(pageElement, NodeFilter.SHOW_TEXT);
    const sourcePositions: ({ node: Text; offset: number } | null)[] = [];
    let sourceText = "";
    let previousNode: Text | null = null;
    let currentNode: Node | null;
    while ((currentNode = walker.nextNode())) {
      const textNode = currentNode as Text;
      if (!textNode.data) continue;
      if (previousNode) {
        const previousRect = previousNode.parentElement?.getBoundingClientRect();
        const nextRect = textNode.parentElement?.getBoundingClientRect();
        if (previousRect && nextRect) {
          const separator = separatorBetweenPdfTextItems(previousRect, nextRect, sourceText, textNode.data);
          if (separator) {
            sourceText += separator;
            sourcePositions.push(null);
          }
        }
      }
      sourceText += textNode.data;
      for (let offset = 0; offset < textNode.length; offset += 1) sourcePositions.push({ node: textNode, offset });
      previousNode = textNode;
    }

    const match = findReliableExcerptMatch(sourceText, navigationTarget.excerpt);
    if (!match) {
      setHighlight(null);
      setHighlightNotice(PDF_HIGHLIGHT_UNAVAILABLE_NOTICE);
      return;
    }

    const normalized = normalizeCitationText(sourceText);
    const sourceStart = normalized.startOffsets[match.start];
    const sourceEnd = normalized.endOffsets[match.end - 1];
    const startPosition = sourcePositions[sourceStart];
    const lastMatchedPosition = sourcePositions[sourceEnd - 1];
    const endPosition = lastMatchedPosition ? { node: lastMatchedPosition.node, offset: lastMatchedPosition.offset + 1 } : null;
    if (!startPosition || !endPosition) {
      setHighlight(null);
      setHighlightNotice(PDF_HIGHLIGHT_UNAVAILABLE_NOTICE);
      return;
    }

    const range = window.document.createRange();
    range.setStart(startPosition.node, startPosition.offset);
    range.setEnd(endPosition.node, endPosition.offset);
    const pageBounds = pageElement.closest<HTMLElement>("[data-page-number]")?.getBoundingClientRect();
    if (!pageBounds) return;
    const rects = Array.from(range.getClientRects())
      .filter((rect) => rect.width > 0 && rect.height > 0)
      .map((rect) => ({ left: rect.left - pageBounds.left, top: rect.top - pageBounds.top, width: rect.width, height: rect.height }));
    if (rects.length === 0) {
      setHighlight(null);
      setHighlightNotice(PDF_HIGHLIGHT_UNAVAILABLE_NOTICE);
      return;
    }
    setHighlight({ page: pageNumber, rects });
    setHighlightNotice("");
    scrollPanelToRect(range.getBoundingClientRect());
  }, [navigationTarget, documentId, scrollPanelToRect]);

  useEffect(() => {
    if (!navigationTarget || navigationTarget.documentId !== documentId) return;
    setHighlight(null);
    setHighlightNotice("");
    if (!file) {
      setHighlightNotice("The cited PDF is not available to preview.");
      return;
    }
    if (pageCount > 0 && !resolveCitationPage(navigationTarget.pageNumber, pageCount).valid) {
      setHighlightNotice("The cited page is outside this PDF; passage highlighting was unavailable.");
      return;
    }
    const pageNumber = Math.max(1, navigationTarget.pageNumber);
    setActivePage(pageNumber);
    setVisiblePages((current) => new Set(current).add(pageNumber));
    const page = panelRef.current?.querySelector<HTMLElement>(`[data-page-number="${pageNumber}"]`);
    if (page && panelRef.current) {
      const panel = panelRef.current;
      const panelBounds = panel.getBoundingClientRect();
      const pageBounds = page.getBoundingClientRect();
      panel.scrollTo({ top: panel.scrollTop + pageBounds.top - panelBounds.top - 20, behavior: "smooth" });
      const frame = window.requestAnimationFrame(() => highlightCitationPassage(pageNumber));
      return () => window.cancelAnimationFrame(frame);
    }
  }, [navigationTarget, documentId, file, pageCount, highlightCitationPassage]);

  const pageWidth = width > 0 ? Math.max(220, Math.min(width - 40, 1100)) * (zoom / 100) : 0;

  return (
    <section className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-[#e5e8e2] bg-[#eef0ec]">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#e1e4dd] bg-white px-3 py-2.5 sm:px-4">
        <span className="text-xs tabular-nums text-[#656961]">{pageCount ? `${pageCount} pages · Viewing ${activePage}` : file ? "Loading document…" : "PDF preview"}</span>
        <div className="flex items-center gap-1.5">
          <button type="button" onClick={() => setZoom((current) => Math.max(60, current - 10))} disabled={!file || zoom <= 60} className="viewer-control" aria-label="Zoom out">−</button>
          <span className="w-12 text-center text-xs tabular-nums text-[#656961]">{zoom}%</span>
          <button type="button" onClick={() => setZoom((current) => Math.min(160, current + 10))} disabled={!file || zoom >= 160} className="viewer-control" aria-label="Zoom in">+</button>
        </div>
      </div>

      <div ref={panelRef} role="region" tabIndex={0} className="min-h-0 flex-1 overflow-auto overscroll-contain p-3 sm:p-5" aria-label="Continuous PDF pages">
        {!file ? (
          <div className="flex min-h-full items-center justify-center rounded-xl border border-dashed border-[#d4d9d1] bg-white/60 px-6 py-12 text-center">
            <div><div className="mx-auto mb-3 flex size-11 items-center justify-center rounded-xl bg-white text-[#66806b] shadow-sm">PDF</div><p className="text-sm font-medium text-[#53574f]">Choose a document to preview</p><p className="mt-1 text-xs leading-5 text-[#878b83]">The selected PDF appears here while you work in chat.</p></div>
          </div>
        ) : (
          <div className="mx-auto w-fit max-w-full">
            {error ? <div role="alert" className="rounded-xl border border-[#f0d9d2] bg-white px-5 py-4 text-sm text-[#a64735]">{error}</div> : (
              <Document
                key={typeof file === "string" ? file : `${file.name}-${file.size}-${file.lastModified}`}
                file={file}
                suspense={false}
                loading={<p className="rounded-xl bg-white px-5 py-4 text-sm text-[#777a73]">Loading PDF…</p>}
                error={<p role="alert" className="rounded-xl bg-white px-5 py-4 text-sm text-[#a64735]">Could not render this PDF. Select the original file again.</p>}
                onLoadSuccess={async (pdf) => {
                  setPageCount(pdf.numPages);
                  setVisiblePages(new Set([1]));
                  setActivePage(1);
                  try {
                    const firstPage = await pdf.getPage(1);
                    const viewport = firstPage.getViewport({ scale: 1 });
                    setPageHeightRatio(viewport.height / viewport.width);
                  } catch {
                    setPageHeightRatio(1.294);
                  }
                  setError("");
                }}
                onLoadError={() => setError("Could not open this PDF. Try selecting the original file again.")}
              >
                <div className="space-y-4">
                  {pdfPageNumbers(pageCount).map((pageNumber) => {
                    const isVisible = shouldRenderPdfPage(pageNumber, visiblePages);
                    return <div
                      key={pageNumber}
                      data-page-number={pageNumber}
                      className="citation-pdf-page relative mx-auto overflow-hidden rounded-sm bg-white shadow-[0_5px_24px_rgba(25,35,25,0.16)]"
                      style={{ width: pageWidth || undefined, minHeight: pageWidth ? estimatedPdfPageHeight(width, pageHeightRatio, zoom) : 300 }}
                    >
                      {isVisible && pageWidth > 0 ? <Page
                        pageNumber={pageNumber}
                        width={pageWidth}
                        renderTextLayer
                        renderAnnotationLayer={false}
                        onRenderTextLayerSuccess={() => highlightCitationPassage(pageNumber)}
                        onRenderTextLayerError={() => {
                          if (navigationTarget && navigationTarget.documentId === documentId && navigationTarget.pageNumber === pageNumber) {
                            setHighlight(null);
                            setHighlightNotice(PDF_HIGHLIGHT_UNAVAILABLE_NOTICE);
                          }
                        }}
                      /> : <div aria-hidden="true" className="w-full" style={{ height: pageWidth ? estimatedPdfPageHeight(width, pageHeightRatio, zoom) : 300 }} />}
                      {highlight?.page === pageNumber && highlight.rects.map((rect, rectIndex) => <span key={`${rectIndex}-${rect.left}-${rect.top}`} aria-hidden="true" className="citation-pdf-highlight" style={rect} />)}
                    </div>;
                  })}
                </div>
              </Document>
            )}
          </div>
        )}
        {highlightNotice && <p role="status" className="sticky bottom-1 z-10 mx-auto mt-3 w-fit max-w-full rounded-lg border border-[#e9e4cd] bg-[#fffdf4] px-3 py-2 text-xs text-[#77704d]">{highlightNotice}</p>}
      </div>
    </section>
  );
}
