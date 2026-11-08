export const PDF_LAZY_RENDER_ROOT_MARGIN = "120% 0px";
export const PDF_HIGHLIGHT_UNAVAILABLE_NOTICE = "Page opened; passage highlighting was unavailable.";

export function shouldRenderPdfPage(pageNumber: number, visiblePages: ReadonlySet<number>): boolean {
  return visiblePages.has(pageNumber);
}

export function pdfPageNumbers(pageCount: number): number[] {
  return Array.from({ length: Math.max(0, pageCount) }, (_, index) => index + 1);
}

export function estimatedPdfPageHeight(width: number, heightToWidthRatio: number, zoom: number): number {
  return Math.max(220, width - 40) * (zoom / 100) * heightToWidthRatio;
}

export function resolveCitationPage(requestedPage: number, pageCount: number): { pageNumber: number | null; valid: boolean } {
  const valid = Number.isInteger(requestedPage) && requestedPage >= 1 && requestedPage <= pageCount;
  return { pageNumber: valid ? requestedPage : null, valid };
}

export function separatorBetweenPdfTextItems(
  previous: { left: number; right: number; top: number; height: number },
  next: { left: number; right: number; top: number; height: number },
  previousText: string,
  nextText: string,
): "" | " " | "\n" {
  if (!previousText || !nextText || /\s$/u.test(previousText) || /^\s/u.test(nextText)) return "";
  const verticalGap = Math.abs(next.top - previous.top);
  if (verticalGap > Math.max(2, previous.height * 0.45)) return "\n";
  const horizontalGap = next.left - previous.right;
  if (horizontalGap > Math.max(1.5, Math.min(previous.height, next.height) * 0.12)) return " ";
  return "";
}
