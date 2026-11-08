import type { AnswerCitation } from "./api";

export interface CitationNavigationTarget {
  requestId: string;
  documentId: string;
  pageNumber: number;
  excerpt: string;
}

export interface NormalizedText {
  text: string;
  startOffsets: number[];
  endOffsets: number[];
}

export function normalizeCitationText(value: string): NormalizedText {
  const source = value;
  let text = "";
  const startOffsets: number[] = [];
  const endOffsets: number[] = [];
  const append = (value: string, start: number, end: number) => {
    for (const character of value.toLowerCase()) {
      text += character;
      startOffsets.push(start);
      endOffsets.push(end);
    }
  };

  for (let index = 0; index < source.length;) {
    const hyphenation = source.slice(index).match(/^([\p{L}\p{N}])[-\u00ad][ \t]*\r?\n[ \t]*([\p{L}\p{N}])/u);
    if (hyphenation) {
      const leftIndex = index;
      const leftEnd = leftIndex + hyphenation[1].length;
      const rightIndex = index + hyphenation[0].length - hyphenation[2].length;
      append(hyphenation[1], leftIndex, leftEnd);
      append(hyphenation[2], rightIndex, rightIndex + hyphenation[2].length);
      index += hyphenation[0].length;
      continue;
    }

    const character = source[index];
    if (/\s/u.test(character)) {
      let end = index + 1;
      while (end < source.length && /\s/u.test(source[end])) end += 1;
      if (text && end < source.length) {
        text += " ";
        startOffsets.push(index);
        endOffsets.push(end);
      }
      index = end;
      continue;
    }
    append(character, index, index + 1);
    index += 1;
  }

  return { text, startOffsets, endOffsets };
}

export function findReliableExcerptMatch(text: string, excerpt: string): { start: number; end: number } | null {
  const normalizedText = normalizeCitationText(text).text;
  const normalizedExcerpt = normalizeCitationText(excerpt).text;
  if (normalizedExcerpt.length < 12 || normalizedExcerpt.split(" ").filter(Boolean).length < 2) return null;

  const first = normalizedText.indexOf(normalizedExcerpt);
  if (first < 0 || normalizedText.indexOf(normalizedExcerpt, first + 1) >= 0) return null;
  return { start: first, end: first + normalizedExcerpt.length };
}

export function citationNavigationTarget(citation: AnswerCitation, requestId: string): CitationNavigationTarget {
  return {
    requestId,
    documentId: citation.document_id,
    pageNumber: citation.page_number,
    excerpt: citation.excerpt,
  };
}
