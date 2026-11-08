import { createElement as h, Fragment, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { AnswerCitation } from "./api";

type InlineAnswerPart =
  | { kind: "text" | "code"; text: string }
  | { kind: "strong" | "emphasis"; children: InlineAnswerPart[] }
  | { kind: "link"; label: InlineAnswerPart[]; href: string }
  | { kind: "cited_span"; children: InlineAnswerPart[]; references: CitationReference[] }
  | { kind: "references"; references: CitationReference[] };

interface CitationReference {
  citation: AnswerCitation;
  displayNumber: number;
}

export function calculateCitationPopoverPosition(
  anchor: { top: number; bottom: number; left: number },
  popup: { width: number; height: number },
  viewport: { width: number; height: number },
): { top: number; left: number } {
  const gutter = 12;
  const gap = 8;
  const left = Math.max(gutter, Math.min(anchor.left, viewport.width - popup.width - gutter));
  let top = anchor.bottom + gap;
  if (top + popup.height > viewport.height - gutter) top = anchor.top - popup.height - gap;
  top = Math.max(gutter, Math.min(top, viewport.height - popup.height - gutter));
  return { top, left };
}

export function handleCitationMarkerClick(
  event: Pick<React.MouseEvent<HTMLButtonElement>, "currentTarget" | "stopPropagation">,
  isOpen: boolean,
  setOpen: (open: boolean) => void,
  viewport: { width: number; height: number },
): { top: number; left: number } | null {
  event.stopPropagation();
  if (isOpen) {
    setOpen(false);
    return null;
  }
  const anchor = event.currentTarget.getBoundingClientRect();
  const position = calculateCitationPopoverPosition(
    anchor,
    { width: Math.min(320, viewport.width - 24), height: Math.min(280, viewport.height - 24) },
    viewport,
  );
  setOpen(true);
  return position;
}

export function dismissCitationPopover(
  setOpen: (open: boolean) => void,
  trigger: { focus: () => void } | null,
  restoreFocus = false,
): void {
  setOpen(false);
  if (restoreFocus) trigger?.focus();
}

type CitationBoundary = { contains: (target: Node) => boolean } | null;

export function isInsideCitationInteraction(
  target: EventTarget | null,
  eventPath: EventTarget[],
  trigger: CitationBoundary,
  popup: CitationBoundary,
): boolean {
  if ((trigger && eventPath.includes(trigger as unknown as EventTarget)) || (popup && eventPath.includes(popup as unknown as EventTarget))) return true;
  if (!target) return false;
  return Boolean(trigger?.contains(target as Node) || popup?.contains(target as Node));
}

export function dismissCitationPopoverOutsideInteraction(
  target: EventTarget | null,
  eventPath: EventTarget[],
  trigger: CitationBoundary,
  popup: CitationBoundary,
  setOpen: (open: boolean) => void,
): boolean {
  if (isInsideCitationInteraction(target, eventPath, trigger, popup)) return false;
  dismissCitationPopover(setOpen, null);
  return true;
}

export function openCitationSource(
  citation: AnswerCitation,
  setOpen: (open: boolean) => void,
  onCitationClick: CitationClickHandler,
  trigger: { focus: () => void } | null,
): void {
  dismissCitationPopover(setOpen, trigger);
  onCitationClick(citation);
}

type AnswerBlock =
  | { kind: "paragraph" | "quote"; children: InlineAnswerPart[] }
  | { kind: "heading"; level: number; children: InlineAnswerPart[] }
  | { kind: "list"; ordered: boolean; items: InlineAnswerPart[][] };

function citationDisplayNumbers(answer: string, citations: AnswerCitation[]): Map<string, number> {
  const available = new Set(citations.map((citation) => citation.source_id));
  const numbers = new Map<string, number>();
  const markers = answer.matchAll(/\[(S\d+)\]/g);
  for (const marker of markers) {
    if (available.has(marker[1]) && !numbers.has(marker[1])) numbers.set(marker[1], numbers.size + 1);
  }
  return numbers;
}

const INLINE_PATTERN = /\[(S\d+)\]|\[([^\]]+)\]\(([^)]+)\)|\*\*([\s\S]+?)\*\*|__([\s\S]+?)__|\*([^*]+)\*|_([^_]+)_|`([^`]+)`/g;

const MAX_CITABLE_SPAN_CHARS = 400;

function plainText(parts: InlineAnswerPart[]): string {
  return parts.map((part) => {
    if (part.kind === "text" || part.kind === "code") return part.text;
    if (part.kind === "strong" || part.kind === "emphasis" || part.kind === "cited_span") return plainText(part.children);
    if (part.kind === "link") return plainText(part.label);
    return "";
  }).join("");
}

function inlineLength(part: InlineAnswerPart): number {
  if (part.kind === "text" || part.kind === "code") return part.text.length;
  if (part.kind === "strong" || part.kind === "emphasis" || part.kind === "cited_span") return part.children.reduce((sum, child) => sum + inlineLength(child), 0);
  if (part.kind === "link") return part.label.reduce((sum, child) => sum + inlineLength(child), 0);
  return 0;
}

function splitInlineParts(parts: InlineAnswerPart[], offset: number): [InlineAnswerPart[], InlineAnswerPart[]] {
  const before: InlineAnswerPart[] = [];
  const after: InlineAnswerPart[] = [];
  let remaining = offset;
  for (const part of parts) {
    const length = inlineLength(part);
    if (remaining >= length) {
      before.push(part);
      remaining -= length;
      continue;
    }
    if (remaining <= 0) {
      after.push(part);
      continue;
    }
    if (part.kind === "text" || part.kind === "code") {
      before.push({ kind: part.kind, text: part.text.slice(0, remaining) });
      after.push({ kind: part.kind, text: part.text.slice(remaining) });
    } else if (part.kind === "strong" || part.kind === "emphasis" || part.kind === "cited_span") {
      const [leftChildren, rightChildren] = splitInlineParts(part.children, remaining);
      if (leftChildren.length) before.push({ ...part, children: leftChildren } as InlineAnswerPart);
      if (rightChildren.length) after.push({ ...part, children: rightChildren } as InlineAnswerPart);
    } else if (part.kind === "link") {
      const [leftLabel, rightLabel] = splitInlineParts(part.label, remaining);
      if (leftLabel.length) before.push({ kind: "link", label: leftLabel, href: part.href });
      if (rightLabel.length) after.push({ kind: "link", label: rightLabel, href: part.href });
    } else {
      after.push(part);
    }
    remaining = 0;
  }
  return [before, after];
}

function splitCitedClaim(parts: InlineAnswerPart[], hasPriorCitation: boolean, priorSpanEndedSentence: boolean): { before: InlineAnswerPart[]; claim: InlineAnswerPart[]; after: InlineAnswerPart[] } | null {
  const text = plainText(parts);
  const end = text.search(/\s*$/);
  const contentEnd = end < 0 ? text.length : end;
  if (!contentEnd) return null;
  const boundaryPattern = /[.!?。！？]["'”’)]*\s+/g;
  let boundary: RegExpExecArray | null;
  let claimStart = 0;
  let hasSentenceBoundary = false;
  while ((boundary = boundaryPattern.exec(text)) !== null) {
    hasSentenceBoundary = true;
    claimStart = boundary.index + boundary[0].length;
  }
  if (claimStart >= contentEnd) claimStart = 0;
  while (claimStart < contentEnd && /\s/.test(text[claimStart])) claimStart += 1;
  const candidate = text.slice(claimStart, contentEnd).trim();
  if (claimStart >= contentEnd || candidate.length < 8 || candidate.length > MAX_CITABLE_SPAN_CHARS) return null;
  if (hasPriorCitation && !hasSentenceBoundary && !priorSpanEndedSentence && !/[.!?。！？]["'”’)]*$/.test(candidate)) return null;
  if (hasPriorCitation && /^(?:and|but|or|also)\b|^(?:และ|แต่|หรือ|รวมถึง)/i.test(candidate)) return null;

  const [before, remainder] = splitInlineParts(parts, claimStart);
  const [claim, after] = splitInlineParts(remainder, contentEnd - claimStart);
  if (!plainText(claim).trim()) return null;
  return { before, claim, after };
}

function parseInline(value: string, bySource: Map<string, AnswerCitation>, numbers: Map<string, number>): InlineAnswerPart[] {
  const parts: InlineAnswerPart[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  const pattern = new RegExp(INLINE_PATTERN.source, "g");
  while ((match = pattern.exec(value)) !== null) {
    if (match.index > lastIndex) parts.push({ kind: "text", text: value.slice(lastIndex, match.index) });
    if (match[1]) {
      const citation = bySource.get(match[1]);
      const displayNumber = numbers.get(match[1]);
      if (citation && displayNumber !== undefined) {
        const reference = { citation, displayNumber };
        const previousCitationIndex = parts.findLastIndex((part) => part.kind === "cited_span" || part.kind === "references");
        const base = parts.slice(0, previousCitationIndex + 1);
        const pending = parts.slice(previousCitationIndex + 1);
        const pendingText = plainText(pending);
        const lastPart = previousCitationIndex >= 0 ? parts[previousCitationIndex] : undefined;
        if (!pendingText.trim() && (lastPart?.kind === "cited_span" || lastPart?.kind === "references")) {
          if (!lastPart.references.some((item) => item.citation.source_id === citation.source_id)) lastPart.references.push(reference);
          parts.splice(previousCitationIndex + 1);
        } else {
          const priorSpanEndedSentence = lastPart?.kind === "cited_span"
            && /[.!?。！？]["'”’)]*$/.test(plainText(lastPart.children).trimEnd());
          const split = splitCitedClaim(pending, previousCitationIndex >= 0, priorSpanEndedSentence);
          if (split) {
            parts.splice(0, parts.length, ...base, ...split.before);
            parts.push({ kind: "cited_span", children: split.claim, references: [reference] }, ...split.after);
          } else {
            parts.push({ kind: "references", references: [reference] });
          }
        }
      }
    } else if (match[2] !== undefined) {
      const label = parseInline(match[2], bySource, numbers);
      if (/^https?:\/\//i.test(match[3])) parts.push({ kind: "link", label, href: match[3] });
      else parts.push(...label);
    } else if (match[4] !== undefined || match[5] !== undefined) {
      parts.push({ kind: "strong", children: parseInline(match[4] ?? match[5], bySource, numbers) });
    } else if (match[6] !== undefined || match[7] !== undefined) {
      parts.push({ kind: "emphasis", children: parseInline(match[6] ?? match[7], bySource, numbers) });
    } else if (match[8] !== undefined) {
      parts.push(...parseInline(match[8], bySource, numbers).map((part) => part.kind === "text" || part.kind === "code" ? { kind: "code" as const, text: part.text } : part));
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < value.length) parts.push({ kind: "text", text: value.slice(lastIndex) });
  return parts;
}

function parseAnswerMarkdown(answer: string, citations: AnswerCitation[]): AnswerBlock[] {
  const bySource = new Map<string, AnswerCitation>();
  for (const citation of citations) if (!bySource.has(citation.source_id)) bySource.set(citation.source_id, citation);
  const numbers = citationDisplayNumbers(answer, citations);
  const lines = answer.replace(/\r\n?/g, "\n").split("\n");
  const blocks: AnswerBlock[] = [];
  let index = 0;
  while (index < lines.length) {
    while (index < lines.length && !lines[index].trim()) index += 1;
    if (index >= lines.length) break;
    const heading = lines[index].match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      blocks.push({ kind: "heading", level: heading[1].length, children: parseInline(heading[2], bySource, numbers) });
      index += 1;
      continue;
    }
    if (/^\s*>\s?/.test(lines[index])) {
      const quoteLines: string[] = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index])) quoteLines.push(lines[index++].replace(/^\s*>\s?/, ""));
      blocks.push({ kind: "quote", children: parseInline(quoteLines.join(" "), bySource, numbers) });
      continue;
    }
    const listLine = lines[index].match(/^\s*([-*+] |\d+[.)] )(.+)$/);
    if (listLine) {
      const ordered = /^\d/.test(listLine[1]);
      const items: InlineAnswerPart[][] = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s*([-*+] |\d+[.)] )(.+)$/);
        if (!item || /^\d/.test(item[1]) !== ordered) break;
        items.push(parseInline(item[2], bySource, numbers));
        index += 1;
      }
      blocks.push({ kind: "list", ordered, items });
      continue;
    }
    const paragraphLines: string[] = [];
    while (index < lines.length && lines[index].trim()) {
      if (paragraphLines.length && (/^#{1,3}\s+/.test(lines[index]) || /^\s*>\s?/.test(lines[index]) || /^\s*([-*+] |\d+[.)] )/.test(lines[index]))) break;
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    if (paragraphLines.length) blocks.push({ kind: "paragraph", children: parseInline(paragraphLines.join(" "), bySource, numbers) });
  }
  return blocks;
}

export type CitationClickHandler = (citation: AnswerCitation) => void;

function renderInline(parts: InlineAnswerPart[], onCitationClick: CitationClickHandler): ReactNode[] {
  return parts.map((part, index) => {
    if (part.kind === "text") return part.text;
    if (part.kind === "code") return h("code", { key: index, className: "rounded bg-[#f1f2ee] px-1 py-0.5 font-mono text-[0.9em] text-[#465047]" }, part.text);
    if (part.kind === "strong") return h("strong", { key: index, className: "font-semibold text-[#303630]" }, renderInline(part.children, onCitationClick));
    if (part.kind === "emphasis") return h("em", { key: index, className: "italic" }, renderInline(part.children, onCitationClick));
    if (part.kind === "link") return h("a", { key: index, href: part.href, target: "_blank", rel: "noopener noreferrer", className: "text-[#315a3d] underline decoration-[#9bb29d] underline-offset-2 hover:text-[#183b29]" }, renderInline(part.label, onCitationClick));
    if (part.kind === "references") return h(CitationPopover, { key: index, references: part.references, onCitationClick });
    if (part.kind === "cited_span") {
      return h(Fragment, { key: index },
        ...renderInline(part.children, onCitationClick),
        h(CitationPopover, { key: `${index}-citation`, references: part.references, onCitationClick }),
      );
    }
    return null;
  });
}

function CitationPopover({ references, onCitationClick }: { references: CitationReference[]; onCitationClick: CitationClickHandler }) {
  const marker = references[0];
  if (!marker) return null;
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const keyboardOpened = useRef(false);
  const label = references.length === 1
    ? `Citation ${marker.displayNumber}, ${marker.citation.filename}, page ${marker.citation.page_number}`
    : `Citation ${marker.displayNumber}, ${references.length} sources`;

  useEffect(() => {
    if (!isOpen) return;
    if (keyboardOpened.current) {
      popoverRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
      keyboardOpened.current = false;
    }
    const onPointerDown = (event: PointerEvent) => {
      dismissCitationPopoverOutsideInteraction(
        event.target,
        event.composedPath(),
        triggerRef.current,
        popoverRef.current,
        setIsOpen,
      );
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      dismissCitationPopover(setIsOpen, triggerRef.current, true);
    };
    const onScroll = (event: Event) => {
      dismissCitationPopoverOutsideInteraction(
        event.target,
        event.composedPath(),
        triggerRef.current,
        popoverRef.current,
        setIsOpen,
      );
    };
    const onResize = () => dismissCitationPopover(setIsOpen, triggerRef.current);
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onResize);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onResize);
    };
  }, [isOpen]);

  const onMarkerClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    const next = handleCitationMarkerClick(event, isOpen, setIsOpen, {
      width: window.innerWidth,
      height: window.innerHeight,
    });
    setPosition(next);
  };

  return h(Fragment, null,
    h("button", {
      ref: triggerRef,
      type: "button",
      "aria-haspopup": "dialog",
      "aria-expanded": isOpen,
      "aria-label": `Show source preview: ${label}`,
      onClick: onMarkerClick,
      onKeyDown: (event: React.KeyboardEvent<HTMLButtonElement>) => {
        if (event.key === "Enter" || event.key === " ") keyboardOpened.current = true;
      },
      className: "citation-marker focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#516b82]",
    }, `[${marker.displayNumber}]`),
    isOpen && position && typeof document !== "undefined" ? createPortal(
      h("div", {
        ref: popoverRef,
        role: "dialog",
        "aria-label": `Source preview: ${label}`,
        className: "citation-popover",
        style: { top: position.top, left: position.left },
      },
      h("div", { className: "citation-popover-content" }, ...references.map(({ citation, displayNumber }, referenceIndex) =>
        h("section", { key: `${citation.source_id}-${referenceIndex}`, className: "citation-source-preview" },
          h("p", { className: "citation-source-heading" },
            references.length > 1 ? h("span", { className: "citation-source-number" }, `[${displayNumber}] `) : null,
            h("span", { className: "citation-source-filename", title: citation.filename }, citation.filename),
            h("span", { className: "citation-source-page" }, ` · Page ${citation.page_number}`),
          ),
          h("p", { className: "citation-source-excerpt" }, citation.excerpt || "No excerpt is available for this source."),
          h("button", {
            type: "button",
            className: "citation-open-pdf focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#516b82]",
            onClick: () => {
              openCitationSource(citation, setIsOpen, onCitationClick, triggerRef.current);
              setPosition(null);
            },
          }, "Open in PDF"),
        ),
      ))),
      document.body,
    ) : null,
  );
}

function renderBlock(block: AnswerBlock, index: number, onCitationClick: CitationClickHandler): ReactNode {
  const children = "children" in block ? renderInline(block.children, onCitationClick) : null;
  if (block.kind === "paragraph") return h("p", { key: index, className: "my-3 whitespace-pre-wrap break-words text-sm leading-7 text-[#41443e] first:mt-0 last:mb-0" }, children);
  if (block.kind === "quote") return h("blockquote", { key: index, className: "my-3 border-l-2 border-[#cbd8ca] pl-4 text-sm leading-7 text-[#62675f]" }, children);
  if (block.kind === "heading") {
    const tag = `h${block.level + 1}`;
    return h(tag, { key: index, className: "mb-2 mt-4 text-base font-semibold leading-6 text-[#303630] first:mt-0" }, children);
  }
  if (block.kind !== "list") return null;
  const listTag = block.ordered ? "ol" : "ul";
  return h(listTag, { key: index, className: `${block.ordered ? "list-decimal" : "list-disc"} my-3 space-y-1 pl-6 text-sm leading-7 text-[#41443e]` },
    ...block.items.map((item, itemIndex) => h("li", { key: itemIndex }, renderInline(item, onCitationClick))));
}

export function AnswerPresentation({ answer, citations, onCitationClick }: {
  answer: string;
  citations: AnswerCitation[];
  onCitationClick: CitationClickHandler;
}) {
  const blocks = parseAnswerMarkdown(answer, citations);
  return h("div", { className: "answer-markdown" }, ...blocks.map((block, index) => renderBlock(block, index, onCitationClick)));
}
