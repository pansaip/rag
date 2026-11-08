import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AnswerPresentation, calculateCitationPopoverPosition, dismissCitationPopover, dismissCitationPopoverOutsideInteraction, handleCitationMarkerClick, openCitationSource } from "../lib/answer_presentation.ts";
import { pdfPageNumbers, estimatedPdfPageHeight, PDF_HIGHLIGHT_UNAVAILABLE_NOTICE, PDF_LAZY_RENDER_ROOT_MARGIN, resolveCitationPage, separatorBetweenPdfTextItems, shouldRenderPdfPage } from "../lib/pdf_reader.ts";
import { findReliableExcerptMatch } from "../lib/citations.ts";

const citations = [
  { source_id: "S4", document_id: "doc-4", filename: "four.pdf", page_number: 7, excerpt: "A supporting passage with enough detail." },
  { source_id: "S1", document_id: "doc-1", filename: "one.pdf", page_number: 2, excerpt: "Another passage with enough detail." },
];

test("continuous reader keeps page order and renders only near-visible pages", () => {
  assert.deepEqual(pdfPageNumbers(5), [1, 2, 3, 4, 5]);
  const visiblePages = new Set([2, 3]);
  assert.equal(shouldRenderPdfPage(1, visiblePages), false);
  assert.equal(shouldRenderPdfPage(2, visiblePages), true);
  assert.equal(PDF_LAZY_RENDER_ROOT_MARGIN, "120% 0px");
  assert.equal(estimatedPdfPageHeight(600, 1.3, 100), 728);
});

test("citation page navigation accepts valid pages and rejects out-of-range pages", () => {
  assert.deepEqual(resolveCitationPage(7, 9), { pageNumber: 7, valid: true });
  assert.deepEqual(resolveCitationPage(10, 9), { pageNumber: null, valid: false });
  assert.deepEqual(resolveCitationPage(0, 9), { pageNumber: null, valid: false });
});

test("highlight fallback is subtle and unmatched or ambiguous excerpts never match", () => {
  assert.equal(PDF_HIGHLIGHT_UNAVAILABLE_NOTICE, "Page opened; passage highlighting was unavailable.");
  const excerpt = "The verified source passage is specific and long enough.";
  assert.notEqual(findReliableExcerptMatch(`Before. ${excerpt} After.`, excerpt), null);
  assert.equal(findReliableExcerptMatch("An unrelated page contains other evidence instead.", excerpt), null);
  assert.equal(findReliableExcerptMatch(`${excerpt} ${excerpt}`, excerpt), null);
});

test("PDF.js text items receive separators from line breaks and visual word gaps", () => {
  assert.equal(separatorBetweenPdfTextItems({ left: 0, right: 20, top: 0, height: 12 }, { left: 24, right: 40, top: 0, height: 12 }, "first", "second"), " ");
  assert.equal(separatorBetweenPdfTextItems({ left: 0, right: 20, top: 0, height: 12 }, { left: 0, right: 20, top: 16, height: 12 }, "hyphen-", "ated"), "\n");
  assert.equal(separatorBetweenPdfTextItems({ left: 0, right: 20, top: 0, height: 12 }, { left: 20, right: 40, top: 0, height: 12 }, "joined", "word"), "");
});

test("citations follow the sentence they support and do not create a bottom reference section", () => {
  const answer = "First supported claim. [S4] Second supported claim. [S1]";
  const markup = renderToStaticMarkup(createElement(AnswerPresentation, { answer, citations, onCitationClick() {} }));
  assert.match(markup, /First supported claim\.<button[^>]*citation-marker[^>]*>\[1\]<\/button>/);
  assert.match(markup, /Second supported claim\.<button[^>]*citation-marker[^>]*>\[2\]<\/button>/);
  assert.doesNotMatch(markup, /answer-references|<details|References/);
  assert.doesNotMatch(markup, /citation-tone|answer-cited-text|text-decoration/);
});

test("Thai punctuation and separate Markdown claims retain inline marker placement", () => {
  const answer = "ผู้วิจัยเสนอแนวทางใหม่。 [S4]\n\n- **ผลการทดลองดีขึ้น** [S1]\n- Uncited follow-up";
  const markup = renderToStaticMarkup(createElement(AnswerPresentation, { answer, citations, onCitationClick() {} }));
  assert.match(markup, /ผู้วิจัยเสนอแนวทางใหม่。<button[^>]*citation-marker[^>]*>\[1\]<\/button>/);
  assert.match(markup, /<strong[^>]*>ผลการทดลองดีขึ้น<\/strong><button[^>]*citation-marker[^>]*>\[2\]<\/button>/);
  assert.match(markup, /<ul/);
  assert.match(markup, /Uncited follow-up/);
});

test("Markdown paragraphs, bold, emphasis, code, and safe links remain readable", () => {
  const answer = "## Finding\n\n**Supported claim** [S4]. *Uncited detail.*\n\n- `example`\n\n[Safe link](https://example.com) and [unsafe](javascript:alert(1)).";
  const markup = renderToStaticMarkup(createElement(AnswerPresentation, { answer, citations, onCitationClick() {} }));
  assert.match(markup, /<h3/);
  assert.match(markup, /<strong/);
  assert.match(markup, /<em/);
  assert.match(markup, /<code/);
  assert.match(markup, /<ul/);
  assert.match(markup, /href="https:\/\/example.com"/);
  assert.doesNotMatch(markup, /href="javascript:/);
});

test("repeated source IDs reuse numbering and duplicate same-boundary controls are removed", () => {
  const repeated = renderToStaticMarkup(createElement(AnswerPresentation, { answer: "First claim. [S4] Second claim. [S4]", citations, onCitationClick() {} }));
  assert.equal((repeated.match(/class="citation-marker/g) ?? []).length, 2);
  assert.equal((repeated.match(/>\[1\]<\/button>/g) ?? []).length, 2);

  const adjacent = renderToStaticMarkup(createElement(AnswerPresentation, { answer: "One claim [S4][S4]", citations, onCitationClick() {} }));
  assert.equal((adjacent.match(/class="citation-marker/g) ?? []).length, 1);
  assert.match(adjacent, />\[1\]<\/button>/);
});

test("ambiguous claim boundaries preserve one source control without inventing a highlighted span", () => {
  const answer = `${"Unsegmented answer text ".repeat(22)}[S4]`;
  const markup = renderToStaticMarkup(createElement(AnswerPresentation, { answer, citations, onCitationClick() {} }));
  assert.equal((markup.match(/class="citation-marker[^"]*"/g) ?? []).length, 1);
  assert.doesNotMatch(markup, /answer-cited-text|citation-tone|>…<\/button>/);
  assert.doesNotMatch(markup, /answer-references|<details/);
});

test("citation marker renders as an accessible button and its click handler opens and switches previews", () => {
  const markup = renderToStaticMarkup(createElement(AnswerPresentation, { answer: "Claim [S4].", citations, onCitationClick() {} }));
  assert.match(markup, /<button[^>]*aria-haspopup="dialog"[^>]*aria-expanded="false"[^>]*class="citation-marker[^>]*>\[1\]<\/button>/);
  assert.doesNotMatch(markup, /popover="auto"|popovertarget|showPopover/);

  let open = false;
  const setOpen = (value) => { open = value; };
  const event = {
    currentTarget: { getBoundingClientRect: () => ({ top: 215, bottom: 235, left: 360 }) },
    stopPropagation() {},
  };
  const position = handleCitationMarkerClick(event, open, setOpen, { width: 400, height: 260 });
  assert.equal(open, true);
  assert.deepEqual(position, { top: 12, left: 68 });
  handleCitationMarkerClick({ ...event, currentTarget: { getBoundingClientRect: () => ({ top: 10, bottom: 30, left: 20 }) } }, open, setOpen, { width: 400, height: 260 });
  assert.equal(open, false);
  handleCitationMarkerClick(event, open, setOpen, { width: 400, height: 260 });
  assert.equal(open, true);
});

test("same-boundary sources share one inline control and remain individually accessible in its preview", () => {
  const markup = renderToStaticMarkup(createElement(AnswerPresentation, { answer: "Claim [S4][S1]", citations, onCitationClick() {} }));
  assert.equal((markup.match(/class="citation-marker/g) ?? []).length, 1);
  assert.match(markup, />\[1\]<\/button>/);
  assert.doesNotMatch(markup, /popover="auto"|popovertarget/);
});

test("popover placement clamps within the viewport and prefers space below the marker", () => {
  assert.deepEqual(
    calculateCitationPopoverPosition({ top: 40, bottom: 60, left: 380 }, { width: 300, height: 200 }, { width: 400, height: 600 }),
    { top: 68, left: 88 },
  );
  assert.deepEqual(
    calculateCitationPopoverPosition({ top: 560, bottom: 580, left: 4 }, { width: 300, height: 200 }, { width: 400, height: 600 }),
    { top: 352, left: 12 },
  );
});

test("popover closes for Escape and outside clicks, restores focus for Escape, and opens the chosen source in the PDF", () => {
  let focused = false;
  const trigger = { focus() { focused = true; } };
  let open = true;
  dismissCitationPopover((value) => { open = value; }, trigger, true);
  assert.equal(open, false);
  assert.equal(focused, true);
  open = true;
  dismissCitationPopover((value) => { open = value; }, trigger);
  assert.equal(open, false);

  const navigated = [];
  open = true;
  openCitationSource(citations[0], (value) => { open = value; }, (citation) => navigated.push(citation), trigger);
  assert.equal(open, false);
  assert.equal(navigated[0].source_id, "S4");
  assert.equal(navigated[0].document_id, "doc-4");
  assert.equal(navigated[0].page_number, 7);
});

test("pointer movement, popup clicks, and popup scrolling stay inside the open citation interaction", () => {
  let open = true;
  const setOpen = (value) => { open = value; };
  const marker = { name: "marker" };
  const popup = { name: "popup" };
  const popupText = { name: "popup text" };
  const asBoundary = (element) => ({ contains: (target) => target === element || (element === popup && target === popupText) });
  const triggerBoundary = asBoundary(marker);
  const popupBoundary = asBoundary(popup);

  // No hover/leave event closes the popover; crossing into portal content is inside its boundary.
  assert.equal(dismissCitationPopoverOutsideInteraction(popupText, [popupText, popup], triggerBoundary, popupBoundary, setOpen), false);
  assert.equal(open, true);
  // Pointerdown and the subsequent click target inside the popup do not dismiss it.
  assert.equal(dismissCitationPopoverOutsideInteraction(popup, [popup, popupText], triggerBoundary, popupBoundary, setOpen), false);
  assert.equal(open, true);
  // The captured scroll event from a scrollable popup descendant also remains inside.
  assert.equal(dismissCitationPopoverOutsideInteraction(popupText, [popupText, popup], triggerBoundary, popupBoundary, setOpen), false);
  assert.equal(open, true);
});

test("outside pointerdown closes the citation preview", () => {
  let open = true;
  const outside = { name: "outside" };
  const boundary = { contains: () => false };
  assert.equal(dismissCitationPopoverOutsideInteraction(outside, [outside], boundary, boundary, (value) => { open = value; }), true);
  assert.equal(open, false);
});

test("clicking the active citation toggles it closed and another citation opens independently", () => {
  let firstOpen = true;
  let secondOpen = false;
  const firstTrigger = { contains: (target) => target === firstTrigger };
  const secondTrigger = { contains: (target) => target === secondTrigger };
  const firstPopup = { contains: () => false };
  const secondPopup = { contains: () => false };

  // Active marker clicks are inside its own boundary, then the marker handler toggles it closed.
  assert.equal(dismissCitationPopoverOutsideInteraction(firstTrigger, [firstTrigger], firstTrigger, firstPopup, (value) => { firstOpen = value; }), false);
  const markerEvent = { currentTarget: { getBoundingClientRect: () => ({ top: 30, bottom: 50, left: 40 }) }, stopPropagation() {} };
  handleCitationMarkerClick(markerEvent, firstOpen, (value) => { firstOpen = value; }, { width: 600, height: 500 });
  assert.equal(firstOpen, false);

  // A second citation is outside the first popup, so the first closes and the second opens.
  assert.equal(dismissCitationPopoverOutsideInteraction(secondTrigger, [secondTrigger], firstTrigger, firstPopup, (value) => { firstOpen = value; }), true);
  handleCitationMarkerClick(markerEvent, secondOpen, (value) => { secondOpen = value; }, { width: 600, height: 500 });
  assert.equal(firstOpen, false);
  assert.equal(secondOpen, true);
});

test("uncited answers render normally without citation controls", () => {
  const answer = "A useful answer without markers.";
  const markup = renderToStaticMarkup(createElement(AnswerPresentation, { answer, citations: [], onCitationClick() {} }));
  assert.match(markup, /A useful answer without markers\./);
  assert.doesNotMatch(markup, /citation-marker|citation-popover|References|<details/);
});
