"""Copperline Finance RAG tool.

Answers questions about Copperline Manufacturing's finance documents
(week-03/copperline-docs) using retrieval-augmented generation:

1. Load each document and split it into one chunk per section.
2. Find the sections most relevant to the question (BM25 keyword search).
3. Send those sections to Claude, which answers from them and cites its sources.

Usage (from the top of the repository):
    .venv\\Scripts\\python.exe week-03\\rag.py                 # interactive: ask questions one after another
    .venv\\Scripts\\python.exe week-03\\rag.py "question"      # answer one question
    .venv\\Scripts\\python.exe week-03\\rag.py --chunks        # list every chunk (no API call)
    .venv\\Scripts\\python.exe week-03\\rag.py --search "q"    # show which chunks a question retrieves (no API call)

The API key is read from the ANTHROPIC_API_KEY environment variable or, if that
is not set, from "API Key.txt" in a folder above the repository. It is never
stored in this repository.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic
import pdfplumber
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "copperline-docs"
KEY_FILE_NAME = "API Key.txt"  # lives in the workspace folder, outside the repo

MODEL = "claude-opus-5-5"
# Number of chunks sent to Claude per question. Keyword search can rank the
# right section a little low (e.g. "$30,000" doesn't match the "$10,000-$50,000"
# tier), and sections are short, so sending 8 is cheap insurance.
TOP_K = 8

# Documents whose first section defines terms the other sections rely on.
# That definition is added to every other chunk from the document.
DEFINITION_SECTION_DOCS = {"month-end-close-calendar.html"}


@dataclass
class Chunk:
    chunk_id: str  # e.g. "travel-expense-policy#meal-limits-by-city-tier"
    source: str  # file name
    title: str  # document title
    meta: str  # version / owner line
    section: str  # section heading
    body: str  # section text
    context: str = ""  # extra note needed to read the section on its own

    @property
    def label(self) -> str:
        return f"{self.title} > {self.section}"

    def text(self) -> str:
        """Self-contained text: header + optional context + body."""
        parts = [f"Document: {self.title} ({self.meta})", f"Section: {self.section}"]
        if self.context:
            parts.append(f"Context: {self.context}")
        parts.append("")
        parts.append(self.body)
        return "\n".join(parts)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- loaders


def load_html(path: Path) -> list[Chunk]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    main = soup.find("main")  # skips the banner, footer and <style>
    title = clean(main.find("h1").get_text())
    meta_tag = main.find("p", class_="meta")
    meta = clean(meta_tag.get_text()) if meta_tag else ""

    sections: list[tuple[str, list[str]]] = []
    for el in main.find_all(["h2", "p", "ul"], recursive=False):
        if el.name == "h2":
            sections.append((clean(el.get_text()), []))
        elif not sections:
            continue  # title/meta before the first heading are already captured
        elif el.name == "ul":
            # keep list items together, as "- " bullets, in the same chunk as their lead-in
            items = [f"- {clean(li.get_text())}" for li in el.find_all("li")]
            sections[-1][1].append("\n".join(items))
        else:
            sections[-1][1].append(clean(el.get_text()))

    context = ""
    if path.name in DEFINITION_SECTION_DOCS and sections:
        context = " ".join(sections[0][1])

    return [
        Chunk(
            chunk_id=f"{path.stem}#{slugify(heading)}",
            source=path.name,
            title=title,
            meta=meta,
            section=heading,
            body="\n\n".join(paragraphs),
            context=context if i > 0 else "",  # the defining section needs no repeat
        )
        for i, (heading, paragraphs) in enumerate(sections)
    ]


def load_pdf(path: Path) -> list[Chunk]:
    """Split a PDF into sections using fonts: bold lines of 15pt+ are the
    letterhead and title, bold 11-15pt lines are section headings, and lines
    under 9pt are page footers."""
    lines = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for line in page.extract_text_lines(return_chars=True):
                char = line["chars"][0]
                bold = "bold" in char["fontname"].lower()
                lines.append((line["text"].strip(), bold, char["size"]))

    # The title is the last large bold line; anything above it is letterhead.
    big = [i for i, (_, bold, size) in enumerate(lines) if bold and size >= 15]
    title_idx = big[-1] if big else -1
    title = lines[title_idx][0] if big else path.stem.replace("-", " ").title()

    meta = ""
    sections: list[tuple[str, list[str]]] = []
    for text, bold, size in lines[title_idx + 1 :]:
        if size < 9:  # page footer ("... Page 1 of 1")
            continue
        if bold and size >= 11:  # section heading
            sections.append((text, []))
        elif not sections:
            meta = clean(f"{meta} {text}")  # version line under the title
        else:
            sections[-1][1].append(text)

    return [
        Chunk(
            chunk_id=f"{path.stem}#{slugify(heading)}",
            source=path.name,
            title=title,
            meta=meta,
            section=heading,
            body=clean(" ".join(body_lines)),  # re-join lines wrapped by the page
        )
        for heading, body_lines in sections
    ]


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() == ".html":
            chunks.extend(load_html(path))
        elif path.suffix.lower() == ".pdf":
            chunks.extend(load_pdf(path))
    return chunks


# -------------------------------------------------------------- retrieval

STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it its me my "
    "of on or our should so that the their them then there this to was we what "
    "when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9$]+(?:[.,][0-9]+)*", text.lower())
    # Crude stemming: cut words to 6 letters so "approves", "approval" and
    # "approved" all match. Numbers and amounts are kept whole.
    return [w if any(ch.isdigit() for ch in w) else w[:6] for w in words if w not in STOPWORDS]


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text()) for c in chunks])

    def search(self, question: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [(c, s) for c, s in ranked[:k] if s > 0]


# ------------------------------------------------------------- generation

SYSTEM_PROMPT = """You answer employee questions about Copperline Manufacturing's finance policies.

Use only the document excerpts provided with each question. Each excerpt is labeled with an id like [1].
- Answer directly and concisely, in plain language, with the specific amounts, deadlines and approvers that apply.
- Cite the excerpts you rely on with their ids, e.g. [1] or [2][4], and name the document and section the first time you cite it.
- If the question combines rules from more than one document, apply all of them.
- If the excerpts don't answer the question, say so and suggest contacting finance-help@copperline.example. Do not guess or use outside knowledge about other companies' policies."""


def load_api_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return None  # the SDK reads the environment variable itself
    for folder in HERE.parents:
        key_file = folder / KEY_FILE_NAME
        if key_file.is_file():
            return key_file.read_text(encoding="utf-8").strip()
    sys.exit(f"No API key found. Set ANTHROPIC_API_KEY or put the key in '{KEY_FILE_NAME}' above the repository.")


def build_user_message(question: str, hits: list[tuple[Chunk, float]]) -> str:
    excerpts = "\n\n".join(
        f'<excerpt id="{i}" source="{c.source}">\n{c.text()}\n</excerpt>'
        for i, (c, _) in enumerate(hits, start=1)
    )
    return f"<excerpts>\n{excerpts}\n</excerpts>\n\nQuestion: {question}"


def answer(client: anthropic.Anthropic, retriever: Retriever, question: str) -> None:
    hits = retriever.search(question)
    if not hits:
        print("I couldn't find anything about that in the Copperline finance documents.")
        return

    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=4000,
        output_config={"effort": "medium"},
        # if a safety classifier declines, re-run on Anthropic's recommended fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, hits)}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        print("\n[Claude declined to answer this question.]")
    print("\n\nSources:")
    for i, (chunk, _) in enumerate(hits, start=1):
        print(f"  [{i}] {chunk.label}  ({chunk.source})")


# -------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about Copperline's finance documents.")
    parser.add_argument("question", nargs="*", help="question to answer (omit for interactive mode)")
    parser.add_argument("--chunks", action="store_true", help="list every chunk and exit (no API call)")
    parser.add_argument("--search", metavar="QUESTION", help="show retrieved chunks and exit (no API call)")
    args = parser.parse_args()

    chunks = load_chunks()
    retriever = Retriever(chunks)

    if args.chunks:
        for c in chunks:
            print(f"=== {c.chunk_id}\n{c.text()}\n")
        print(f"{len(chunks)} chunks from {len({c.source for c in chunks})} documents")
        return

    if args.search:
        for c, score in retriever.search(args.search):
            print(f"{score:6.2f}  {c.label}  ({c.chunk_id})")
        return

    client = anthropic.Anthropic(api_key=load_api_key())

    if args.question:
        answer(client, retriever, " ".join(args.question))
        return

    print(f"Copperline Finance Q&A ({len(chunks)} sections loaded). Type a question, or 'quit' to exit.")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question:
            print()
            answer(client, retriever, question)


if __name__ == "__main__":
    main()
