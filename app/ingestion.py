
from pathlib import Path
import re
import io

import fitz
import pandas as pd
from docx import Document

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


SUPPORTED = {".pdf", ".docx", ".txt", ".csv"}


def extract_file(filename: str, raw: bytes) -> list[dict]:
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}")

    pages = []

    if ext == ".pdf":
        with fitz.open(stream=raw, filetype="pdf") as pdf:
            for i, page in enumerate(pdf):
                text = page.get_text("text")
                if text.strip():
                    pages.append({"text": text, "page": i + 1})

        if not pages:
            raise ValueError(
                "No selectable text was found in this PDF. "
                "It may be scanned; OCR is not enabled yet."
            )

    elif ext == ".docx":
        doc = Document(io.BytesIO(raw))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]

        for table in doc.tables:
            for row in table.rows:
                parts.append(
                    " | ".join(cell.text.strip() for cell in row.cells)
                )

        pages = [{"text": "\n\n".join(parts), "page": 1}]

    elif ext == ".txt":
        pages = [{
            "text": raw.decode("utf-8-sig", errors="replace"),
            "page": 1
        }]

    elif ext == ".csv":
        df = pd.read_csv(io.BytesIO(raw)).fillna("")
        lines = ["Columns: " + ", ".join(map(str, df.columns))]

        for idx, row in df.iterrows():
            lines.append(
                f"Row {idx + 1}: "
                + "; ".join(f"{col}: {row[col]}" for col in df.columns)
            )

        pages = [{"text": "\n".join(lines), "page": 1}]

    return pages


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long_sentence(sentence: str, chunk_size: int, overlap: int) -> list[str]:
    """Split an unusually long sentence without exceeding chunk_size."""
    chunks = []
    start = 0

    while start < len(sentence):
        end = min(start + chunk_size, len(sentence))

        if end < len(sentence):
            boundary = sentence.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        piece = sentence[start:end].strip()

        if piece:
            chunks.append(piece)

        if end >= len(sentence):
            break

        start = max(start + 1, end - overlap)

        while start < len(sentence) and sentence[start].isspace():
            start += 1

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:

    text = clean_text(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    # Keep separate paragraphs separate to avoid mixing unrelated topics.
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n+", text)
        if p.strip()
    ]

    chunks = []

    for paragraph in paragraphs:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", paragraph)
            if s.strip()
        ]

        current = ""

        for sentence in sentences:

            if len(sentence) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""

                chunks.extend(
                    split_long_sentence(sentence, chunk_size, overlap)
                )
                continue

            candidate = f"{current} {sentence}".strip()

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

                # Carry a small overlap from the previous chunk.
                prefix = current[-overlap:].lstrip() if overlap else ""
                candidate = f"{prefix} {sentence}".strip()

                current = (
                    candidate
                    if len(candidate) <= chunk_size
                    else sentence
                )
            else:
                current = sentence

        if current:
            chunks.append(current)

    return chunks


def prepare_documents(
    filename: str,
    raw: bytes,
    domain: str
) -> list[dict]:

    prepared = []

    for page in extract_file(filename, raw):
        page_chunks = chunk_text(page["text"])

        for chunk_index, chunk in enumerate(page_chunks):
            prepared.append({
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "domain": domain,
                    "page": int(page["page"]),
                    "chunk_index": int(chunk_index)
                }
            })

    return prepared
