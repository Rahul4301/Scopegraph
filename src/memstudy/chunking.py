"""Verbatim chunking shared by the arms. Nothing is cut: every character of the input ends up in
exactly one chunk, and a line longer than the limit is split into token windows, not dropped."""

from __future__ import annotations

from memstudy.tokens import count_tokens, encoding


def split_oversized(text: str, max_tokens: int) -> list[str]:
    """Split one very long line into token windows. All content is kept (no truncation)."""
    enc = encoding()
    ids = enc.encode(text, disallowed_special=())
    return [enc.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)]


def chunk_document(text: str, max_tokens: int) -> list[str]:
    """Pack whole lines into chunks of at most max_tokens (a single over-long line is split into
    windows). Chunks are verbatim; joining them with a newline restores the text."""
    lines: list[str] = []
    for line in text.split("\n"):
        if count_tokens(line) > max_tokens:
            lines.extend(split_oversized(line, max_tokens))
        else:
            lines.append(line)
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for line in lines:
        n = count_tokens(line) + 1
        if buffer and size + n > max_tokens:
            chunks.append("\n".join(buffer))
            buffer, size = [], 0
        buffer.append(line)
        size += n
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks
