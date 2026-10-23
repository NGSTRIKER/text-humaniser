"""Split large texts into paragraph-based chunks that fit the LLM context.

The chunker is intentionally simple at this stage:
1. Split on blank lines (``\\n\\n``) → paragraphs.
2. Estimate token counts using the configured chars-per-token ratio.
3. Pack consecutive paragraphs into batches whose total estimated tokens
   stay within ``config.max_prompt_tokens``.
"""

from __future__ import annotations

import re
from typing import Sequence

from humanizer.config import RewriterConfig
from humanizer.logging_config import get_logger
from humanizer.schemas import TextChunk

log = get_logger("chunker")

# Matches one or more blank lines (possibly containing whitespace).
_PARAGRAPH_SEP = re.compile(r"\n\s*\n")


def split_paragraphs(text: str) -> list[str]:
    """Split *text* on blank lines and return non-empty paragraphs.

    Leading/trailing whitespace within each paragraph is stripped, but
    internal whitespace is preserved.
    """
    parts = _PARAGRAPH_SEP.split(text)
    return [p.strip() for p in parts if p.strip()]


def create_chunks(
    text: str,
    config: RewriterConfig | None = None,
) -> list[TextChunk]:
    """Turn *text* into a list of :class:`TextChunk` objects.

    Each chunk contains one or more consecutive paragraphs whose combined
    estimated token count does not exceed ``config.max_prompt_tokens``.

    A single paragraph that exceeds the limit on its own is kept as a
    solo chunk — the LLM backend will need to handle truncation if the
    model can't cope.

    Parameters:
        text: The full input document.
        config: Pipeline configuration (defaults are used if ``None``).

    Returns:
        Ordered list of ``TextChunk`` instances covering the entire input.
    """
    if config is None:
        config = RewriterConfig()

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    max_tokens = config.max_prompt_tokens

    # Build a map from paragraph text → (char_start, char_end) in original.
    offsets = _compute_offsets(text, paragraphs)

    chunks: list[TextChunk] = []
    batch_paras: list[int] = []  # indices into `paragraphs`
    batch_tokens: int = 0

    for i, para in enumerate(paragraphs):
        para_tokens = config.estimate_tokens(para)

        # If adding this paragraph would bust the limit, flush the batch.
        if batch_paras and (batch_tokens + para_tokens) > max_tokens:
            chunks.append(_build_chunk(len(chunks), paragraphs, batch_paras, offsets))
            batch_paras = []
            batch_tokens = 0

        batch_paras.append(i)
        batch_tokens += para_tokens

    # Flush remaining.
    if batch_paras:
        chunks.append(_build_chunk(len(chunks), paragraphs, batch_paras, offsets))

    log.info(
        "Chunked %d paragraphs into %d chunks (max_prompt_tokens=%d)",
        len(paragraphs),
        len(chunks),
        max_tokens,
    )
    return chunks


# ── Internal helpers ─────────────────────────────────────────────


def _compute_offsets(
    text: str, paragraphs: list[str]
) -> list[tuple[int, int]]:
    """Return ``(char_start, char_end)`` for each paragraph in *text*.

    Uses a simple forward scan.  If a paragraph can't be found (shouldn't
    happen with our splitting logic), it falls back to ``(0, 0)``.
    """
    offsets: list[tuple[int, int]] = []
    search_start = 0
    for para in paragraphs:
        idx = text.find(para, search_start)
        if idx == -1:
            # Fallback — should never happen.
            offsets.append((0, 0))
        else:
            offsets.append((idx, idx + len(para)))
            search_start = idx + len(para)
    return offsets


def _build_chunk(
    chunk_index: int,
    paragraphs: list[str],
    para_indices: Sequence[int],
    offsets: list[tuple[int, int]],
) -> TextChunk:
    """Assemble a :class:`TextChunk` from a list of paragraph indices."""
    texts = [paragraphs[i] for i in para_indices]
    joined = "\n\n".join(texts)
    char_start = offsets[para_indices[0]][0]
    char_end = offsets[para_indices[-1]][1]
    return TextChunk(
        index=chunk_index,
        text=joined,
        char_start=char_start,
        char_end=char_end,
        paragraph_indices=tuple(para_indices),
    )
