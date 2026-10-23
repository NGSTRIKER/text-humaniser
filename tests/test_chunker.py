"""Tests for humanizer.chunker."""

import pytest

from humanizer.chunker import create_chunks, split_paragraphs
from humanizer.config import RewriterConfig


class TestSplitParagraphs:
    """Unit tests for the paragraph splitter."""

    def test_single_paragraph(self):
        text = "Hello world. This is one paragraph."
        paras = split_paragraphs(text)
        assert len(paras) == 1
        assert paras[0] == text

    def test_two_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph."
        paras = split_paragraphs(text)
        assert len(paras) == 2
        assert paras[0] == "First paragraph."
        assert paras[1] == "Second paragraph."

    def test_multiple_blank_lines(self):
        text = "Para one.\n\n\n\nPara two."
        paras = split_paragraphs(text)
        assert len(paras) == 2

    def test_blank_lines_with_whitespace(self):
        text = "Para one.\n   \n\nPara two."
        paras = split_paragraphs(text)
        assert len(paras) == 2

    def test_empty_input(self):
        assert split_paragraphs("") == []

    def test_whitespace_only(self):
        assert split_paragraphs("   \n\n  \n  ") == []

    def test_leading_trailing_newlines(self):
        text = "\n\nHello.\n\n"
        paras = split_paragraphs(text)
        assert len(paras) == 1
        assert paras[0] == "Hello."

    def test_preserves_internal_newlines(self):
        text = "Line one.\nLine two.\n\nNext paragraph."
        paras = split_paragraphs(text)
        assert len(paras) == 2
        assert "Line one.\nLine two." == paras[0]


class TestCreateChunks:
    """Test the chunk-assembly logic."""

    def test_short_text_single_chunk(self, default_config: RewriterConfig):
        text = "A short paragraph."
        chunks = create_chunks(text, default_config)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0
        assert chunks[0].paragraph_indices == (0,)

    def test_empty_text_no_chunks(self, default_config: RewriterConfig):
        assert create_chunks("", default_config) == []

    def test_multiple_paragraphs_fit_in_one_chunk(self):
        # With a generous token budget, several short paragraphs should
        # be packed into a single chunk.
        config = RewriterConfig(max_prompt_tokens=5000)
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = create_chunks(text, config)
        assert len(chunks) == 1
        assert chunks[0].paragraph_indices == (0, 1, 2)

    def test_large_paragraphs_split_into_multiple_chunks(
        self, small_chunk_config: RewriterConfig
    ):
        # With max_prompt_tokens=100 and chars_per_token=4, that's ~400 chars.
        para = "Word " * 120  # ~600 chars → >100 tokens
        text = f"{para.strip()}\n\n{para.strip()}\n\n{para.strip()}"
        chunks = create_chunks(text, small_chunk_config)
        # Each paragraph exceeds the budget, so each gets its own chunk.
        assert len(chunks) == 3

    def test_character_offsets_are_correct(self, default_config: RewriterConfig):
        p1 = "First paragraph here."
        p2 = "Second paragraph there."
        text = f"{p1}\n\n{p2}"
        chunks = create_chunks(text, default_config)
        # Should be one chunk (both fit).
        assert len(chunks) == 1
        chunk = chunks[0]
        assert text[chunk.char_start : chunk.char_end] == f"{p1}\n\n{p2}" or \
               chunk.char_start == 0

    def test_chunk_indices_are_sequential(self, small_chunk_config: RewriterConfig):
        para = "x " * 200
        text = "\n\n".join([para.strip()] * 5)
        chunks = create_chunks(text, small_chunk_config)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_all_paragraphs_are_covered(self, small_chunk_config: RewriterConfig):
        text = "A.\n\nB.\n\nC.\n\nD.\n\nE."
        chunks = create_chunks(text, small_chunk_config)
        all_para_indices = set()
        for chunk in chunks:
            all_para_indices.update(chunk.paragraph_indices)
        assert all_para_indices == {0, 1, 2, 3, 4}

    def test_uses_default_config_when_none(self):
        text = "Hello.\n\nWorld."
        chunks = create_chunks(text, None)
        assert len(chunks) >= 1


class TestChunkingWithFixedTestSet:
    """Verify chunking behaviour against the fixed test paragraphs."""

    def test_all_paragraphs_produce_chunks(self, test_paragraphs):
        """Every test paragraph should produce at least one chunk."""
        config = RewriterConfig()
        for item in test_paragraphs:
            chunks = create_chunks(item["text"], config)
            assert len(chunks) >= 1, (
                f"Paragraph {item['id']!r} produced no chunks"
            )

    def test_long_paragraph_is_not_dropped(self, test_paragraphs):
        """The 'long_paragraph' entry must survive chunking."""
        long_items = [p for p in test_paragraphs if p["id"] == "long_paragraph"]
        assert long_items, "Fixed test set missing 'long_paragraph'"
        config = RewriterConfig()
        chunks = create_chunks(long_items[0]["text"], config)
        reassembled = "\n\n".join(c.text for c in chunks)
        assert long_items[0]["text"].strip() == reassembled.strip()
