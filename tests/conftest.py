"""Shared test fixtures for the rewriter test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from humanizer.config import RewriterConfig


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_PARAGRAPHS_PATH = DATA_DIR / "test_paragraphs.jsonl"


@pytest.fixture()
def default_config() -> RewriterConfig:
    """A RewriterConfig with all defaults."""
    return RewriterConfig()


@pytest.fixture()
def small_chunk_config() -> RewriterConfig:
    """Config with a tiny max_prompt_tokens for testing the chunker."""
    return RewriterConfig(max_prompt_tokens=100, chars_per_token=4.0)


@pytest.fixture()
def test_paragraphs() -> list[dict]:
    """Load the fixed evaluation paragraphs from ``data/test_paragraphs.jsonl``."""
    assert TEST_PARAGRAPHS_PATH.is_file(), (
        f"Test paragraphs file missing: {TEST_PARAGRAPHS_PATH}"
    )
    items = []
    with open(TEST_PARAGRAPHS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


@pytest.fixture()
def sample_short_text() -> str:
    """A short text that fits in a single chunk."""
    return (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence is perfectly fine and needs no changes."
    )


@pytest.fixture()
def sample_long_text() -> str:
    """A multi-paragraph text that should be split into multiple chunks."""
    paragraphs = [
        (
            "The development of modern computing has gone through several "
            "revolutionary phases. Each phase introduced new paradigms that "
            "transformed how we think about information processing."
        ),
        (
            "The first major revolution was the invention of the transistor, "
            "which replaced vacuum tubes and made computers smaller, faster, "
            "and more reliable. This breakthrough laid the foundation for "
            "all subsequent advances in digital technology."
        ),
        (
            "The second revolution came with the development of integrated "
            "circuits, which allowed multiple transistors to be combined "
            "on a single chip. This dramatically reduced the cost and size "
            "of computing devices while increasing their capabilities."
        ),
        (
            "The third revolution was the rise of personal computing in "
            "the 1970s and 1980s. Companies like Apple and IBM brought "
            "computers out of specialized labs and into homes and offices, "
            "fundamentally changing how people work and communicate."
        ),
        (
            "The fourth revolution is still unfolding today with the "
            "emergence of artificial intelligence and machine learning. "
            "These technologies promise to automate complex decision-making "
            "processes and unlock new possibilities in fields ranging from "
            "healthcare to creative arts."
        ),
    ]
    return "\n\n".join(paragraphs)
