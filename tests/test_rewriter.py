"""Tests for humanizer.rewriter (the orchestrator).

Unit tests use a fake LLM backend.  Integration tests hit real Ollama.
"""

import pytest

from humanizer.config import Intensity, RewriterConfig
from humanizer.llm_backend import LLMError
from humanizer.rewriter import RewriterPipeline
from humanizer.schemas import RewriteRequest, RewriteStatus


PLAIN = RewriterConfig(llm_mode="plain", validation_enabled=False)  # these tests cover the Stage 1 text-in/text-out path

# ── Fake backend for unit tests ─────────────────────────────────


class FakeBackend:
    """Deterministic mock that echoes text in uppercase."""

    def __init__(self, *, fail: bool = False):
        self._fail = fail
        self.call_count = 0

    def health_check(self) -> bool:
        return not self._fail

    def generate(
        self,
        prompt: str,
        system: str = "",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        self.call_count += 1
        if self._fail:
            raise LLMError("Fake failure")
        # Simple transform: uppercase the prompt text.
        return prompt.upper()


# ── Unit tests ───────────────────────────────────────────────────


class TestRewriterPipelineUnit:
    """Test orchestrator logic with a fake backend."""

    def test_empty_text_is_skipped(self):
        pipeline = RewriterPipeline(config=PLAIN, backend=FakeBackend())
        result = pipeline.rewrite_text(RewriteRequest(text=""))
        assert result.status == RewriteStatus.SKIPPED
        assert result.total_chunks == 0

    def test_whitespace_only_is_skipped(self):
        pipeline = RewriterPipeline(config=PLAIN, backend=FakeBackend())
        result = pipeline.rewrite_text(RewriteRequest(text="   \n\n  "))
        assert result.status == RewriteStatus.SKIPPED

    def test_single_paragraph_rewrites(self):
        backend = FakeBackend()
        pipeline = RewriterPipeline(config=PLAIN, backend=backend)
        result = pipeline.rewrite_text(
            RewriteRequest(text="Hello world.")
        )
        assert result.status == RewriteStatus.SUCCESS
        assert result.total_chunks == 1
        assert result.successful_chunks == 1
        assert result.rewritten_text == "HELLO WORLD."
        assert backend.call_count == 1

    def test_multi_paragraph_rewrites(self):
        backend = FakeBackend()
        pipeline = RewriterPipeline(config=PLAIN, backend=backend)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = pipeline.rewrite_text(RewriteRequest(text=text))
        assert result.status == RewriteStatus.SUCCESS
        assert result.successful_chunks == result.total_chunks

    def test_failed_chunks_fallback_to_original(self):
        backend = FakeBackend(fail=True)
        pipeline = RewriterPipeline(config=PLAIN, backend=backend)
        text = "This will fail."
        result = pipeline.rewrite_text(RewriteRequest(text=text))
        assert result.status == RewriteStatus.FAILED
        assert result.successful_chunks == 0
        # Fallback: rewritten_text should contain the original.
        assert "This will fail." in result.rewritten_text

    def test_result_preserves_original(self):
        pipeline = RewriterPipeline(config=PLAIN, backend=FakeBackend())
        original = "Keep me safe."
        result = pipeline.rewrite_text(RewriteRequest(text=original))
        assert result.original_text == original

    def test_duration_is_positive(self):
        pipeline = RewriterPipeline(config=PLAIN, backend=FakeBackend())
        result = pipeline.rewrite_text(
            RewriteRequest(text="Something to rewrite.")
        )
        assert result.duration_seconds >= 0.0

    def test_chunking_respects_config(self):
        # Very small chunk limit → each paragraph becomes a separate chunk.
        # Each paragraph is ~20 chars = ~5 tokens at 4 chars/token,
        # so max_prompt_tokens=4 forces each into its own chunk.
        config = RewriterConfig(max_prompt_tokens=4, chars_per_token=4.0, llm_mode="plain", validation_enabled=False)
        backend = FakeBackend()
        pipeline = RewriterPipeline(config=config, backend=backend)
        text = "Alpha bravo charlie.\n\nDelta echo foxtrot.\n\nGolf hotel india."
        result = pipeline.rewrite_text(RewriteRequest(text=text))
        assert result.total_chunks == 3
        assert backend.call_count == 3


# ── Integration tests ────────────────────────────────────────────


@pytest.mark.integration
class TestRewriterPipelineIntegration:
    """End-to-end tests that require a running Ollama instance."""

    @pytest.fixture(autouse=True)
    def _setup(self, default_config: RewriterConfig):
        from humanizer.llm_backend import OllamaBackend

        backend = OllamaBackend(default_config)
        if not backend.health_check():
            backend.close()
            pytest.skip(f"Ollama is not running or model {default_config.model_name!r} is not installed.")
        backend.close()

    def test_rewrite_short_text(self):
        pipeline = RewriterPipeline()
        result = pipeline.rewrite_text(
            RewriteRequest(text="The quick brown fox jumps over the lazy dog.")
        )
        assert result.status in (RewriteStatus.SUCCESS, RewriteStatus.PARTIAL)
        assert len(result.rewritten_text) > 10

    def test_rewrite_multi_paragraph(self, sample_long_text):
        pipeline = RewriterPipeline()
        result = pipeline.rewrite_text(
            RewriteRequest(text=sample_long_text)
        )
        assert result.status in (RewriteStatus.SUCCESS, RewriteStatus.PARTIAL)
        assert result.total_chunks >= 1
        assert result.successful_chunks >= 1
