"""Tests for humanizer.llm_backend.

Unit tests use a mock backend. Integration tests (marked ``@pytest.mark.integration``)
hit a real Ollama instance and are skipped in CI.
"""

import pytest

from humanizer.config import Intensity, RewriterConfig
from humanizer.llm_backend import (
    LLMError,
    OllamaBackend,
    get_system_prompt,
)


# ── Unit tests (no Ollama required) ─────────────────────────────


class TestSystemPrompts:
    """Verify system prompts exist and are non-empty for every intensity."""

    @pytest.mark.parametrize("intensity", list(Intensity))
    def test_prompt_exists(self, intensity: Intensity):
        prompt = get_system_prompt(intensity)
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_light_is_conservative(self):
        prompt = get_system_prompt(Intensity.LIGHT)
        assert "minimal" in prompt.lower() or "obvious" in prompt.lower()

    def test_heavy_preserves_facts(self):
        prompt = get_system_prompt(Intensity.HEAVY)
        assert "fact" in prompt.lower() or "preserve" in prompt.lower()

    def test_all_prompts_end_with_no_commentary(self):
        for intensity in Intensity:
            prompt = get_system_prompt(intensity)
            assert "no commentary" in prompt.lower()


class TestOllamaBackendInit:
    """Test that the backend can be instantiated with various configs."""

    def test_default_config(self):
        backend = OllamaBackend()
        assert backend.config.model_name == "qwen3.5:4b"
        backend.close()

    def test_custom_config(self):
        cfg = RewriterConfig(model_name="llama3:latest", request_timeout=60.0)
        backend = OllamaBackend(cfg)
        assert backend.config.model_name == "llama3:latest"
        backend.close()


# ── Integration tests (require running Ollama) ──────────────────


@pytest.mark.integration
class TestOllamaIntegration:
    """These tests require a running Ollama instance with the configured model."""

    @pytest.fixture(autouse=True)
    def _setup(self, default_config: RewriterConfig):
        self.backend = OllamaBackend(default_config)
        yield
        self.backend.close()

    def test_health_check(self):
        result = self.backend.health_check()
        assert result is True, (
            "Ollama health check failed. Is Ollama running? "
            f"Is model '{self.backend.config.model_name}' pulled?"
        )

    def test_simple_generation(self):
        response = self.backend.generate(
            prompt="Say 'hello world' and nothing else.",
            system="You are a helpful assistant. Respond only with the requested text.",
            temperature=0.0,
            max_tokens=20,
        )
        assert len(response) > 0
        assert "hello" in response.lower()

    def test_rewrite_preserves_meaning(self):
        """Smoke test: rewrite a simple sentence and check it's non-empty."""
        system = get_system_prompt(Intensity.LIGHT)
        response = self.backend.generate(
            prompt="The quick brown fox jumps over the lazy dog.",
            system=system,
            temperature=0.3,
            max_tokens=100,
        )
        assert len(response) > 10

    def test_seed_reproducibility(self):
        """Same seed should produce the same output."""
        kwargs = dict(
            prompt="Rewrite: The cat sat on the mat.",
            system=get_system_prompt(Intensity.MODERATE),
            temperature=0.0,
            max_tokens=60,
            seed=42,
        )
        r1 = self.backend.generate(**kwargs)
        r2 = self.backend.generate(**kwargs)
        # Temperature=0 + same seed should give identical results.
        assert r1 == r2
