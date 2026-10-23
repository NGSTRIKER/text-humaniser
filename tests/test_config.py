"""Tests for humanizer.config."""

import pytest
from pydantic import ValidationError

from humanizer.config import Intensity, RewriterConfig


class TestRewriterConfigDefaults:
    """Verify that defaults are sane for 8 GB / Intel iGPU."""

    def test_default_model(self, default_config: RewriterConfig):
        assert default_config.model_name == "qwen3.5:4b"

    def test_default_intensity(self, default_config: RewriterConfig):
        assert default_config.intensity == Intensity.MODERATE

    def test_default_temperature(self, default_config: RewriterConfig):
        assert 0.0 <= default_config.temperature <= 2.0

    def test_default_ollama_url(self, default_config: RewriterConfig):
        assert default_config.ollama_base_url == "http://localhost:11434"

    def test_default_max_tokens(self, default_config: RewriterConfig):
        assert default_config.max_tokens > 0

    def test_default_context_window(self, default_config: RewriterConfig):
        assert default_config.context_window >= 2048

    def test_default_seed_is_none(self, default_config: RewriterConfig):
        assert default_config.seed is None


class TestRewriterConfigValidation:
    """Verify that invalid values are rejected."""

    def test_negative_temperature(self):
        with pytest.raises(ValidationError):
            RewriterConfig(temperature=-0.1)

    def test_over_max_temperature(self):
        with pytest.raises(ValidationError):
            RewriterConfig(temperature=2.5)

    def test_zero_max_tokens(self):
        with pytest.raises(ValidationError):
            RewriterConfig(max_tokens=0)

    def test_invalid_log_level(self):
        with pytest.raises(ValidationError):
            RewriterConfig(log_level="VERBOSE")

    def test_valid_log_level_case_insensitive(self):
        cfg = RewriterConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"


class TestRewriterConfigHelpers:
    """Test derived helper methods."""

    def test_estimate_tokens(self, default_config: RewriterConfig):
        # With default chars_per_token=4.0, 100 chars ≈ 25 tokens.
        assert default_config.estimate_tokens("a" * 100) == 25

    def test_estimate_tokens_empty(self, default_config: RewriterConfig):
        assert default_config.estimate_tokens("") == 1  # min 1

    def test_available_input_tokens(self, default_config: RewriterConfig):
        available = default_config.available_input_tokens()
        # Must be positive and less than context_window.
        assert 256 <= available < default_config.context_window


class TestIntensityEnum:
    """Sanity-check the intensity enum."""

    def test_values(self):
        assert set(Intensity) == {Intensity.LIGHT, Intensity.MODERATE, Intensity.HEAVY}

    def test_from_string(self):
        assert Intensity("light") == Intensity.LIGHT
        assert Intensity("moderate") == Intensity.MODERATE
        assert Intensity("heavy") == Intensity.HEAVY
