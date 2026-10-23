"""CLI entry point for the Rewriter AI.

Usage examples::

    # Rewrite a file (output to stdout)
    python -m humanizer.main --input essay.txt

    # Rewrite with heavy intensity, write to file
    python -m humanizer.main -i essay.txt -o rewritten.txt --intensity heavy

    # Deterministic mode
    python -m humanizer.main -i essay.txt --seed 42

    # Pipe from stdin
    echo "Some wordy text" | python -m humanizer.main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from humanizer.config import Intensity, RewriterConfig
from humanizer.llm_backend import OllamaBackend
from humanizer.logging_config import setup_logging, get_logger
from humanizer.rewriter import RewriterPipeline
from humanizer.schemas import RewriteRequest, RewriteStatus

log = get_logger("main")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        prog="rewriter",
        description="Rewrite text using a local LLM to improve naturalness and readability.",
    )
    p.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input text file. If omitted, reads from stdin.",
    )
    p.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file. If omitted, prints to stdout.",
    )
    p.add_argument(
        "--intensity",
        choices=[e.value for e in Intensity],
        default=None,
        help="Rewrite intensity (default: moderate).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model to use (default: qwen3.5:4b).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible output.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM generation temperature (0.0–2.0).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    p.add_argument("--rules-only", action="store_true", help="Apply rule-engine edits only; skip the LLM.")
    p.add_argument("--no-rules", action="store_true", help="Skip the rule engine; LLM only.")
    p.add_argument("--deterministic", action="store_true", help="Always choose the best-scoring edit (no randomness).")
    p.add_argument("--voice", choices=["disabled", "selective", "frequent"], default=None,
                   help="Active/passive conversion mode (default: selective).")
    p.add_argument("--reorder", action="store_true", help="Allow reordering independent sentences (heavy only).")
    p.add_argument("--randomness", type=float, default=None, help="0 (predictable) to 1 (varied) edit choice.")
    p.add_argument("--max-edits", type=int, default=None, help="Absolute maximum number of edits.")
    p.add_argument("--selection-method", choices=["softmax", "linear", "uniform"], default=None)
    p.add_argument("--weights", default=None, metavar="FILE", help="JSON file with scoring weights/settings.")
    p.add_argument("--llm-mode", choices=["structured", "plain"], default=None,
                   help="structured (JSON with rule suggestions, default) or plain text.")
    p.add_argument("--tone", default=None, help="Extra tone instruction for the model.")
    p.add_argument("--style", default=None, choices=["neutral", "casual", "professional", "academic"],
                   help="Writing mode (word choice, contractions, model instruction).")
    p.add_argument("--unload", action="store_true", help="Free the model's memory when finished.")
    p.add_argument("--strictness", choices=["low", "medium", "high"], default=None,
                   help="How strictly rewrites are checked for meaning changes (default: medium).")
    p.add_argument("--no-validate", action="store_true", help="Skip meaning checks (not recommended).")
    p.add_argument("--explain", action="store_true", help="Print the list of edits to stderr.")
    p.add_argument("--protect", action="append", default=[], metavar="TERM", help="Term that must not be changed (repeatable).")
    p.add_argument(
        "--health-check",
        action="store_true",
        help="Only test Ollama connectivity and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Configuration ────────────────────────────────────────────
    overrides: dict = {}
    if args.model:
        overrides["model_name"] = args.model
    if args.intensity:
        overrides["intensity"] = Intensity(args.intensity)
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.verbose:
        overrides["log_level"] = "DEBUG"
    if args.rules_only:
        overrides["rules_only"] = True
    if args.no_rules:
        overrides["rules_enabled"] = False
    if args.deterministic:
        overrides["deterministic"] = True
    if args.voice:
        overrides["voice_mode"] = args.voice
    if args.reorder:
        overrides["sentence_reordering"] = True

    if args.strictness:
        overrides["semantic_strictness"] = args.strictness
    if args.no_validate:
        overrides["validation_enabled"] = False
    if args.llm_mode:
        overrides["llm_mode"] = args.llm_mode
    if args.tone:
        overrides["tone"] = args.tone
    if args.style:
        overrides["style"] = args.style
    if args.unload:
        overrides["unload_model_on_close"] = True
    if args.randomness is not None:
        overrides["randomness"] = args.randomness
    if args.max_edits is not None:
        overrides["max_edits"] = args.max_edits
    if args.selection_method:
        overrides["selection_method"] = args.selection_method

    try:
        config = (RewriterConfig.from_weights_file(args.weights, **overrides)
                  if args.weights else RewriterConfig(**overrides))
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    setup_logging(level=config.log_level, log_file=config.log_file)

    # ── Health check ─────────────────────────────────────────────
    backend = OllamaBackend(config)

    if args.health_check and not config.rules_only:
        ok = backend.health_check()
        if ok:
            log.info("✓ Ollama is reachable and model %r is available.", config.model_name)
            return 0
        else:
            log.error("✗ Cannot reach Ollama or model %r is not loaded.", config.model_name)
            return 1

    # ── Read input ───────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            log.error("Input file not found: %s", input_path)
            return 1
        text = input_path.read_text(encoding="utf-8")
        log.info("Read %d chars from %s", len(text), input_path.name)
    else:
        if sys.stdin.isatty():
            log.error("No input file given and stdin is a terminal. Use -i or pipe text.")
            return 1
        text = sys.stdin.read()
        log.info("Read %d chars from stdin", len(text))

    if not text.strip():
        log.warning("Input is empty; nothing to do.")
        return 0

    # ── Rewrite ──────────────────────────────────────────────────
    pipeline = RewriterPipeline(config=config, backend=backend)
    request = RewriteRequest(
        text=text,
        intensity=config.intensity.value,
        seed=config.seed,
        protected_terms=args.protect,
    )
    try:
        if args.input and Path(args.input).suffix.lower() in (".md", ".markdown"):
            # Markdown: only prose is rewritten; headings, code, lists, tables stay exact.
            result = pipeline._rewrite_markdown(text, config.intensity.value, config.seed, protected_terms=args.protect)
        else:
            result = pipeline.rewrite_text(request)
    finally:
        backend.close()

    if args.explain:
        from humanizer.edit_history import EditHistory

        history = EditHistory(original_text=result.original_text, records=result.edits)
        sys.stderr.write("\n── Edits ──\n" + history.explain(include_rejected=args.verbose) + "\n\n")
        for cr in result.chunk_results:
            notes = []
            if cr.rejected_suggestions:
                notes.append("model rejected: " + "; ".join(f"{r['id']} ({r['reason']})" for r in cr.rejected_suggestions))
            if cr.uncertain:
                notes.append("model unsure about: " + "; ".join(cr.uncertain))
            notes += cr.warnings
            if cr.error:
                notes.append("fallback: " + cr.error)
            v = cr.validation
            if v is not None:
                notes.append(f"validation: {v.decision} via {cr.source} (score {v.overall_score:.2f}, confidence {v.confidence:.2f})")
                notes += [f"meaning: {m}" for m in v.suspected_meaning_changes]
                notes += [w for w in v.warnings[:5]]
            if notes:
                sys.stderr.write(f"[chunk {cr.chunk.index}] " + " | ".join(notes) + "\n")

    # ── Output ───────────────────────────────────────────────────
    # Failed chunks already fall back to the rule-edited (or original) text.
    if result.status == RewriteStatus.FAILED:
        log.error("The model failed on every chunk; output is the rule-engine result.")
    output_text = result.rewritten_text

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        log.info("Wrote %d chars to %s", len(output_text), out_path.name)
    else:
        sys.stdout.write(output_text)
        if not output_text.endswith("\n"):
            sys.stdout.write("\n")

    # ── Summary ──────────────────────────────────────────────────
    log.info(
        "Status: %s | Chunks: %d/%d OK | Time: %.1fs",
        result.status.value,
        result.successful_chunks,
        result.total_chunks,
        result.duration_seconds,
    )
    return 0 if result.status != RewriteStatus.FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
