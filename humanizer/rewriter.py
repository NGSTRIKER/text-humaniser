"""Rewriter pipeline orchestrator.

Reads input text → rule engine → chunks → LLM per chunk → reassembles.

Stage 7 ``structured`` mode sends each ORIGINAL chunk with its rule
suggestions, protected facts and a compact analysis as JSON, validates the
JSON reply, re-asks on bad format, and falls back to the rule-edited chunk
if the model still fails.  ``plain`` mode keeps the Stage 1 behaviour.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from humanizer.analyzer import AnalyzerUnavailableError
from humanizer.chunker import create_chunks
from humanizer.config import RewriterConfig
from humanizer.config import Intensity
from humanizer.edit_history import apply_edits
from humanizer.llm_backend import LLMError, LLMMemoryError, OllamaBackend, get_system_prompt
from humanizer.llm_request import (
    OUTPUT_SCHEMA, SYSTEM_PROMPT, build_request, suggestions_for_chunk, summarize_analysis,
)
from humanizer.llm_response import REPAIR_INSTRUCTION, OutputFormatError, parse_response
from humanizer.logging_config import get_logger
from humanizer.styles import effective_tone
from humanizer.schemas import EditRecord, ChunkResult, RewriteRequest, RewriteResult, RewriteStatus

log = get_logger("rewriter")


class RewriterPipeline:
    """Top-level orchestrator for text rewriting.

    Parameters:
        config: Pipeline configuration.
        backend: An LLM backend instance; if *None*, an
                 :class:`OllamaBackend` is created from *config*.
    """

    def __init__(
        self,
        config: RewriterConfig | None = None,
        backend: object | None = None,
    ) -> None:
        self.config = config or RewriterConfig()
        self.backend = backend or OllamaBackend(self.config)
        self._engine = None
        self._last_analysis = None
        self._validator = None

    # ── Public API ───────────────────────────────────────────────

    def rewrite_text(self, request: RewriteRequest) -> RewriteResult:
        """Rewrite the text described by *request*.

        Returns a :class:`RewriteResult` containing the reassembled
        output and per-chunk diagnostics.
        """
        t0 = time.monotonic()
        text = request.text.strip()

        if not text:
            return RewriteResult(
                original_text=request.text,
                rewritten_text=request.text,
                status=RewriteStatus.SKIPPED,
            )

        edits, text = self._apply_rules(text, request)
        if self.config.rules_only:
            return RewriteResult(
                original_text=request.text,
                rewritten_text=text,
                status=RewriteStatus.SUCCESS,
                duration_seconds=time.monotonic() - t0,
                edits=edits,
                rule_text=text,
            )

        structured = self.config.llm_mode == "structured"
        source = request.text.strip() if structured else text
        chunks = create_chunks(source, self.config)
        if not chunks:
            return RewriteResult(
                original_text=request.text,
                rewritten_text=request.text,
                status=RewriteStatus.SKIPPED,
            )

        system = get_system_prompt(self.config.intensity) + " Style: " + effective_tone(self.config.style, self.config.tone)
        chunk_results: list[ChunkResult] = []
        success_count = 0

        memory_failure = False
        for chunk in chunks:
            if memory_failure:  # don't hammer a model that cannot load
                cr = ChunkResult(chunk=chunk, rewritten_text=self._fallback(chunk, edits, structured),
                                 success=False, error="skipped after memory error", model=self.config.model_name)
            elif structured:
                cr = self._rewrite_chunk_structured(chunk, edits, request)
            else:
                cr = self._rewrite_chunk(chunk, system, request.seed)
                if cr.success:
                    cr = self._validate_chunk(cr, chunk.text, None)
            memory_failure = memory_failure or (cr.error or "").startswith("memory:")
            chunk_results.append(cr)
            if cr.success:
                success_count += 1

        # Reassemble — use rewritten text for successful chunks,
        # original text for failed ones.
        assembled = "\n\n".join(cr.rewritten_text for cr in chunk_results)

        if success_count == len(chunks):
            status = RewriteStatus.SUCCESS
        elif success_count > 0:
            status = RewriteStatus.PARTIAL
        else:
            status = RewriteStatus.FAILED

        elapsed = time.monotonic() - t0
        log.info(
            "Rewrite complete: %d/%d chunks OK  (%.1fs)",
            success_count,
            len(chunks),
            elapsed,
        )

        return RewriteResult(
            original_text=request.text,
            rewritten_text=assembled,
            status=status,
            chunk_results=chunk_results,
            total_chunks=len(chunks),
            successful_chunks=success_count,
            duration_seconds=elapsed,
            edits=edits,
            rule_text=text,
        )

    def _apply_rules(self, text: str, request: RewriteRequest) -> tuple[list[EditRecord], str]:
        """Run the Stage 3 rule engine; on any analyzer problem keep the text."""
        if not self.config.rules_enabled:
            return [], text
        try:
            if self._engine is None:
                from humanizer.rule_engine import RuleEngine

                self._engine = RuleEngine(self.config)
            result = self._engine.run(
                text,
                intensity=request.intensity,
                protected_terms=request.protected_terms,
                seed=request.seed,
            )
        except AnalyzerUnavailableError as exc:
            log.warning("Rule engine skipped: %s", exc)
            return [], text
        self._last_analysis = result.analysis
        return result.history.records, result.text

    def rewrite_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        *,
        intensity: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> RewriteResult:
        """Read a file, rewrite it, and optionally write the result.

        Parameters:
            input_path: Path to the input text file.
            output_path: Where to write the result.  ``None`` = don't write.
            intensity: Override the configured intensity.
            seed: Override the configured seed.

        Returns:
            The :class:`RewriteResult`.
        """
        input_path = Path(input_path)
        text = input_path.read_text(encoding="utf-8")
        log.info(
            "Read %d chars (%d words) from %s",
            len(text),
            len(text.split()),
            input_path.name,
        )

        level = intensity or self.config.intensity.value
        seed = seed if seed is not None else self.config.seed
        if input_path.suffix.lower() in (".md", ".markdown"):
            result = self._rewrite_markdown(text, level, seed)
        else:
            result = self.rewrite_text(RewriteRequest(text=text, intensity=level, seed=seed))

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.rewritten_text, encoding="utf-8")
            log.info("Wrote %d chars to %s", len(result.rewritten_text), output_path.name)

        return result

    def _rewrite_markdown(self, text: str, intensity: str, seed: Optional[int]) -> RewriteResult:
        """Rewrite only prose; headings, code, lists, tables etc. stay exact.

        Consecutive prose paragraphs are rewritten together so the model and
        the paragraph-flow rules still see their neighbours."""
        from humanizer.markdown import protected_inline, split_blocks

        t0 = time.monotonic()
        out: list[str] = []
        group: list[str] = []
        all_edits: list[EditRecord] = []
        chunk_results: list[ChunkResult] = []
        statuses: list[RewriteStatus] = []

        def flush() -> None:
            if not group:
                return
            prose = "\n\n".join(group)
            res = self.rewrite_text(RewriteRequest(
                text=prose, intensity=intensity, seed=seed, protected_terms=protected_inline(prose)))
            out.append(res.rewritten_text.strip())
            all_edits.extend(res.edits)
            chunk_results.extend(res.chunk_results)
            statuses.append(res.status)
            group.clear()

        for block in split_blocks(text):
            if block.prose:
                group.append(block.text)
            else:
                flush()
                out.append(block.text)
        flush()
        if not statuses or all(s == RewriteStatus.SUCCESS for s in statuses):
            status = RewriteStatus.SUCCESS if statuses else RewriteStatus.SKIPPED
        elif any(s in (RewriteStatus.SUCCESS, RewriteStatus.PARTIAL) for s in statuses):
            status = RewriteStatus.PARTIAL
        else:
            status = RewriteStatus.FAILED
        ok = sum(c.success for c in chunk_results)
        return RewriteResult(
            original_text=text, rewritten_text="\n\n".join(out) + "\n", status=status,
            chunk_results=chunk_results, total_chunks=len(chunk_results), successful_chunks=ok,
            duration_seconds=time.monotonic() - t0, edits=all_edits,
        )

    # ── Internals ────────────────────────────────────────────────

    def _fallback(self, chunk, edits: list[EditRecord], structured: bool) -> str:
        """Rule-edited version of an original-text chunk (or the chunk itself)."""
        if not structured:
            return chunk.text
        inside = [
            EditRecord(**{**e.__dict__, "char_start": e.char_start - chunk.char_start,
                          "char_end": e.char_end - chunk.char_start})
            for e in edits
            if e.accepted and e.user_decision is not False
            and chunk.char_start <= e.char_start and e.char_end <= chunk.char_end
        ]
        try:
            return apply_edits(chunk.text, inside)
        except ValueError:
            return chunk.text

    def _rewrite_chunk_structured(self, chunk, edits: list[EditRecord], request: RewriteRequest) -> ChunkResult:
        """JSON request → validated JSON reply, with format-repair retries."""
        t0 = time.monotonic()
        summary = summarize_analysis(self._last_analysis, chunk.char_start, chunk.char_end)
        try:
            req = build_request(
                chunk.text, self.config,
                intensity=Intensity(request.intensity) if request.intensity else None,
                suggestions=suggestions_for_chunk(edits, chunk.char_start, chunk.char_end),
                analysis_summary=summary,
                protected_terms=request.protected_terms,
            )
        except ValueError as exc:
            log.error("Chunk %d not sent to the model: %s", chunk.index, exc)
            return ChunkResult(chunk=chunk, rewritten_text=self._fallback(chunk, edits, True),
                               success=False, error=str(exc), model=self.config.model_name)
        prompt = req.to_prompt()
        out_tokens = min(self.config.max_tokens,
                         int(self.config.estimate_tokens(chunk.text) * self.config.max_expansion_ratio * 2) + 128)
        history: list[dict[str, str]] = []
        error = None
        attempts = 0
        for attempts in range(1, self.config.format_retries + 2):
            try:
                raw = self.backend.generate(
                    prompt=prompt, system=SYSTEM_PROMPT, seed=request.seed,
                    max_tokens=out_tokens, json_schema=OUTPUT_SCHEMA, history=history,
                )
            except LLMMemoryError as exc:
                error = f"memory: {exc}"
                break
            except LLMError as exc:
                error = str(exc)
                break
            try:
                parsed = parse_response(raw, req)
            except OutputFormatError as exc:
                error = f"invalid output: {exc}"
                log.warning("Chunk %d attempt %d: %s", chunk.index, attempts, exc)
                history = [{"role": "user", "content": prompt}, {"role": "assistant", "content": raw[:2000]}]
                prompt = REPAIR_INSTRUCTION.format(error=exc)
                continue
            cr = ChunkResult(
                chunk=chunk, rewritten_text=parsed.rewritten_text, success=True,
                model=self.config.model_name, duration_seconds=time.monotonic() - t0,
                applied_suggestions=parsed.applied, rejected_suggestions=parsed.rejected,
                uncertain=parsed.uncertain, warnings=parsed.warnings, attempts=attempts,
            )

            def repair(candidate: str, report) -> str | None:
                return self._repair_structured(req, candidate, report, request.seed, out_tokens)

            return self._validate_chunk(cr, self._fallback(chunk, edits, True), repair)
        log.error("Chunk %d fell back to rule output: %s", chunk.index, error)
        return ChunkResult(
            chunk=chunk, rewritten_text=self._fallback(chunk, edits, True), success=False,
            error=error, model=self.config.model_name, duration_seconds=time.monotonic() - t0,
            attempts=attempts, source="rules",
        )

    # ── Stage 8: validation and repair ───────────────────────────

    @property
    def validator(self):
        if self._validator is None:
            from humanizer.validation import Validator
            self._validator = Validator(self.config)
        return self._validator

    def _validate_chunk(self, cr: ChunkResult, rule_version: str, repair) -> ChunkResult:
        """Validate a successful LLM chunk; repair or fall back if it fails."""
        if not self.config.validation_enabled:
            return cr
        from humanizer.decision import decide

        fallbacks = [("rules", rule_version)] if rule_version.strip() != cr.chunk.text.strip() else []
        text, report, source, repairs = decide(
            cr.chunk.text, cr.rewritten_text, self.validator,
            repair=repair, max_repairs=self.config.max_retry_count, fallbacks=fallbacks,
        )
        cr.rewritten_text, cr.validation, cr.source = text, report, source
        cr.attempts += repairs
        if source not in {"llm", "repaired"}:
            cr.success = False
            cr.error = f"validation failed; used {source} version"
        return cr

    def _repair_structured(self, req, candidate: str, report, seed, out_tokens) -> str | None:
        """Ask the model to fix only the listed problems; ``None`` on failure."""
        problems = report.repair_suggestions or report.errors or report.warnings
        payload = {
            "task": "repair",
            "original": req.text,
            "previous_rewrite": candidate,
            "problems": problems[:10],
            "instructions": "Fix ONLY these problems. Keep the rest of previous_rewrite. "
                            "Reply with one JSON object matching output_schema.",
            "protected": req.protected,
            "max_output_chars": req.max_output_chars,
            "output_schema": OUTPUT_SCHEMA,
        }
        import json as _json
        try:
            raw = self.backend.generate(
                prompt=_json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                system=SYSTEM_PROMPT, seed=seed, max_tokens=out_tokens, json_schema=OUTPUT_SCHEMA,
            )
            return parse_response(raw, req).rewritten_text
        except (LLMError, OutputFormatError) as exc:
            log.warning("Repair failed: %s", exc)
            return None

    def _rewrite_chunk(
        self,
        chunk,
        system: str,
        seed: int | None,
    ) -> ChunkResult:
        """Send a single chunk to the LLM and return a :class:`ChunkResult`."""
        t0 = time.monotonic()
        try:
            rewritten = self.backend.generate(
                prompt=chunk.text,
                system=system,
                seed=seed,
            )
            elapsed = time.monotonic() - t0
            return ChunkResult(
                chunk=chunk,
                rewritten_text=rewritten,
                success=True,
                model=self.config.model_name,
                duration_seconds=elapsed,
            )
        except LLMError as exc:
            elapsed = time.monotonic() - t0
            log.error("Chunk %d failed: %s", chunk.index, exc)
            return ChunkResult(
                chunk=chunk,
                rewritten_text=chunk.text,  # fallback to original
                success=False,
                error=str(exc),
                model=self.config.model_name,
                duration_seconds=elapsed,
            )
