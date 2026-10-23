# Rewriter AI

Hybrid rule-based + local LLM text humanizer.

## Setup

    uv sync --all-extras
    uv run python -m spacy download en_core_web_sm
    uv run python -m nltk.downloader wordnet omw-1.4
    # optional, for word-vector ranking of synonyms (~40 MB): uv run python -m spacy download en_core_web_md

## Run

    uv run rewriter --help
    uv run python -m humanizer.analyzer "Text to inspect."   # Stage 2 analyzer debug view

    # Stage 3 rule engine only (no LLM needed), showing every edit:
    echo "Due to the fact that the end result was very good, we left." | uv run rewriter --rules-only --explain
    # Options: --intensity light|moderate|heavy  --seed 42  --deterministic  --protect "Term"  --no-rules
    # Word swaps for repeated words run at moderate/heavy; turn on vectors with REWRITER_USE_EMBEDDINGS=true
    # Stage 5: --voice disabled|selective|frequent   --reorder (heavy only; off by default)
    uv run python -m humanizer.paragraph_flow "Some text. It continues."   # sentence dependency report

## Test

    uv run pytest -m "not integration"

## Progress

- [x] Stage 1: Foundation
- [x] Stage 2: Linguistic analyzer (`humanizer/analyzer.py`)
- [x] Stage 3: Basic rule engine (`rule_engine.py`, `transformations/`, `weighting.py`, `edit_history.py`)
- [x] Stage 4: Word relationship graph (`word_graph.py`, `context_analyzer.py`, `morphology.py`, `transformations/synonyms.py`)
- [x] Stage 5: Structural transformations (`transformations/voice.py`, `word_order.py`, `sentence_structure.py`, `paragraph_flow.py`)
- [x] Stage 6: Probabilistic selection (`profiles.py`, `probability.py`, `conflicts.py`, `weighting.py`)
- [x] Stage 7: Local LLM integration (`llm_request.py`, `llm_response.py`, `llm_backend.py`, `rewriter.py`)
- [x] Stage 8: Semantic validation (`validation/`, `decision.py`)
- [x] Stage 9: End-to-end evaluation (`evaluation.py`, `data/eval_pairs.jsonl`)
- [x] Stage 10: User interface and feedback (`web/`, `session.py`, `feedback.py`)
- [x] Stage 11: Learning from feedback (`learning.py`)

### Stage 6: probabilistic selection

- **Intensity profiles** (`profiles.py`): risk ceiling, edit budget, restructuring share,
  temperature and per-type multipliers for light / moderate / heavy. Light disables
  structural edits. User settings layer on top: `vocabulary_variation`, `restructuring`
  (low/medium/high), `type_weights`, `randomness` (0–1), `max_edits`, `max_edit_ratio`.
- **Probabilities** (`probability.py`): `softmax`, `linear` or `uniform`; drawn only from
  candidates that passed every hard gate. Each applied edit records its probability.
- **Seeds**: same seed + config + input = same output. Sub-seeds per chunk via SHA-256.
- **Conflicts** (`conflicts.py`): overlap, adjacent same-type edits, two restructurings in
  one sentence, and replacements that would echo the same new word nearby.
- **Weights file**: `--weights weights.json` with any config fields, e.g.
  `{"weight_meaning_risk": 2.0, "type_weights": {"voice": 0.5}}`. Unknown keys are errors.

### Stage 7: local LLM integration

- **Structured request** (`llm_request.py`): each chunk of the ORIGINAL text is sent as JSON with
  intensity goal, tone, protected facts (your terms + names + numbers), rule suggestions with
  confidence, a compact analysis, constraints, `max_output_chars` and the output schema. The model
  may apply or reject each suggestion.
- **Validated reply** (`llm_response.py`): accepts `rewritten_text`, `applied`, `rejected`,
  `uncertain`. Tolerates markdown fences, `<think>` blocks and chatter. Rejects bad JSON, empty
  text, over-long output and missing protected strings, then re-asks (`format_retries`). If the
  model still fails, that chunk uses the rule-engine result. Model explanations are stored as
  claims only; Stage 8 will verify them.
- **Memory**: fixed `num_ctx` (= `context_window`), prompt trimmed to fit context minus output
  and system prompt, output tokens sized to the chunk, `keep_alive`, `--unload` frees the model,
  optional `num_threads`. Out-of-memory errors are not retried and stop further calls; a missing
  model prints the `ollama pull` command.
- `--llm-mode plain` keeps the old text-in/text-out behaviour. `--tone formal` sets tone.
  `--explain` also shows suggestions the model rejected and what it was unsure about.

### Stage 8: semantic validation

Every model rewrite is compared with the original chunk before it is used (`humanizer/validation/`):

| Check | Catches |
|---|---|
| `facts` | changed/missing/added numbers, names, dates; negation count; lost uncertainty or added certainty; if/unless changes; added or removed cause-and-effect; reversed comparisons |
| `events` | who-did-what-to-whom, voice-aware ("Rocks were thrown by John" = "John threw rocks"); reversed roles; dropped agents; an actor turned into a receiver |
| `additions_omissions` | unsupported modifiers/intensifiers ("landmark", "strong"); key words with no counterpart (WordNet relatives accepted; removable filler ignored) |
| `similarity` | advisory only: spaCy vectors with `use_embeddings`, otherwise lemma overlap |
| `grammar` | problems the rewrite introduced: duplicated words, fragments, agreement, tense shift, repeated phrases, new ambiguous pronouns |

Decision (`decision.py`): any critical issue, or too many major ones (`--strictness` low 3 / medium 2 /
high 1), rejects the rewrite. The model gets a targeted repair request listing the problems, the
result is re-checked, at most `max_retry_count` times. Then the rule-engine version is used if it
passes, otherwise the original text. `--explain` shows each chunk's decision and problems;
`--no-validate` turns checks off.

Known limits: checks are heuristic. Valid paraphrases can be flagged (handled conservatively), and
subtle meaning changes with no lexical or structural trace can be missed.

### Stage 10: local review interface

```bash
python -m humanizer.web            # opens http://127.0.0.1:8765  (or: rewriter-ui)
```

Standard library only; listens on this computer only unless you pass `--host`.

- Original and rewrite side by side with word-level highlights; switch between the rule version
  and the model version.
- Every rule edit with its reason, confidence, risk and selection probability. Keep, reject or
  clear each one; **Undo/Redo** work on these decisions. Edits the engine skipped are shown with
  the reason.
- Meaning checks for the rule version (re-run after every toggle) and for each model part,
  including which version was finally used, suggestions the model declined and what it was unsure about.
- Final text box (edit freely), a 1–5 naturalness rating, comment, **Accept & save** / **Reject**.
  The final text is checked too, and you are warned if it may change the meaning.
- Feedback goes to `data/feedback.jsonl` (`--feedback PATH`), with per-edit decisions, settings,
  validation and a stable train/dev/eval split. `GET /api/feedback/summary` gives acceptance
  rates per edit type, which is the input for Stage 11.
- One rewrite runs at a time, so only one model is ever loaded.

CLI: `--randomness 0.3 --max-edits 10 --selection-method linear --weights w.json --seed 42`.


### Stage 9: end-to-end evaluation

`rewriter-eval` (or `python -m humanizer.evaluation`) compares **rules**, **llm** and
**hybrid** modes on `data/test_paragraphs.jsonl`, then tunes the validator.

- Meaning: validator score/accept rate, critical issues, numbers and negations kept.
- Naturalness (proxy statistics, not a human judgement): filler phrases, repeated
  words, sentence-length variation, padding penalty; plus how much text changed.
- Tuning: `data/eval_pairs.jsonl` holds good and deliberately broken rewrites; each
  `semantic_strictness` level is scored (precision/recall/F1) and the best is recommended.
  Add your own pairs (or Stage 10 feedback) to make this evidence stronger.
- LLM modes are skipped with a note when Ollama is not running.

```
rewriter-eval --modes rules llm hybrid --out eval_report.json
```

Current evidence (rules mode, 15 paragraphs): 100% accepted, numbers/negations all kept.
Tuning: `medium` (the default) scores best, F1 1.00; reversed comparisons
("A is faster than B" -> "B is faster than A") are reliably caught and rejected.

### Stage 11: learning from feedback

`rewriter-learn` reads the accept/reject decisions saved by `rewriter-ui` and turns
them into per-change-type weights (`type_weights`) for the selection step.

- Uses only the **train** split; acceptance rates are smoothed so a few reviews can't
  swing a weight. 50% accepted -> 1.0; down to 0.2 (mostly rejected), up to 1.6.
- Types with fewer than `--min-examples` (default 5) reviews keep their weight.
- Safety check: new weights are tested on the **dev** split and only written if they
  predict your decisions at least as well as the current ones.
- Rules you mostly reject (e.g. a bad synonym rule) are listed for fixing, not hidden.
- Fine-tuning is not done; the tool reports whether there's enough data to consider it
  (>= 1000 records, >= 200 eval, >= 500 accepted rewrites).

```
rewriter-learn --feedback data/feedback.jsonl --out learned_weights.json
rewriter --weights learned_weights.json input.txt
```

### Writing modes and Markdown

- `--style neutral|casual|professional|academic` (also a "Style" menu in `rewriter-ui`).
  Academic and professional modes avoid slang, expand contractions ("don't" -> "do not")
  and slightly prefer measured formal words ("demonstrate", "examine"). They never pick
  ornate ones ("utilize", "aforementioned"). Casual mode allows relaxed words.
- Synonyms that are much rarer than the original word for the same meaning are skipped
  (e.g. "team" -> "squad"), and "plan" -> "program" is blocked.
- `.md` files: only prose paragraphs are rewritten. Front matter, headings, code blocks,
  lists, tables, quotes and HTML are kept exactly; inline code, links and URLs are protected.

```
rewriter -i essay.md -o essay_out.md --model <your tag from `ollama list`> --style academic --intensity moderate
```
