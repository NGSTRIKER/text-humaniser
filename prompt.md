# Master Project Prompt: Build a Hybrid Rule-Based and Local LLM Humanizer in Python

## 1. Project Overview

Build a modular, locally running Python application that rewrites text to improve its naturalness, fluency, readability, and stylistic variety while preserving its original core meaning.

The system must combine:

1. A rule-based linguistic analysis and transformation engine.
2. A probabilistic edit-selection algorithm.
3. A word relationship graph containing synonyms, word senses, and grammatical information.
4. A local instruction-tuned LLM for natural language rewriting.
5. A semantic and structural validation system that compares the final result with the original input.
6. A feedback and evaluation system that helps us improve the quality of the transformations over time.

The goal is to create a controllable rewriting system, not simply a wrapper around an LLM prompt.

The rule engine must guide the LLM, propose candidate transformations, enforce constraints, and evaluate the result. The LLM must handle contextual language generation that deterministic rules cannot reliably perform alone.

The system must run locally, protect user data, and avoid requiring cloud inference.

## 2. Hardware Constraints

The development machine has:

- 8 GB system RAM.
- Intel integrated graphics.
- No confirmed dedicated GPU.
- Python available for development.

Design for CPU inference and shared-memory constraints.

Requirements:

- Use one local LLM at a time.
- Prefer a quantized 2B–4B instruction-tuned model as the initial target.
- Support Ollama as the first inference backend.
- Keep the model backend replaceable so llama.cpp or another local runtime can be integrated later.
- Use configurable context lengths and generation limits.
- Avoid loading multiple large models simultaneously.
- Release temporary resources when practical.
- Avoid unnecessary embedding models or large NLP pipelines.
- Ensure the rule engine can operate independently when the LLM is unavailable.
- Gracefully handle out-of-memory errors, timeouts, and model loading failures.
- Make hardware-intensive features optional.

Do not assume that Intel integrated graphics will accelerate inference. Start with CPU-compatible operation.

## 3. Core Design Philosophy

Follow this pipeline:

Original Text → Linguistic Analysis → Rule Engine → Candidate Edit Generation → Probabilistic Selection → Local LLM Rewrite → Semantic and Structural Validation → Decision Controller → Final Text.

The stages must have distinct responsibilities.

### Rule engine

Detect problems, analyze grammatical structure, generate possible edits, score their usefulness, and protect meaning.

### Local LLM

Generate fluent rewrites using the linguistic analysis, candidate edits, transformation policy, and user settings.

### Validator

Compare the original and rewritten versions, detect semantic changes and structural errors, and determine whether the rewrite is safe to accept.

### Decision controller

Accept, repair, retry, or reject a rewrite according to configurable validation thresholds.

Do not apply every possible transformation to every sentence. Only make edits that are justified by context and the selected rewrite intensity.

## 4. Linguistic Analysis

Use spaCy with a suitable English pipeline to analyze the input.

Extract and represent:

- Tokens and character offsets.
- Lemmas and surface forms.
- Part-of-speech tags.
- Morphological features.
- Dependency relationships.
- Dependency heads and children.
- Sentence boundaries and sentence indices.
- Noun phrases and verb phrases.
- Clauses and clause relationships where detectable.
- Subject, object, and indirect object relationships.
- Named entities.
- Numbers, dates, quantities, and units.
- Negation.
- Tense, aspect, mood, and modality where detectable.
- Conjunctions and discourse markers.
- Pronouns and possible antecedents.
- Repeated words and phrases.
- Sentence openings and syntactic patterns.
- Potentially redundant modifiers.
- Candidate filler phrases and wordy constructions.

Create typed Python data structures for the analysis results.

Preserve original character offsets so edits can be traced to their source spans.

Treat parser output as evidence rather than absolute truth. If an analysis is ambiguous or unreliable, avoid aggressive transformations.

## 5. Rule-Based Analysis and Transformation Engine

Implement rules as independent, testable modules.

Each rule should have:

- A unique identifier.
- A description.
- Conditions for activation.
- The text span it applies to.
- A proposed transformation.
- A confidence score.
- An estimated meaning-change risk.
- A grammatical compatibility check.
- A reason explaining the proposed edit.
- A severity or priority.
- A way to reject or reverse the transformation.

The initial rule categories should include the following.

### A. Redundancy removal

Detect expressions that repeat the same meaning without adding useful emphasis.

Examples:

- "completely eliminate entirely" → "eliminate"
- "the reason why ... is because" → a cleaner equivalent construction when grammatically appropriate.
- Repeated explanations that do not contribute new information.

Never delete repetition that is intentional, meaningful, rhetorical, or necessary for clarity.

### B. Wordiness and filler

Detect unnecessarily long constructions and replace them with simpler alternatives when the meaning is preserved.

Examples:

- "due to the fact that" → "because"
- "in order to" → "to", when the shorter form retains the intended meaning.
- "at this point in time" → "now", when context permits.

Do not simplify technical, legal, academic, or otherwise specialized wording without sufficient confidence.

### C. Repeated vocabulary

Analyze repetition within sentences and paragraphs.

Identify overused content words, repeated phrases, and repetitive sentence openings.

Distinguish accidental repetition from deliberate emphasis and necessary terminology.

Do not replace technical terms, names, or core concepts simply to increase vocabulary variety.

### D. Contextual synonym selection

Generate candidate synonyms using a lexical resource such as WordNet.

Optionally use a lightweight embedding model to rank candidates by contextual similarity.

Filter candidates according to:

- Intended word sense.
- Part of speech.
- Grammatical role.
- Morphology and agreement.
- Register and formality.
- Contextual meaning.
- Semantic similarity.
- Domain-specific terminology.
- Risk of changing the claim.

Never perform blind synonym replacement.

If no alternative is sufficiently reliable, retain the original word.

### E. Modifier management

Identify unnecessary adjectives, adverbs, intensifiers, and redundant descriptive phrases.

Remove modifiers only when they add no meaningful information.

Allow useful modifiers to be added only when they clarify an implication already supported by the original text. Do not invent facts or unsupported details.

### F. Phrase restructuring

Allow grammatical alternatives that preserve meaning, including changes in phrase order, clause structure, and modifier placement.

Check scope, emphasis, attachment, and ambiguity before accepting a transformation.

### G. Sentence variety

Detect monotonous sentence structures, repeated openings, and excessive uniformity.

Offer alternative constructions while preserving the original ideas and relationships.

Do not force every sentence to have a different structure.

## 6. Active and Passive Voice Transformation

Voice conversion is a core feature, not an optional afterthought.

The system must identify active and passive constructions and generate alternative forms when grammatically appropriate.

Example:

Active:
"John threw rocks into the lake."

Passive:
"Rocks were thrown into the lake by John."

The transformation should preserve the event and the relevant participants.

Analyze the underlying roles:

- Agent: John.
- Action: threw.
- Theme: rocks.
- Destination: the lake.

The system must account for:

- Subject and object relationships.
- Tense.
- Aspect.
- Verb agreement.
- Modality.
- Negation.
- Auxiliary verbs.
- Pronoun case.
- Prepositions.
- Agent preservation or intentional omission.
- Information focus and discourse context.

Examples of distinctions that must be preserved:

- "John throws the ball." must not become a past-tense statement.
- "John did not throw the ball." must remain negative.
- "John might throw the ball." must retain uncertainty.
- "John has thrown the ball." must preserve aspect as closely as possible.

Voice transformations should be selective.

Do not force passive voice into conversational text when it sounds awkward. Do not force active voice when the agent is unknown, irrelevant, or intentionally omitted.

Record every voice change in the edit history.

## 7. Word Order and Sentence Restructuring

Support more than individual word substitutions.

The transformation engine should generate and evaluate alternatives involving:

- Word and phrase rearrangement.
- Modifier repositioning.
- Clause rearrangement.
- Active-to-passive and passive-to-active conversion.
- Sentence combining.
- Sentence splitting.
- Alternative grammatical constructions.
- Changes in sentence openings.
- Reordering of sentences within a paragraph when coherence is preserved.

Example:

Original:
"She carefully opened the door."

Alternative:
"She opened the door carefully."

These sentences preserve the basic event while varying phrase order.

The system must check whether a rearrangement changes emphasis, scope, ambiguity, or implied relationships.

Sentence combining requires particular care.

For example, two adjacent sentences must not automatically be connected with "because", "therefore", or "although". Such connectors introduce logical relationships that must be supported by the original text or context.

Never rearrange words merely to produce novelty. Every transformation must remain grammatical, contextually appropriate, and semantically defensible.

## 8. Probabilistic Edit-Selection Algorithm

Create a configurable scoring system to decide which edits should be attempted.

A candidate edit's score should consider:

- Redundancy or repetition reduction.
- Contextual suitability.
- Grammatical compatibility.
- Expected readability improvement.
- Stylistic diversity.
- Semantic preservation confidence.
- Meaning-change risk.
- Grammatical risk.
- Domain terminology risk.
- Transformation intensity.
- Conflicts with previously selected edits.

Start with a transparent weighted scoring formula. The weights must be configurable rather than permanently hardcoded.

An illustrative scoring model is:

score = usefulness + context_fit + grammar_fit + diversity_gain - meaning_risk - grammar_risk.

Convert candidate scores into selection probabilities using a configurable approach, such as a softmax distribution or normalized nonnegative weights.

Use weighted random selection only among acceptable candidates.

Important requirements:

- High-risk candidates must be rejected or require special handling, not rescued by randomness.
- Names, numbers, dates, negation, technical terms, and protected phrases must receive appropriate safeguards.
- Overlapping edits must be resolved before application.
- Apply edits in a controlled order so that character offsets do not become invalid.
- Provide a fixed random seed for reproducibility.
- Support deterministic mode for testing.
- Limit the number or proportion of edits.
- Allow the same input to be processed reproducibly when the same seed and configuration are used.

Randomness must select among linguistically valid alternatives. It must never override grammatical or semantic constraints.

## 9. Rewrite Intensity and User Controls

Expose configurable controls for:

- Rewrite intensity: light, moderate, heavy.
- Vocabulary variation: conservative to flexible.
- Sentence restructuring: low to high.
- Active/passive voice conversion: disabled, selective, or more frequent.
- Paragraph sentence reordering: disabled by default or enabled.
- Redundancy removal strength.
- Modifier changes.
- Randomness.
- Maximum number or proportion of edits.
- Formality and tone.
- Sentence length variation.
- Protected terms and phrases.
- Semantic preservation strictness.
- Model selection.
- Generation temperature and token limits.
- Retry limits.
- Validation thresholds.
- Random seed.

### Intensity behavior

Light:
- Fix obvious redundancy, grammar issues, and wordiness.
- Preserve most wording and structure.

Moderate:
- Improve readability and flow.
- Permit contextual vocabulary changes, selective voice conversion, and sentence restructuring.

Heavy:
- Allow broader restructuring, sentence combining or splitting, and more stylistic variation.
- Still preserve the original claims, important details, uncertainty, negation, and intended relationships.

Do not define heavy rewriting as permission to invent content or alter meaning.

## 10. Word Relationship Graph

Implement a lightweight word relationship graph.

Represent words or word senses as nodes and relationships as edges.

Possible relationships include:

- Synonym.
- Antonym.
- Related concept.
- Similar meaning.
- Broader or narrower concept.
- Morphological relationship.
- Grammatical compatibility.
- Register or formality.
- Contextual usage restrictions.

Use WordNet as the initial lexical source.

Optionally add embedding-based similarity and curated rules.

Do not treat every relationship as interchangeable. Antonyms, related concepts, and near-synonyms must have different relationship types.

Where possible, represent word senses separately so that one spelling can have multiple meanings.

The graph should provide candidate alternatives, not make final substitution decisions independently.

## 11. Local LLM Integration

Implement a replaceable model interface.

Start with Ollama. Keep the rest of the application independent of Ollama-specific implementation details.

The LLM should receive a structured request containing:

- Original text.
- Linguistic analysis.
- Proposed rule-based edits.
- Edit confidence scores.
- Protected facts and phrases.
- Required transformations.
- Rewrite intensity.
- Tone and style preferences.
- Semantic preservation constraints.
- Maximum output length or expansion limits.
- Instructions against invented information.
- Desired output format.

The model should generate a coherent rewrite rather than blindly applying every proposed edit.

Allow it to reject a rule suggestion if the suggestion is inappropriate in context.

Request structured output when practical, including:

- Rewritten text.
- Applied transformations.
- Rejected suggestions and reasons, where feasible.
- Uncertainty flags.

Validate model output against the requested schema.

Treat model explanations as claims to verify, not proof that the output is correct.

Do not assume that a model's self-reported confidence is calibrated.

Use a low-to-moderate generation temperature initially and make it configurable.

Avoid excessively long prompts or contexts that consume unnecessary memory.

## 12. Semantic and Structural Validation

This is one of the highest-priority modules.

Compare the original input against the final rewritten text.

The goal is to determine whether the core meaning has been preserved despite changes in vocabulary, grammatical voice, word order, sentence structure, and paragraph organization.

Do not rely solely on embedding similarity or a single LLM judgment.

Implement multiple complementary checks.

### A. Fact preservation

Track and compare:

- Named entities.
- Numbers and quantities.
- Dates and times.
- Units and measurements.
- Negation.
- Comparisons.
- Conditions.
- Causal claims.
- Modal language and uncertainty.
- Technical terms.
- Important relationships between entities.

Detect unexpected additions and omissions.

### B. Event and semantic-role preservation

Where possible, compare the underlying events and their participants.

For example:

Original:
"John threw rocks into the lake."

Valid candidate:
"Rocks were thrown into the lake by John."

Invalid candidate:
"John was hit by rocks near the lake."

The second candidate changes the event and must be rejected even though several words are shared.

Preserve who did what to whom, where, when, and under what relevant conditions.

Account for changes in grammatical subject caused by passive voice.

### C. Semantic similarity

Optionally use a lightweight sentence-embedding model to measure similarity.

Use this as one signal rather than an absolute decision rule.

A high similarity score does not prove equivalence, and a lower score does not necessarily imply a meaning change when the sentence has been substantially restructured.

### D. Contradiction and logical consistency

Detect likely changes in:

- Polarity.
- Certainty.
- Quantity.
- Temporal relationships.
- Conditions.
- Causal relationships.
- Comparisons.
- Agent and recipient roles.

Flag ambiguous cases for conservative handling.

### E. Grammatical and stylistic validation

Run the rewritten text through the linguistic analyzer again.

Check for:

- Broken syntax.
- Agreement errors.
- Incorrect tense.
- Awkward or duplicated wording.
- Repeated phrases introduced by the rewrite.
- Pronoun ambiguity.
- Fragmented or run-on sentences.
- Inconsistent terminology.
- Paragraph incoherence.

### F. Unsupported additions and omissions

Compare the information conveyed by both texts.

Flag details that appear only in the rewrite, including unsupported adjectives, intensifiers, examples, explanations, quantities, and causal claims.

Also detect meaningful information lost during simplification.

### G. Decision policy

The validator should return a structured report with:

- Overall result.
- Individual check results.
- Suspected meaning changes.
- Missing information.
- Unsupported additions.
- Grammar problems.
- Confidence and uncertainty indicators.
- Suggested repair actions.

If a critical check fails, do not accept the rewrite automatically.

Depending on the problem:

1. Reject the candidate.
2. Ask the LLM for a targeted repair.
3. Revalidate the repaired version.
4. Retry only a limited number of times.
5. Fall back to the original text or the last validated version.

Avoid endless rewriting loops.

## 13. Paragraph-Level Analysis

Analyze the paragraph as a connected unit, not merely a list of isolated sentences.

Track:

- Topic continuity.
- Repeated ideas.
- Repeated sentence openings.
- Discourse markers.
- Pronoun antecedents.
- Chronology.
- Causal and conditional relationships.
- Contrast and concession.
- References to earlier or later sentences.
- Introduction and conclusion structure.
- Logical dependencies between claims.

Sentence reordering must be disabled by default.

When enabled, generate candidate reorderings and check whether each preserves chronology, discourse relations, and reference clarity.

Do not randomly shuffle sentences.

The system must distinguish a sentence that is grammatically independent from one that is logically independent.

A sentence can be grammatically complete while still depending on a previous sentence for its meaning.

## 14. Postprocessing and Conditional Second Pass

After the LLM generates a rewrite, run the rule-based validation and analysis pipeline again.

Do not blindly apply all rules a second time.

The postprocessor should:

- Reanalyze the rewritten text.
- Identify remaining clear issues.
- Detect problems introduced by the LLM.
- Check whether protected information was preserved.
- Verify the transformations actually applied.
- Reject or repair unsafe edits.
- Prevent repeated transformations from undoing each other.

Only invoke a second LLM generation when a specific validation failure or clearly identified quality problem warrants it.

Prefer targeted repair instructions over requesting another unrestricted rewrite.

Keep the original text and all intermediate versions so the system can recover from quality regressions.

## 15. Edit Tracking and Explainability

Record every proposed and accepted edit.

Each edit record should include:

- Original text span.
- Replacement text.
- Transformation type.
- Rule identifier or LLM operation.
- Reason for the edit.
- Confidence estimate.
- Estimated semantic risk.
- Validation outcome.
- Whether the user accepted or rejected it.

Support an edit history that explains changes such as:

- Removed redundant wording.
- Reduced repeated vocabulary.
- Changed active voice to passive voice.
- Rearranged a phrase.
- Split or combined sentences.
- Restructured a clause.
- Rejected a synonym because of contextual mismatch.

The user should eventually be able to inspect, accept, reject, or undo individual changes.

Do not claim an edit is meaning-preserving merely because it was applied successfully.

## 16. User Feedback and Learning

Create a feedback system that records:

- Original text.
- Proposed rewrite.
- Final accepted rewrite, if any.
- User acceptance or rejection.
- Accepted or rejected individual edits.
- Transformation types.
- User-selected intensity and style.
- Validation results.

Use feedback initially to improve rule weights, candidate rankings, and transformation preferences.

Store feedback locally in a structured format, such as JSONL or SQLite.

Do not automatically treat every generated output as a correct training example.

If sufficient high-quality data becomes available, consider optional LoRA or QLoRA fine-tuning later. Fine-tuning is not part of the initial minimum viable product.

Keep training, development, and evaluation examples separate to reduce data leakage.

## 17. Project Structure

Use a modular architecture similar to:

humanizer/
    main.py
    config.py
    schemas.py
    analyzer.py
    word_graph.py
    context_analyzer.py
    weighting.py
    candidates.py
    transformations/
        redundancy.py
        repetition.py
        wordiness.py
        synonyms.py
        modifiers.py
        voice.py
        word_order.py
        sentence_structure.py
        paragraph_flow.py
    rewriter.py
    llm_backend.py
    validator.py
    semantic_checks.py
    decision_controller.py
    edit_history.py
    feedback.py
    evaluation.py
    logging_config.py

tests/
    test_analyzer.py
    test_rules.py
    test_voice.py
    test_rearrangement.py
    test_word_graph.py
    test_weighting.py
    test_validator.py
    test_paragraph_flow.py
    test_end_to_end.py

data/
    examples.jsonl
    feedback.jsonl

README.md
requirements.txt

Adjust the structure when useful, but keep components modular, typed, documented, and independently testable.

Avoid unnecessary abstractions and dependencies.

## 18. Testing and Evaluation

Create a curated evaluation dataset containing:

- Short and long paragraphs.
- Formal and conversational writing.
- Academic and technical passages.
- Repetitive text.
- Wordy text.
- Active and passive voice examples.
- Sentence rearrangement examples.
- Ambiguous pronouns.
- Names, numbers, dates, and measurements.
- Negation and uncertainty.
- Causal and conditional statements.
- Comparisons and exceptions.
- Multiple sentences with discourse dependencies.
- Cases where no rewrite is needed.
- Cases where a tempting transformation would change meaning.

For every transformation, test both positive and negative cases.

Evaluate:

1. Core meaning preservation.
2. Fact preservation.
3. Grammar.
4. Naturalness and readability.
5. Unnecessary changes.
6. Unsupported additions.
7. Important omissions.
8. Transformation diversity.
9. Reproducibility in deterministic mode.
10. Runtime and memory use.

Compare three configurations:

- Rule-based only.
- LLM-only rewriting.
- Hybrid rules → LLM → validation.

Use human preference ratings for naturalness when possible.

Do not optimize for superficial measures such as the number of changed words.

## 19. Implementation Stages

Implement the application incrementally.

### Stage 1: Foundation

- Create project structure and configuration.
- Implement text input/output.
- Add logging and error handling.
- Create a baseline local LLM interface.
- Ensure the program runs without a GPU.

### Stage 2: Linguistic analyzer

- Integrate spaCy.
- Extract sentence, token, dependency, entity, and morphology information.
- Add tests.

### Stage 3: Basic rule engine

- Implement redundancy, wordiness, repetition, and safe phrase transformations.
- Add candidate scoring and edit tracking.

### Stage 4: Word relationship graph

- Integrate WordNet.
- Add contextual candidate filtering.
- Add optional embedding support only if practical.

### Stage 5: Structural transformations

- Implement active/passive conversion.
- Add phrase rearrangement.
- Add sentence combining and splitting.
- Add conservative paragraph-flow analysis.
- Test tense, agreement, negation, modality, and participant preservation.

### Stage 6: Probabilistic selection

- Implement configurable weights.
- Add intensity profiles and deterministic seeds.
- Resolve edit conflicts.

### Stage 7: Local LLM integration

- Integrate Ollama.
- Pass structured linguistic analysis and edit suggestions.
- Enforce output format and memory constraints.

### Stage 8: Semantic validation

- Implement fact checks and event-role comparisons.
- Add optional semantic similarity.
- Implement accept, repair, retry, and fallback behavior.

### Stage 9: End-to-end evaluation

- Compare rule-only, LLM-only, and hybrid results.
- Measure meaning preservation and naturalness.
- Tune thresholds based on evidence.

### Stage 10: User interface and feedback

- Add a lightweight local interface if appropriate.
- Show original and rewritten text side by side.
- Display individual edits and validation warnings.
- Add accept, reject, undo, and feedback functions.

### Stage 11: Optional advanced learning

- Improve weights from feedback.
- Consider fine-tuning only after a strong evaluation dataset exists.

## 20. Code Quality Requirements

- Use modern Python and type hints.
- Document public functions and non-obvious algorithms.
- Validate configuration values.
- Handle errors explicitly.
- Keep transformations testable and reversible where possible.
- Avoid modifying the original input.
- Use deterministic tests.
- Pin or document dependency versions.
- Verify actual library APIs rather than inventing functions.
- Keep the code compatible with the stated hardware.
- Include installation and troubleshooting instructions.
- Explain known limitations.
- Never claim that a feature is implemented unless working code and tests demonstrate it.

Do not produce the entire project as one giant script.

Implement one stage at a time. At each stage, provide complete files, explain their responsibilities, run or describe appropriate tests, and ensure the project remains usable before proceeding.

## 21. Final Objective

Deliver a practical local humanizer whose architecture is:

- Rule-driven for control.
- Linguistically informed for grammatical accuracy.
- Probabilistic for selective variation.
- LLM-assisted for natural language generation.
- Semantically validated against the original.
- Conservative when uncertain.
- Explainable through edit history.
- Lightweight enough to run on an 8 GB RAM Intel integrated-graphics computer.

The most important success criterion is not how different the final wording looks. It is whether the result reads naturally while preserving the original message, facts, logical relationships, and intended meaning.

Start with Stage 1 and proceed incrementally. Ask for essential environment details only when needed. Do not skip ahead to fine-tuning or a complicated user interface before the core rewriting and validation pipeline works.
