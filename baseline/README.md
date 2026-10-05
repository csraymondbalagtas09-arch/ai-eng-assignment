# Baseline

Labelled fixtures and a harness that measure a pipeline implementation against the same cases, so the original system and any change to it can be compared like for like.

```bash
# From the project root; no API key or network needed
uv run python baseline/run_baseline.py --impl baseline/reference/original --label original
```

Results are written to `baseline/results/<label>.json` (full detail) and `<label>.md` (expected vs actual tables). Two runs on the same implementation produce byte-identical files.

## Layout

| Path | What it is |
|---|---|
| `reference/original/llm_pipeline/` | Frozen, unmodified copy of the original package (the system under test for the baseline). Byte-identical to the downloaded assignment code. |
| `fixtures/F*.json` | Labelled cases. Recipe and review text is copied verbatim from `data/`; the harness re-checks this on every run. |
| `fixtures/observed_llm.json` | Labels for the real-LLM modifications in the committed output files (the only real-model evidence available). |
| `run_baseline.py` | Harness. |
| `results/` | Generated results. |

## Fixtures

| Fixture | Covers | Source |
|---|---|---|
| F1_multi_tweak | One review, four numbered tweaks of four types | cookies, reviews[1] |
| F2_untested_suggestions | "Next time I will…" / "I would prefer…", plus a tried change with no specifics | apple cake, reviews[0–1] |
| F3_conflicting_reviews | Two reviews swap half-and-half for different things | soup, reviews[3] and [5] |
| F4_ambiguous_targets | Edit-level: ambiguous, absent, paraphrased, and short-span targets | apple cake and cookies |
| F5_unflagged_reviews | Tweaks in a review the scraper's keyword flag missed; a disliked change; advice | nikujaga, reviews[0–1] |

Each tweak label records the verbatim quote, type, `status` (tried/suggested), `outcome` (positive/mixed/negative/unclear), `motive` (improvement/availability/unclear) and the ideal edits with explicit line numbers. `expected` gives the result under the agreed policy (apply only tried + positive tweaks; higher-rated review wins conflicts, then file order).

## How it scores

The LLM is replaced by an oracle that returns the labelled ideal extraction, translated into the implementation's own response format (for the original: tried tweaks of the review merged into one object, as its prompt asks). Random review selection is replaced by one run per possible choice. Each stage is judged only on the input it actually received:

- **Extraction completeness**: reviews analysed; labelled tweaks captured; captured tweaks with their own record; ineligible tweaks (suggested, or not reported positive) passed to the modifier.
- **Edit correctness**: the edits the modifier was given are applied by a trivial reference applier using the labelled line numbers; the implementation's recipe is compared line by line (correct, missing, unexpected). F4 checks single edits: applied correctly, refused, wrong edit, or phantom record.
- **Reporting accuracy**: the written output is checked against what actually happened to the text (no-op records, claimed-but-not-done, done-but-not-recorded), against the labels (tweaks accounted for, type labels), and for honesty (title claims enhancement only if text changed) and traceability (review pointer, evidence quote).
- **End to end**: the output recipe against the policy's expected recipe.

Before scoring, every fixture is self-checked: text matches its source file, quotes are verbatim, each labelled edit's find text occurs exactly once on its line, and applying `expected.applied` with the reference applier reproduces `expected.recipe`. Any failure aborts the run.

## Limits

- The oracle measures what each stage does with ideal input. It does not measure the real model's extraction quality; that needs recorded live responses (requires an API key).
- `observed_llm.json` is real model output, but `data/enhanced/` was produced by an earlier code version with an unknown prompt.
- A refusal counts as a pass where a case allows it ("or refuse"). The original refuses every short span inside a step, so its passes on F4.4 and F4.5 are incidental rather than ambiguity detection.
- Only an adapter for the original implementation exists so far.
