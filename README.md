# Recipe Enhancement Platform

Automatically enhances recipes by analyzing and applying community-tested modifications from AllRecipes.com. Uses LLM processing to extract meaningful recipe tweaks and apply them with full citation tracking.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Prerequisites

- Python 3.13+
- `uv` package manager

## Setup

```bash
# Install dependencies
uv venv
source .venv/bin/activate
uv pip sync pyproject.toml
```

Or simply `uv sync`, which also installs the test dependencies (`uv run pytest` does this automatically).

On Windows, clone to a short path (or enable long paths): some files in the `openai` package are nested deeply enough that a long clone path pushes them past the 260-character limit, and imports then fail.

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

## Usage

### 1. Scrape Recipes (Optional - data already provided)

```bash
uv run python src/scraper_v2.py
```

### 2. Run Recipe Enhancement Pipeline

```bash
cd src

# Test single recipe (chocolate chip cookies)
uv run python test_pipeline.py single

# Process all recipes
uv run python test_pipeline.py all

# Without an API key: replay recorded responses instead of calling a model
uv run python test_pipeline.py single --replay ../baseline/recordings/recipe_10813.json
uv run python test_pipeline.py all --replay ../baseline/recordings/*.json
```

Options: `--recipe FILE`, `--data-dir DIR`, `--out DIR`, `--max-reviews N` (analyse only the N best-ranked reviews). The script exits with 2 if no model is configured, and 1 if any recipe failed.

The recordings in `baseline/recordings/` are **hand-written ideal responses, not real model output**: they prove the workflow end to end but say nothing about a real model's extraction quality. Cookies, cake and nikujaga have every review recorded; the soup has two of six, so its other four reviews show up as failed extractions. Every report states where its extraction answers came from (`extraction_source`). `replay.RecordingClient` wraps a live client to save its answers for exact replay later.

**Model access** is injectable (`LLMAnalysisPipeline(llm_client=...)`). Otherwise an OpenAI client is built with a 30 s timeout and 1 SDK retry per request; with the single corrective retry that bounds each review to 2 requests. A missing API key fails before any call.

## Output

### Enhanced Recipes

Each recipe run ends in an explicit status: `enhanced`, `unchanged` (nothing applicable or nothing could be applied; the report is still written, without the "Community Enhanced" claim), `skipped` (no reviews) or `failed` (`invalid_input`, `extraction_failed`, or `error`, with the detail).

Reports are saved in `src/data/enhanced/` (or `--out`):

- `enhanced_[recipe_id]_[recipe-name].json` - The enhanced recipe with every extraction result, tweak outcome and a verified line-level diff
- `enhanced_[recipe_id]_[recipe-name].md` - The same, readable: what changed (by line, with the tweak responsible), applied tweaks with their review and evidence, everything not applied and why, extraction per review
- `pipeline_summary_report.json` - Status and counts for every recipe file, including skipped and failed ones

The diff (`recipe_diff`) is computed from the original and final text, independently of the modifier, then reconciled with the applied edits: every changed line must be explained by an applied edit and every applied edit must be visible. `diff_verified` records the result and `diff_problems` lists anything that does not reconcile.

### Data Structure

Original scraped recipes in `data/` directory contain reviews with `has_modification: true` flags. Enhanced recipes include:

```json
{
  "recipe_id": "10813_enhanced",
  "title": "Best Chocolate Chip Cookies (Community Enhanced)",
  "ingredients": ["1 cup butter", "1 additional egg yolk", ...],
  "modifications_applied": [
    {
      "source_review": {
        "text": "I added an extra egg yolk for chewier texture",
        "rating": 5
      },
      "modification_type": "addition",
      "reasoning": "Improves texture and chewiness",
      "changes_made": [...]
    }
  ],
  "enhancement_summary": {
    "total_changes": 1,
    "change_types": ["addition"],
    "expected_impact": "Chewier texture and improved consistency"
  }
}
```

## How It Works

The LLM Analysis Pipeline processes recipes in 3 steps:

0. **Input loading** (`loader.py`): Validates and normalizes the recipe file, merges duplicate reviews and assigns stable IDs (see [Input contract](#input-contract))
1. **Tweak Extraction**: Sends every review to an OpenAI model (currently `gpt-3.5-turbo`, JSON mode) and keeps every tweak that passes validation (see [Extraction contract](#extraction-contract))
2. **Recipe Modification**: Applies every tried, positively reported tweak from all reviews, in review rank order, each all-or-nothing (see [Edit rules](#edit-rules))
3. **Enhanced Recipe Generation**: Creates enhanced version with full citation tracking back to source review

Each run produces one enhanced recipe per original recipe, with complete attribution showing exactly what changed and why.

## Input contract

`loader.py` turns each `data/recipe_*.json` file into a `Recipe` and a list of `Review`s. A file that breaks the contract (missing `recipe_id` or `title`, `ingredients`/`instructions` not a non-empty list of strings) raises `RecipeDataError` and the recipe is skipped.

**Normalization.** Recipe lines, titles and review text have HTML entities decoded (`&#39;` → `'`), Unicode NFC-normalized and whitespace collapsed; empty lines are dropped. Amounts are not rewritten. The scraped text is kept as `RecipeLine.raw_text`. Times stay ISO 8601 (`PT20M`); a zero duration is treated as missing.

**Stable IDs.**

| Record | ID | Stable because |
|---|---|---|
| Recipe line | `ing-<n>`, `step-<n>` | Position in the normalized list; `Recipe.content_hash` (also written to output as `source_recipe_hash`) identifies the text it refers to |
| Review | `rev-<10 hex>` | SHA-1 of the normalized, case-folded text: the same in `reviews` and `featured_tweaks`, after re-ordering, and across re-scrapes |
| Tweak / edit | `<review id>.t<n>` / `<tweak id>.e<n>` | Derived from the review ID |

**Deduplication.** `reviews` and `featured_tweaks` are merged on normalized text. A merged review is `featured`, keeps every source position in `sources` (e.g. `["reviews[0]", "featured_tweaks[0]"]`), and takes the first known rating and username and the highest vote count. Ratings outside 1–5 become missing.

**Ranking.** Reviews are ranked best first; each signal only breaks ties left by the one before:

1. `featured`: listed in `featured_tweaks` (shown with photos on AllRecipes)
2. `helpful_votes`: read from `helpful_votes` or `helpful_count` if the scrape has them; missing counts as 0. *The supplied data and current scraper have no vote data, so this level never separates reviews today.*
3. `has_modification`: the scraper's keyword flag (a weak signal: it misses real tweaks)
4. `rating`: missing counts as 0
5. File order: `reviews` first, then featured-only entries

**Order.** Every review is extracted, flagged or not, in rank order. A recipe with no reviews is skipped without an LLM call.

**Records.** `models.py` defines the records, each validated on construction: `Tweak` (one change a reviewer describes), `TweakEdit` (one text operation targeting a line ID), `ExtractionResult` (one review's extraction) and `EditOutcome` / `TweakOutcome` (applied, refused with a reason, rolled back, or skipped; a tweak is all-or-nothing).

## Extraction contract

For each review the model returns `{"tweaks": [...]}`: one entry per distinct tweak, with `modification_type`, a verbatim `quote`, `status` (`tried` / `suggested`), `reviewer_outcome` (`positive` / `mixed` / `negative` / `unclear`), `reasoning`, and `edits` that each name a `line_id` (`ing-3`, `step-1`).

A tweak is accepted only if (`tweak_extractor.py`, `support.py`):

- every field is present with an allowed value
- its quote appears in the review (ignoring case, spacing and curly quotes)
- each edit's `line_id` exists in the right section, its `find` text is on that line, and it is complete (`replace` has replacement text; `add_after` has a line; `remove` names the whole line)
- every amount an edit introduces is stated in the review, or follows from a review amount and the line's amount by one +, −, × or ÷ (e.g. "1/2 c less" on a "3 cups" line supports "2.5 cups"); invented amounts and unit conversions are rejected

New words in an ingredient line that appear in neither the review nor the recipe are recorded as `warnings` on the tweak, not rejected, because paraphrase is common.

If anything fails, the model gets **one** corrective retry listing the exact problems. Tweaks valid in the first answer are kept even if the correction drops them; tweaks still invalid are kept in `rejected` with their problems. Each review's `ExtractionResult` has a status:

| Status | Meaning |
|---|---|
| `ok` | At least one valid tweak (some proposals may also be rejected) |
| `empty` | The review describes no tweak: a valid answer |
| `failed` | No usable answer after the retry, or every proposed tweak was rejected |

All results are written to the enhanced recipe as `extraction_results`, so suggestions and rejected proposals stay on record even though they are not applied. Only tweaks with `status: tried` and `reviewer_outcome: positive` are applied.

## Edit rules

`recipe_modifier.py` applies tweaks predictably. Edits are written against the original recipe, and every edit is checked before anything changes:

| Check | Refused as |
|---|---|
| The named line exists | `invalid` |
| The line, as it was in the original recipe, contains the find text | `not_found` |
| ...exactly once (the line ID already settles which line) | `ambiguous` |
| `remove` names the whole line; a replacement does not empty a line | `invalid` |
| An earlier tweak of the **same** review has not changed or removed the expected text | `stale` |
| An earlier tweak of a **different** review has not changed or removed the expected text, changed the same ingredient line, or changed a step this edit would remove | `conflict` |
| The change has not already been made (line already present, same replacement already on the line, line already removed) | `duplicate` |
| The edit changes the text | `no_op` |

A different review may edit another part of the same instruction step, or add a line after one that changed.

**Order.** Tweaks are applied by review rank (featured, votes, keyword flag, rating, file order; see [Input contract](#input-contract)), then in the order the review lists them. The first tweak to change a target wins, so a conflict is always resolved in favour of the higher-ranked review.

**Atomic tweaks.** A tweak's edits are tried on a copy. If any is refused, the copy is discarded: the tweak is `refused`, its refused edits carry the reason and which tweak caused it, and its other edits are `rolled_back`. So an ingredient change never lands without the step edits that go with it.

Every extracted tweak gets a `TweakOutcome` in the output's `tweak_outcomes`: `applied`, `refused`, or `skipped` (with the policy reason: `not_tried`, `outcome_not_positive`, `no_edits`).

## Development

```bash
# Add dependencies
uv add <package_name>

# Run offline tests (no API key needed)
uv run pytest

# Run the live pipeline on one recipe (needs OPENAI_API_KEY)
cd src && uv run python test_pipeline.py single

# Baseline comparison (see baseline/README.md)
uv run python baseline/run_baseline.py --impl src --label current
```
