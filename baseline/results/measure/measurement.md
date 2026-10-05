# Measurement: original vs improved

| Mode | Meaning | Results |
|---|---|---|
| **Mocked** | Stand-in answers written by hand (ideal labelled answers, or deliberately defective ones) | All tables below unless marked |
| **Cached** | Real model output saved earlier (the 4 committed outputs, original-era code) | Cached section |
| **Live** | New model calls | **None**: no API key was available |

| Version | Code | Model (configured) | Prompt version |
|---|---|---|---|
| Original | `baseline/reference/original` | gpt-3.5-turbo | `8ed7136348b7` (hash of prompts.py) |
| Improved | `src` | gpt-3.5-turbo | `37cc3c30e9f6` (PROMPT_VERSION) |

> **Post-measurement fixes (20:39).** Two changes were made after the first measurement, prompted by it: lines added after the same line now keep their order (found by the held-out set: improved held-out 'recipe exactly as expected' was **0/1 before the fix**, 1/1 after), and the validator now checks units and plan wording in 'tried' quotes (defects rejected **7/10 before**, 9/10 after). The improved figures below are post-fix, so the held-out set is **no longer blind** for these fixes.

## Development fixtures F1–F5 (used while building)

Mocked: the stand-in returns the labelled ideal extraction in each version's format, so extraction precision is 100% by construction and is not reported here (see Cached and Validator). The original picks a review at random, so its figures are the expected value over every possible choice.

| Area | Metric | Original | Improved |
|---|---|---|---|
| Extraction | reviews analysed | 4/7 (57%) | 7/7 (100%) |
| Extraction | recall: labelled tweaks captured | 9/18 (50%) | 18/18 (100%) |
| Extraction | recall: captured as their own record | 0.5/18 (3%) | 18/18 (100%) |
| Extraction | untested or disliked tweaks applied | 1.5 | 0.0 |
| Edits | expected line changes produced | 19/26 (73%) | 29/29 (100%) |
| Edits | unexpected line changes | 1.0 | 0.0 |
| Edits | recipe exactly as the policy expects | 1.0/4 | 4.0/4 |
| Edits | single-edit cases (F4) | 3/8 | 8/8 |
| Attribution | tweaks accounted for (applied, refused or skipped) | 9/18 (50%) | 18/18 (100%) |
| Attribution | records naming their review / quoting evidence | 0/4 (0%) / 0/4 (0%) | 18/18 (100%) / 18/18 (100%) |
| Attribution | tweaks with the wrong type label | 3.5 | 0.0 |
| Attribution | title misstates whether anything changed | 1.0 | 0.0 |

## Held-out H1 (never used while building)

Mocked: the stand-in returns the labelled ideal extraction in each version's format, so extraction precision is 100% by construction and is not reported here (see Cached and Validator). The original picks a review at random, so its figures are the expected value over every possible choice.

| Area | Metric | Original | Improved |
|---|---|---|---|
| Extraction | reviews analysed | 1/4 (25%) | 4/4 (100%) |
| Extraction | recall: labelled tweaks captured | 1.7/8 (21%) | 8/8 (100%) |
| Extraction | recall: captured as their own record | 0/8 (0%) | 8/8 (100%) |
| Extraction | untested or disliked tweaks applied | 0.3 | 0.0 |
| Edits | expected line changes produced | 3/6.3 (47%) | 13/13 (100%) |
| Edits | unexpected line changes | 0.0 | 0.0 |
| Edits | recipe exactly as the policy expects | 0.0/1 | 1.0/1 |
| Attribution | tweaks accounted for (applied, refused or skipped) | 1.7/8 (21%) | 8/8 (100%) |
| Attribution | records naming their review / quoting evidence | 0/1 (0%) / 0/1 (0%) | 8/8 (100%) / 8/8 (100%) |
| Attribution | tweaks with the wrong type label | 0.7 | 0.0 |
| Attribution | title misstates whether anything changed | 0.3 | 0.0 |

## Cached real-model output (original-era code)

Tweak recall 4/10; extracted items that are real tweaks 4/4; items faithful to the review 2/4 (one invented amounts, one applied an untried suggestion). Model and prompt were not recorded with these outputs. No cached output exists for the improved version.

## Validator precision (mocked defective answers)

Defective answers rejected: **9/10**. Valid labelled tweaks accepted: **26/26** (development and held-out). Missed defects:

- D9 invented descriptor (only warned: ing-2: words in neither review nor recipe: bell, diced)

## Latency and cost (mocked responses, all 6 recipes)

| | Original | Improved |
|---|---|---|
| LLM calls | 4 | 19 |
| Input tokens | 3,030 | 30,501 |
| Output tokens | 530 | 3,537 |
| Pipeline time excluding the model (sum of per-recipe medians) | 17.8 ms | 88.2 ms |
| Request timeout / SDK retries | 600 s / 2 | 30 s / 1 |
| Worst-case requests per review call | 9 | 4 |

Tokens counted with tiktoken (cl100k_base) plus 4 framing tokens per message; output tokens come from the hand-written responses, so they are indicative only. Model latency was not measured (no live calls).

