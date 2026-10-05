# Engineering decisions: recipe tweak pipeline

Date: 5 October 2026. Scope: `src/llm_pipeline/`, `src/test_pipeline.py`, `baseline/`, `tests/`.

Goal: improve the pipeline that turns AllRecipes reviews into an "enhanced" recipe, in this order of priority: **complete tweak extraction**, **correct edits**, **traceable explanations**.

**Evidence status.** No OpenAI API key was available. Every result below is **mocked** (hand-written stand-in model answers) unless marked **cached** (the four real model outputs committed with the assignment). There are **no live results**. The code around the model is measured; how well a real model extracts tweaks with the new prompt is not.

---

## 1. The problem

### What the original did

| Stage | Original behaviour | Consequence |
|---|---|---|
| Select | One `random.choice` among reviews flagged by a keyword regex | Unflagged reviews with real tweaks are never read; the same input gives different recipes |
| Extract | One `ModificationObject` per review, one type; `gpt-3.5-turbo`, JSON mode, 1000 tokens | Several tweaks merge into one record with one type; suggestions are indistinguishable from changes made |
| Apply | Most similar *whole line* (`SequenceMatcher` ≥ 0.6), then `str.replace` | Short phrases inside steps never match; near-misses edit the wrong line; failed replacements are recorded as changes |
| Output | `"(Community Enhanced)"` always; review text only | No review ID, no evidence quote, no record of what was not applied |

### Reproduced failures (baseline, mocked)

The baseline harness runs the frozen original (`baseline/reference/original/`, byte-identical to the download) on labelled fixtures, with a stand-in model that returns the ideal extraction, so every failure below is the pipeline's, not the model's.

| Failure | Evidence |
|---|---|
| At most 13 of 34 labelled tweaks (38%) reachable in any run, even with perfect extraction | One review per run; 3 tweaks sit in unflagged reviews |
| Removing "1 cup chopped pecans" deletes "1 cup chopped walnuts" | Similarity 0.83 ≥ 0.6 (F4.6) |
| Substitution changes the ingredient but not the steps | All 6 step edits in F3 fail (similarity 0.23–0.34); recipe says 2% milk in ingredients, half-and-half in steps |
| Same input, two different recipes | F3 under each possible random choice |
| Change recorded that never happened | "1/2 teaspoon salt" fuzzy-matches "0.5 teaspoon salt", `str.replace` does nothing, a record is still written (F4.7) |
| "Community Enhanced" with zero changes | F2 |
| Real model output (cached) invented detail | Committed output adds "1 cup roasted Brussels Sprouts, halved; 1 cup roasted bell peppers, diced" for "I added roasted Brussels Sprouts and peppers"; another applied the untried "next time I will use fresh ginger" |
| Committed sample outputs do not match the code | `data/enhanced/*.json` contain `confidence_score`, absent from the models; they came from an earlier revision |

Reproduce: `uv run python baseline/run_baseline.py --impl baseline/reference/original --label original`.

### Acceptance criteria (agreed before implementation)

Extraction: every unique review analysed; every distinct tweak captured as its own record; no fabricated tweak (quote must be verbatim); suggestions never applied. Edits: output equals original plus recorded changes; no wrong-line edits; no invented amounts; ingredient changes reflected in steps; conflicts never overwrite. Traceability: every applied change names its review, quote and line; every extracted tweak accounted for; titles claim enhancement only when something changed. Operational: deterministic, offline-testable, explicit failures.

Agreed policy: apply only tweaks the reviewer **tried** and reported **positively**; one enhanced recipe per original; when reviews conflict, the higher-ranked review wins.

---

## 2. Decisions

### D1. Evidence before implementation

**Problem.** Changes need to be justified by evidence and reversible, but the folder is not a git repository and no model was available.

**Alternatives.** (a) Implement directly and test afterwards. (b) Build a baseline first: labelled fixtures, a frozen copy of the original, and a harness that scores any implementation the same way.

**Chosen: (b).** An early full rewrite was started before baselining. It was **parked** as `patches/candidate-rewrite.patch` (it applies to the original tree) because it bundled unverified changes, and the work restarted in small steps, each scored by the same harness.

**Trade-off.** About 40 minutes went on baseline, labels and harness before the first fix. In return every later claim is a measured before/after on identical cases, and the harness caught several of my own bugs (below).

### D2. Edit targeting: the highest-risk failure

**Problem.** Misapplied edits turn a correct recipe into a wrong one, silently. Ranked above incomplete extraction because wrong output is worse than missing output, and because extracting more first would push more edits through the broken matcher.

| Option | Correctness | Reliability | Explainability | Effort / cost | Reversibility |
|---|---|---|---|---|---|
| **1. Strict deterministic match + all-or-nothing** | Measured: single edits 3/8 → 8/8 | Deterministic | Simple rule; every refusal has a reason | Small; no API cost | High |
| 2. Model returns line numbers and full new lines | Unmeasurable offline; risks silent collateral edits in long steps | Depends on model | Whole-line rewrites are hard to audit | 2–3× | Medium |
| 3. Model rewrites the whole recipe; diff afterwards | Invented detail is exactly the observed real failure | Varies per run | Diff shows what, not why or from whom | Moderate + 1 call per recipe | Medium |

**Chosen: option 1, later strengthened with line IDs.** An edit names a line (`ing-6`, `step-3`); the line *as it was in the original recipe* must contain the find text exactly once (exact, then case/space-insensitive); otherwise the edit is refused. Calibration: every wrong target the original accepted scored ≤ 0.83, genuine near-identical lines 0.98 (7 data points).

**Trade-off.** A paraphrased find is refused rather than guessed: correct but lossy. The rate on real model output is unmeasured.

### D3. Input contract, stable IDs, ranking (`loader.py`)

- **Normalization:** HTML entities decoded, NFC, whitespace collapsed, empty lines dropped; scraped text kept as `raw_text`; amounts not rewritten. `PT0M` treated as missing. Invalid files raise `RecipeDataError` before any model call.
- **Line IDs:** positional (`ing-<n>`, `step-<n>`), with `content_hash` (written to output as `source_recipe_hash`) identifying the text they refer to. Content-addressed line IDs were rejected because duplicate lines collide.
- **Review IDs:** `rev-` + SHA-1 of normalized, case-folded text, so the same review has the same ID in `reviews` and `featured_tweaks`, after re-ordering and across re-scrapes. Duplicates merge and keep every source position.
- **Ranking:** featured → helpful votes → keyword flag → rating → file order, each only breaking ties left by the one before. **There is no vote data** in the supplied files or the scraper, so that level never separates reviews today; it reads `helpful_votes`/`helpful_count` if a future scrape adds them. On this data the ranking reproduces the earlier "rating, then file order" conflict outcomes.

### D4. Extraction: complete and checked (`prompts.py`, `tweak_extractor.py`, `support.py`)

**Problem.** One merged record per review; no evidence; suggestions indistinguishable from changes.

**Chosen.**

- Every review is extracted (flagged or not). The model returns one entry per tweak: type, **verbatim quote**, `tried`/`suggested`, reviewer outcome, reasoning, and edits that name a **line ID**.
- **Validation** in code, not only in the schema: fields and values; quote appears in the review; line exists, in the right section, with the find text on it; operation complete; and every **amount** the edit introduces is stated in the review or follows from one review amount and the line's amount by one arithmetic step (e.g. "1/2 c less" on 3 cups supports 2.5).
- **Exactly one corrective retry**, showing the model its answer and the precise problems. Tweaks valid in the first answer survive a correction that drops them; tweaks still invalid are kept as `rejected` with their problems.
- **Empty is not failure:** `ok`, `empty` (a valid "no tweaks"), `failed` (no usable answer, or every proposal rejected).

**Alternatives rejected.** Switching to structured outputs on a newer model (cannot verify availability or behaviour without a key); trusting the schema alone (the cached outputs show schema-valid answers with invented amounts); unlimited retries (unbounded cost and latency); rejecting unfamiliar *words* (my own ideal labels would fail: "plenty" for "lots"), so words only produce warnings.

**Trade-offs.** 19 model calls per full run instead of 4; amounts and their units are checked, but invented descriptors only warn; tried/suggested comes from the model, except that plan wording ("next time", "will", "would") in a `tried` quote is rejected.

### D5. Applying tweaks predictably (`recipe_modifier.py`)

Edits are written against the original recipe and verified before anything changes. Refusal reasons: `invalid`, `not_found`, `ambiguous`, `stale` (same review already changed the target), `conflict` (a higher-ranked review did), `duplicate` (the change already exists), `no_op`. A different review may edit another part of the same step but not a changed ingredient line, nor remove a changed step. Tweaks apply in review rank order; each tweak is tried on a copy and kept only if every edit applies (otherwise its edits are `rolled_back`), so an ingredient change never lands without its step edits.

**Alternatives rejected.** Last-write-wins (silent overwrites); per-edit application (half-applied tweaks, inconsistent recipes); semantic conflict detection (no reliable deterministic method).

**Trade-off.** Conflicts are line-level: tweaks that interact across lines still combine into a recipe nobody has cooked, a consequence of the "one combined recipe" product decision.

### D6. Explanation (`report.py`, `enhanced_recipe_generator.py`)

JSON plus Markdown for every recipe: applied tweaks with review ID, source positions, rating, rank and quote; every tweak not applied with its reason; extraction per review; model, prompt version and **where the answers came from** (live model or replay). The **diff is computed from the original and final text alone** and reconciled with the applied edits; `diff_verified` and `diff_problems` record the result instead of hiding mismatches. This check caught three defects in my own diff code (positional pairing, `difflib` autojunk on 200+ character lines, dissimilar substitutions shown as unrelated lines), all fixed with regression tests.

### D7. Model access and failures (`pipeline.py`, `replay.py`)

Model access is injectable (`llm_client=`). A built client has a 30 s timeout and 1 SDK retry, so a review costs at most 2 requests × 2 attempts; a missing key fails before any call (CLI exit 2). Every run returns a `RecipeRun`: `enhanced`, `unchanged` (report still written, no enhancement claimed), `skipped` (no reviews) or `failed` (`invalid_input`, `extraction_failed`, `error`, with detail). `ReplayClient` answers from recordings and turns a missing recording into an explicit failure; `RecordingClient` saves a live run for exact replay.

### D8. How the improvement is measured

Same fixtures for both versions; stand-in answers in each version's own format; random selection replaced by one run per possible choice (figures are expected values). Extraction precision is **not** taken from mocked runs (100% by construction): it comes from the cached real outputs and from a set of deliberately defective answers fed to the validator. A **held-out** fixture (H1: the four soup reviews never used while building) had its quotes labelled before implementation and its edits written from the spec, then frozen before either version ran (SHA-256 `93f98323…f4266b` as written with CRLF; `054b6341…588951` after normalizing line endings to LF, content unchanged).

---

## 3. Verified results

All mocked unless marked. Source: `baseline/results/measure/measurement.md`.

| Area | Metric | Dev fixtures: original | Dev: improved | Held-out: original | Held-out: improved |
|---|---|---|---|---|---|
| Extraction | Recall: labelled tweaks captured | 50% | 100% | 21% | 100% |
| Extraction | Recall: captured as their own record | 3% | 100% | 0% | 100% |
| Extraction | Untested or disliked tweaks applied | 1.5 | 0 | 0.3 | 0 |
| Edits | Expected line changes made | 73% | 100% | 47% | 100% |
| Edits | Unexpected line changes | 1.0 | 0 | 0 | 0 |
| Edits | Recipe exactly as the policy expects | 1 of 4 | 4 of 4 | 0 of 1 | **0 of 1 → 1 of 1†** |
| Edits | Single-edit cases (F4) | 3/8 | 8/8 | – | – |
| Attribution | Tweaks accounted for | 50% | 100% | 21% | 100% |
| Attribution | Changes naming review and quoting evidence | 0% | 100% | 0% | 100% |
| Attribution | Wrong type labels | 3.5 | 0 | 0.7 | 0 |

- **Held-out finding (fixed afterwards, †):** two lines added after the same line came out in reverse order (onion → peppers → Brussels sprouts instead of Brussels sprouts → peppers). Every line was correct; only order was wrong. Fixed at 18:36 with regression tests (additions now follow edit order within a tweak and application order across tweaks); the held-out case then matches exactly, but the set is **no longer blind** for this fix.
- **Validator (mocked defects):** 7 of 10 defective answers rejected at first measurement; **9 of 10** after adding a unit check and a plan-wording check (20:39). 26 of 26 valid labelled tweaks accepted before and after. Still missed: invented descriptor (warning only).
- **Cached real output (original era):** tweak recall 4 of 10; 4 of 4 items were real tweaks; 2 of 4 faithful to the review. No real output exists for the improved version.
- **End to end (replayed hand-written answers):** all 6 recipes run; 3 enhanced, 1 unchanged, 2 skipped (no reviews); every written diff verified; two runs byte-identical apart from timestamps. Outputs in `baseline/results/workflow/`.
- **Tests:** 205 offline tests, passing under several hash seeds; the modifier suite catches 5 of 5 deliberately injected faults.

### Cost and latency (mocked responses, all 6 recipes)

| | Original | Improved |
|---|---|---|
| Model calls | 4 | 19 |
| Input / output tokens (tiktoken) | 3,030 / 530 | 30,501 / 3,537 |
| Approx. cost per full run* | ≈ $0.002 | ≈ $0.02 |
| Pipeline time excluding the model | 18 ms | 88 ms |
| Worst case per run (timeouts × retries) | ≈ 6 h (600 s SDK default) | ≈ 38 min (30 s) |

\* At $0.50 / $1.50 per million input/output tokens, the last `gpt-3.5-turbo` list price known to me; check before relying on it. 70% of input tokens are the same 1,112-token system prompt repeated per call. Model latency was not measured.

---

## 4. Remaining limitations

1. **No live model run.** Real extraction precision, quote fidelity, line-ID accuracy and latency are unknown.
2. **Validator gaps:** invented descriptors only warn; `tried`/`suggested` still comes from the model when no plan wording is present.
3. **Paraphrased find text is refused**, not matched: safe but lossy; rate unknown.
4. **Conflicts are per line**, not by meaning; combined tweaks from several reviewers are untested as a recipe.
5. **Duplicates are matched on text**: "1 egg yolk" and "1 additional egg yolk" would both be added.
6. **Small evidence base:** 6 recipes, 19 reviews; held-out set is 8 tweaks from one recipe and is no longer blind for the ordering fix; labels are one person's judgement.
7. **Open policy decisions:** whether "used what I had" substitutions count as improvements; the recall target for live runs.
8. **Not addressed:** stale sample outputs in `data/enhanced/` and `src/data/enhanced/`; reviewer usernames (scraper collects none); output filenames can contain parentheses.

---

## 5. Prioritisation within the time box

Started 15:39. Active work about 2 h 10 min; two pauses (an hour before 18:34, and a session limit until 20:37) put the final fixes outside the 3-hour budget and the 4-hour window.

| Time | Stage | Why then |
|---|---|---|
| 15:39 | Discovery: code, data, bug probes | Understand before changing |
| 15:42–15:56 | Early rewrite (later parked) | Moved too fast; corrected at the next review |
| 15:58 | Analysis, acceptance criteria, gold labels (34 tweaks) | Criteria before code |
| 16:05 | Baseline harness, fixtures, frozen original | Reproducible before/after |
| 16:21 | Smallest defensible fix chosen (edit targeting), prototyped | Highest risk, verifiable offline |
| 16:28 | Input contract, stable IDs, ranking | Foundation for line-ID edits and attribution |
| 16:36 | Complete, validated extraction | Top priority, made safe by validation |
| 16:51 | Predictable edits, conflicts, atomic tweaks | Correctness |
| 16:57 | End-to-end workflow, reports, explicit failures | Traceability; one recipe verified before all |
| 17:10 | Measurement incl. held-out, cost, latency | Evidence before optimisation |
| 17:17 | Decision record and presentation | Explain before extending |
| 18:34 | Fix the held-out ordering bug | Correctness defect found by measurement |
| 20:37 | Unit and plan-wording checks; re-measure | Close the two validator gaps with deterministic fixes |

**Deliberately deferred:** model or prompt optimisation, switching models or structured outputs (needs a key), scraper fixes, refreshing the stale samples.

## 6. Next steps (in order)

1. **Live run with `RecordingClient`** on all recipes: real precision and recall against the gold labels, retry rate, refusal rate, latency. Commit the recordings so the run replays exactly.
2. **Label a fresh held-out set** (the current one is no longer blind for the ordering fix).
3. **Decide the open policies** ("used what I had" swaps; recall target).
4. **Review descriptor warnings** from the live run before deciding whether to reject them.
5. **Only then reduce cost:** shorter system prompt (the worked example is 462 tokens), prompt caching, one call per recipe, or parallel calls, each re-measured against the same cases.
6. Replace or delete the stale committed sample outputs.

## 7. Reproduce

```bash
uv sync
uv run pytest                                                                   # 205 offline tests
uv run python baseline/run_baseline.py --impl baseline/reference/original --label original
uv run python baseline/run_baseline.py --impl src --label improved
uv run python baseline/run_baseline.py --impl src --label improved-heldout --fixtures "heldout/H*.json"
uv run --with tiktoken python baseline/measure.py cost --impl src               # also: validator, cached, summary
cd src && uv run python test_pipeline.py all --replay ../baseline/recordings/*.json
```

## 8. Where things are

| Path | Contents |
|---|---|
| `src/llm_pipeline/loader.py` | Input contract, normalization, IDs, deduplication, ranking |
| `src/llm_pipeline/prompts.py`, `tweak_extractor.py`, `support.py` | Extraction prompt (`PROMPT_VERSION`), validation, retry, support checks |
| `src/llm_pipeline/recipe_modifier.py` | Line-ID edits, conflicts, atomic tweaks |
| `src/llm_pipeline/report.py`, `enhanced_recipe_generator.py` | Verified diff, JSON and Markdown reports |
| `src/llm_pipeline/pipeline.py`, `replay.py` | Orchestration, `RecipeRun`, injectable and replay clients |
| `baseline/` | Frozen original, fixtures (incl. held-out), recordings, harness, measurement, results |
| `patches/candidate-rewrite.patch` | The parked early rewrite |
