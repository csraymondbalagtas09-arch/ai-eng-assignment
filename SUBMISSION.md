# Submission: recipe tweak pipeline

**Candidate:** Raymond Balagtas · **Branch:** `submission` · **Based on:** `Casper-Studios/ai-eng-assignment@573d9287b6` (starter history and authorship unchanged; all work is in the commits after it)

**Video walkthrough:** [VIDEO LINK: to be added] · **Slides:** [SLIDES LINK OR `docs/presentation.pdf`: to be added]

## What I did

I prioritised the three goals in order: **complete tweak extraction**, **correct edits**, **traceable explanations**. I built a baseline first (labelled fixtures, a frozen copy of the starter, and a harness that scores any version the same way), then changed the pipeline in small, measured steps:

- **Extraction:** every review is analysed (not one random flagged review); each tweak is its own record with a verbatim quote, tried/suggested status and edits that target stable line IDs. Output is validated (quote in review, line references, amounts and units supported by the review) with one corrective retry; "no tweaks" is a valid result, distinct from failure.
- **Edits:** each edit must find its exact text on the named line or it is refused; tweaks apply all-or-nothing in a documented review rank order; stale, conflicting and duplicate changes are refused with reasons.
- **Explanations:** every recipe gets JSON and Markdown reports with a line-level diff computed from the text and reconciled with the applied edits, the review and quote behind each change, and a reason for everything not applied.

Full reasoning, alternatives and trade-offs: [`docs/engineering-decisions.md`](docs/engineering-decisions.md).

## Results

**No live model results:** no API key was available. Figures are from hand-written stand-in answers (mocked) unless marked cached (the starter's committed real outputs). Details: [`baseline/results/measure/measurement.md`](baseline/results/measure/measurement.md).

| Metric | Starter | Submission |
|---|---|---|
| Tweak recall, development fixtures / held-out | 50% / 21% | 100% / 100% |
| Expected line changes made, development / held-out | 73% / 47% | 100% / 100% |
| Recipe exactly as expected, development / held-out | 1 of 4 / 0 of 1 | 4 of 4 / 1 of 1† |
| Changes citing their review and evidence | 0% | 100% |
| Validator: defective answers rejected / valid accepted | – | 9 of 10 / 26 of 26 |
| Model calls per full run (6 recipes) | 4 | 19 |

† 0 of 1 until a bug the held-out set found was fixed; that set is no longer blind for the fix. Cached real output from the starter era found 4 of 10 labelled tweaks.

## Run it

```bash
uv sync
uv run pytest                                                       # 205 offline tests, no API key needed
uv run python baseline/run_baseline.py --impl src --label check     # score the submission on the labelled fixtures
cd src && uv run python test_pipeline.py all --replay ../baseline/recordings/*.json   # full pipeline, recorded answers
```

With `OPENAI_API_KEY` in `.env`, `uv run python test_pipeline.py all` (from `src/`) runs against the live model. The replay recordings are hand-written ideal answers, not model output; reports say which source they used.

## Known limitations

No live run yet, so real extraction precision and latency are unknown; invented descriptors (e.g. "bell peppers, diced") are only warned about; edits whose text the model paraphrases are refused rather than matched; conflicts are checked per line, not by meaning; the held-out set is small (8 tweaks). The starter's sample outputs in `data/enhanced/` and `src/data/enhanced/` are left as they were and do not match the new output format.

## AI assistance

This work was done with Claude Code (Claude Opus 5.5) as a pair: I set the priorities and stage goals and made the policy decisions; the agent proposed, implemented and verified changes. The unedited session transcript is in [`agent-conversation/`](agent-conversation/). Commits are co-authored accordingly.
