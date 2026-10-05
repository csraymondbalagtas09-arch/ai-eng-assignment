"""
Step 4: Explanation

Builds the line-level diff between the original and the enhanced recipe and renders
the human-readable Markdown report.

The diff is computed from the two texts with difflib, independently of what the
modifier says it did, and then cross-checked against the applied edit outcomes:
every changed line must be explained by an applied edit, and every applied edit must
be visible in the diff. Anything that does not reconcile is reported as a problem
instead of being hidden.
"""

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from .models import EnhancedRecipe, LineDiff, Recipe, TweakOutcome

SECTIONS = (("ingredients", "ing"), ("instructions", "step"))


def _norm(text: str) -> str:
    return " ".join((text or "").split()).casefold()


PAIR_THRESHOLD = 0.6  # lines less similar than this are a removal plus an addition


def _align(before: List[str], after: List[str]) -> List[Tuple[int, int]]:
    """Order-preserving pairing of similar lines that maximizes total similarity."""
    n, m = len(before), len(after)
    # autojunk=False: on strings of 200+ characters the default treats common
    # characters as junk and badly underestimates similarity
    ratio = [[SequenceMatcher(None, b.casefold(), a.casefold(), autojunk=False).ratio() for a in after]
             for b in before]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            pair = ratio[i][j] + best[i + 1][j + 1] if ratio[i][j] >= PAIR_THRESHOLD else -1.0
            best[i][j] = max(pair, best[i + 1][j], best[i][j + 1])
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if ratio[i][j] >= PAIR_THRESHOLD and best[i][j] == ratio[i][j] + best[i + 1][j + 1]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            i += 1
        else:
            j += 1
    return pairs


def compute_diff(original: Recipe, final: Recipe) -> List[LineDiff]:
    """Line changes from original to final, located by original line ID."""
    diffs: List[LineDiff] = []
    for section, prefix in SECTIONS:
        before, after = getattr(original, section), getattr(final, section)
        matcher = SequenceMatcher(a=before, b=after, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            # Inside a changed block, pair each line with its most similar counterpart
            pairs = dict(_align(before[i1:i2], after[j1:j2]))
            paired_after = {j: i for i, j in pairs.items()}
            anchor = i1 - 1  # last original line at or before the current position
            i = i1
            for j in range(j1, j2):
                if (j - j1) in paired_after:
                    k = i1 + paired_after[j - j1]
                    for r in range(i, k):  # unpaired original lines before this one
                        diffs.append(LineDiff(section=section, kind="removed", line_id=f"{prefix}-{r}",
                                              before=before[r]))
                    diffs.append(LineDiff(section=section, kind="changed", line_id=f"{prefix}-{k}",
                                          before=before[k], after=after[j]))
                    i, anchor = k + 1, k
                else:
                    diffs.append(LineDiff(section=section, kind="added",
                                          line_id=f"{prefix}-{anchor}" if anchor >= 0 else None, after=after[j]))
            for r in range(i, i2):
                diffs.append(LineDiff(section=section, kind="removed", line_id=f"{prefix}-{r}", before=before[r]))
    return diffs


def _merge_substitutions(diffs: List[LineDiff], outcomes: List[TweakOutcome]) -> List[LineDiff]:
    """
    Show a replacement as one changed line even when old and new text are too
    dissimilar to pair by text (e.g. half-and-half -> 2% milk). Only merged when the
    diff independently shows the old line gone and the edit's exact result present.
    """
    results = {}
    for outcome in outcomes:
        if outcome.status == "applied":
            for edit in outcome.edit_outcomes:
                if edit.before is not None:
                    results[edit.line_id] = edit.after  # last edit wins; None = removed
    removed = {(d.section, d.line_id): d for d in diffs if d.kind == "removed"}
    merged, dropped = {}, set()
    for i, d in enumerate(diffs):
        if d.kind != "added":
            continue
        for (section, line_id), r in removed.items():
            text = results.get(line_id)
            if section == d.section and line_id not in merged and text and _norm(text) == _norm(d.after):
                merged[line_id] = d.after
                dropped.add(i)
                break
    return [
        LineDiff(section=d.section, kind="changed", line_id=d.line_id, before=d.before, after=merged[d.line_id])
        if d.kind == "removed" and d.line_id in merged else d
        for i, d in enumerate(diffs) if i not in dropped
    ]


def attribute_diff(diffs: List[LineDiff], outcomes: List[TweakOutcome]) -> Tuple[List[LineDiff], List[str]]:
    """Attach the applied tweaks that explain each line change; list what does not reconcile."""
    diffs = _merge_substitutions(diffs, outcomes)
    by_line: Dict[str, List[Tuple[str, object]]] = defaultdict(list)
    adds: List[Tuple[str, str]] = []
    for outcome in outcomes:
        if outcome.status != "applied":
            continue
        for edit in outcome.edit_outcomes:
            if edit.before is None:
                adds.append((outcome.tweak_id, edit.after))
            else:
                by_line[edit.line_id].append((outcome.tweak_id, edit))

    problems: List[str] = []
    explained_lines, explained_adds = set(), set()
    for diff in diffs:
        if diff.kind == "added":
            diff.tweak_ids = [tid for tid, text in adds if _norm(text) == _norm(diff.after)]
            explained_adds.update((tid, _norm(diff.after)) for tid in diff.tweak_ids)
        else:
            edits = by_line.get(diff.line_id, [])
            diff.tweak_ids = list(dict.fromkeys(tid for tid, _ in edits))
            last = edits[-1][1] if edits else None
            expected_after = None if last is None else last.after
            if last is not None and _norm(expected_after) != _norm(diff.after):
                problems.append(f"{diff.section} {diff.line_id}: the text change is not what the applied edits produce")
            explained_lines.add(diff.line_id)
        if not diff.tweak_ids:
            problems.append(f"{diff.section} {diff.line_id}: {diff.kind} line not explained by any applied tweak")

    for line_id, edits in by_line.items():
        if line_id not in explained_lines:
            problems.append(f"{line_id}: applied by {edits[-1][0]} but no change is visible")
    for tid, text in adds:
        if (tid, _norm(text)) not in explained_adds:
            problems.append(f"line '{text}' added by {tid} is not in the result")
    return diffs, problems


def _cell(text) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def render_markdown(recipe: EnhancedRecipe) -> str:
    """Human-readable report: what changed, why, from whom, and what was not applied."""
    reviews = {r.review_id: r for r in recipe.reviews}
    tweaks = {t.id: t for result in recipe.extraction_results for t in result.tweaks}
    outcomes = {o.tweak_id: o for o in recipe.tweak_outcomes}
    count = defaultdict(int)
    for o in recipe.tweak_outcomes:
        count[o.status] += 1

    def who(review_id: str) -> str:
        r = reviews.get(review_id)
        if not r:
            return review_id
        rating = f"{r.rating}★" if r.rating else "no rating"
        return f"{review_id} ({', '.join(r.sources)}; {rating}; rank {r.rank})"

    verified = {True: "verified", False: "**NOT verified**", None: "not checked"}[recipe.diff_verified]
    lines = [
        f"# {recipe.title}",
        "",
        f"Recipe `{recipe.original_recipe_id}` · status **{recipe.status}** · "
        f"pipeline {recipe.pipeline_version} · source hash `{recipe.source_recipe_hash}`",
        "",
        f"Extraction answers: **{recipe.extraction_source or 'unknown'}** · model `{recipe.model}` · "
        f"prompt `{recipe.prompt_version}`",
        "",
        f"Reviews analysed: {len(recipe.extraction_results)} · tweaks extracted: {len(tweaks)} · "
        f"applied: {count['applied']} · refused: {count['refused']} · skipped: {count['skipped']} · "
        f"line changes: {len(recipe.recipe_diff)} ({verified} against the applied edits)",
        "",
        "## What changed",
        "",
    ]
    if recipe.recipe_diff:
        lines += ["| Section | Line | Change | Before | After | Tweak |", "|---|---|---|---|---|---|"]
        for d in recipe.recipe_diff:
            lines.append(f"| {d.section} | {d.line_id or 'top'} | {d.kind} | {_cell(d.before)} | "
                         f"{_cell(d.after)} | {', '.join(d.tweak_ids) or '**unexplained**'} |")
    else:
        lines.append("Nothing. No tweak could be applied; see below for why.")
    for problem in recipe.diff_problems:
        lines.append(f"\n> **Diff problem:** {problem}")

    lines += ["", "## Applied tweaks", ""]
    for o in recipe.tweak_outcomes:
        if o.status != "applied":
            continue
        t = tweaks[o.tweak_id]
        lines += [
            f"### {t.id} · {t.modification_type}",
            f"- **Review:** {who(t.review_id)}",
            f"- **Evidence:** \"{t.evidence_quote}\"",
            f"- **Reviewer's reason or result:** {t.reasoning}",
            "- **Edits:** " + "; ".join(
                f"{e.line_id} `{_cell(e.before) or '(new line)'}` → `{_cell(e.after) or '(removed)'}`"
                for e in o.edit_outcomes),
        ]
        lines += [f"- **Warning:** {w}" for w in t.warnings]
        lines.append("")
    if not count["applied"]:
        lines += ["None.", ""]

    lines += ["## Not applied", ""]
    not_applied = [o for o in recipe.tweak_outcomes if o.status != "applied"]
    if not_applied:
        lines += ["| Tweak | Review | Evidence | Outcome | Why |", "|---|---|---|---|---|"]
        for o in not_applied:
            t = tweaks.get(o.tweak_id)
            if o.status == "skipped":
                why = {"not_tried": "suggestion the reviewer did not try",
                       "outcome_not_positive": f"reviewer reported a {t.reviewer_outcome if t else ''} result",
                       "no_edits": "no concrete edit possible"}.get(o.reason, o.reason)
            else:
                why = "; ".join(f"{e.edit_id.rsplit('.', 1)[-1]} {e.reason}: {e.detail}"
                                for e in o.edit_outcomes if e.status == "refused")
            lines.append(f"| {o.tweak_id} | {who(t.review_id) if t else ''} | "
                         f"{_cell(t.evidence_quote) if t else ''} | {o.status} | {_cell(why)} |")
    else:
        lines.append("Nothing: every extracted tweak was applied.")

    rejected = [(r.review_id, rej) for r in recipe.extraction_results for rej in r.rejected]
    failed = [r for r in recipe.extraction_results if r.status == "failed"]
    lines += ["", "## Extraction", "",
              "| Review | Status | Tweaks | Rejected proposals | Attempts |", "|---|---|---|---|---|"]
    for r in recipe.extraction_results:
        lines.append(f"| {who(r.review_id)} | {r.status} | {len(r.tweaks)} | {len(r.rejected)} | {r.attempts} |")
    for review_id, rej in rejected:
        lines.append(f"\n- Rejected from {review_id}: \"{rej.quote}\": {'; '.join(rej.problems)}")
    for r in failed:
        lines.append(f"\n- Extraction failed for {r.review_id}: {'; '.join(r.errors) or 'all proposals rejected'}")

    lines += ["", "## Recipe", "", "### Ingredients", ""] + [f"- {x}" for x in recipe.ingredients]
    lines += ["", "### Instructions", ""] + [f"{i}. {x}" for i, x in enumerate(recipe.instructions, start=1)]
    return "\n".join(lines) + "\n"
