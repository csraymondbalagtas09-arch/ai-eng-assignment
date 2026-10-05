#!/usr/bin/env python3
"""
Baseline harness for the recipe-enhancement pipeline.

Runs the labelled fixtures in baseline/fixtures through one implementation of the
llm_pipeline package and records expected versus actual results, scored separately:

  extraction  which reviews were analysed and which labelled tweaks were captured
  edits       whether the edits handed to the modifier produced the right recipe text
  reporting   whether the written output describes what actually happened

The LLM is replaced by an oracle that returns the labelled ideal extraction in the
implementation's own response format, so runs are deterministic and each stage is
judged only on the input it actually received. Random review selection is replaced
by running once per possible choice. No network calls are made.

Usage (from the project root):
    uv run python baseline/run_baseline.py --impl baseline/reference/original --label original
"""

import argparse
import hashlib
import html
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "baseline" / "fixtures"
RESULTS = ROOT / "baseline" / "results"
ENHANCED_SUFFIX = "(Community Enhanced)"
TARGETS = ("ingredients", "instructions")


# --- Labels: reference applier and self-checks -----------------------------------


def clean(text):
    """Compare recipe text without HTML entities; decoding them is not an edit."""
    return html.unescape(text)


def reference_apply(recipe, edits):
    """Apply labelled edits by their explicit line numbers (no searching)."""
    slots = {
        t: [{"text": x, "removed": False, "after": []} for x in recipe[t]] for t in TARGETS
    }
    for e in edits:
        slot = slots[e["target"]][e["line"]]
        if e["operation"] == "replace":
            if slot["removed"] or slot["text"].count(e["find"]) != 1:
                raise ValueError(
                    f"label error: '{e['find']}' is not exactly once in {e['target']}[{e['line']}]"
                )
            slot["text"] = slot["text"].replace(e["find"], e["replace"])
        elif e["operation"] == "remove":
            slot["removed"] = True
        else:
            slot["after"].append(e["add"])
    result = {}
    for t, items in slots.items():
        result[t] = []
        for s in items:
            if not s["removed"]:
                result[t].append(s["text"])
            result[t].extend(s["after"])
    return result


def expected_recipe(fx):
    exp, original = fx["expected"]["recipe"], fx["recipe"]
    if exp == "unchanged":
        return {t: original[t] for t in TARGETS}
    return {t: original[t] if exp[t] == "unchanged" else exp[t] for t in TARGETS}


def normalized(recipe):
    return {t: [clean(x) for x in recipe[t]] for t in TARGETS}


def tweak_by_id(fx):
    return {t["id"]: t for t in fx.get("tweaks", [])}


def check_labels(fx):
    """Return problems with a fixture's labels; the baseline is invalid if any exist."""
    problems = []
    prov = fx["provenance"]
    if "recipe" in fx:
        src = json.loads((ROOT / prov["file"]).read_text(encoding="utf-8"))
        recipe = fx["recipe"]
        if any(recipe[t] != src[t] for t in TARGETS):
            problems.append("recipe text differs from its source file")
        if recipe["reviews"] != [src["reviews"][i] for i in prov["review_indices"]]:
            problems.append("review text differs from its source file")
        for t in fx["tweaks"]:
            if t["quote"] not in recipe["reviews"][t["review"]]["text"]:
                problems.append(f"{t['id']}: quote is not verbatim in review {t['review']}")
            for e in t["edits"]:
                line = recipe[e["target"]][e["line"]]
                if e["operation"] == "replace" and line.count(e["find"]) != 1:
                    problems.append(f"{t['id']}: '{e['find']}' not exactly once in its line")
                if e["operation"] != "replace" and line != e["find"]:
                    problems.append(f"{t['id']}: {e['operation']} find is not the whole line")
        exp = fx["expected"]
        ids = set(tweak_by_id(fx))
        if set(exp["applied"]) & set(exp["skipped"]) or set(exp["applied"]) | set(exp["skipped"]) != ids:
            problems.append("expected applied/skipped do not partition the tweaks")
        edits = [e for tid in exp["applied"] for e in tweak_by_id(fx)[tid]["edits"]]
        try:
            if normalized(reference_apply(recipe, edits)) != normalized(expected_recipe(fx)):
                problems.append("expected.recipe differs from applying expected.applied")
        except ValueError as err:
            problems.append(str(err))
    else:
        for name, path in prov.items():
            src = json.loads((ROOT / path).read_text(encoding="utf-8"))
            if any(fx["recipes"][name][t] != src[t] for t in TARGETS):
                problems.append(f"recipe '{name}' differs from its source file")
    return problems


# --- Comparing recipe text -------------------------------------------------------


def line_changes(original, result):
    """Multiset of line-level changes: ('-', target, text) and ('+', target, text)."""
    changes = Counter()
    for t in TARGETS:
        before = Counter(clean(x) for x in original[t])
        after = Counter(clean(x) for x in result[t])
        for text, n in (before - after).items():
            changes[("-", t, text)] += n
        for text, n in (after - before).items():
            changes[("+", t, text)] += n
    return changes


def net_record_effects(records):
    """Net line changes claimed by a list of ChangeRecords, applied in order."""
    net = Counter()

    def apply(sign, target, text):
        opposite = ("+" if sign == "-" else "-", target, text)
        if net[opposite] > 0:
            net[opposite] -= 1
        else:
            net[(sign, target, text)] += 1

    for r in records:
        target = r["type"] + "s"
        if r["operation"] in ("replace", "remove"):
            apply("-", target, clean(r["from_text"]))
        if r["operation"] in ("replace", "add"):
            apply("+", target, clean(r["to_text"]))
    return +net


def describe(changes):
    return [f"{sign} {target}: {text}" for (sign, target, text), n in sorted(changes.items()) for _ in range(n)]


# --- Implementation adapters -----------------------------------------------------


def load_impl(impl):
    impl = (ROOT / impl).resolve()
    sys.path.insert(0, str(impl))
    for name in [m for m in sys.modules if m == "llm_pipeline" or m.startswith("llm_pipeline.")]:
        del sys.modules[name]
    import llm_pipeline  # noqa: E402
    import llm_pipeline.pipeline  # noqa: E402

    if Path(llm_pipeline.__file__).resolve().parent != impl / "llm_pipeline":
        raise SystemExit(f"imported llm_pipeline from the wrong place: {llm_pipeline.__file__}")
    return llm_pipeline, impl


class OriginalAdapter:
    """Drives the original pipeline: one randomly chosen flagged review, one modification."""

    def __init__(self, pkg):
        self.pkg = pkg

    @staticmethod
    def oracle(fx, review_index):
        """Ideal answer to the original prompt: changes the user made, merged into one object."""
        tweaks = [t for t in fx["tweaks"] if t["review"] == review_index]
        tried = [t for t in tweaks if t["status"] == "tried"]
        edits = [e for t in tried for e in t["edits"]]
        source = tried or tweaks
        response = {
            "modification_type": source[0]["type"] if source else "addition",
            "reasoning": "; ".join(t["quote"] for t in tried) or "No change was made",
            "edits": [
                {k: e[k] for k in ("target", "operation", "find", "replace", "add") if k in e}
                for e in edits
            ],
        }
        return response, [t["id"] for t in tried], edits

    def run_fixture(self, fx):
        reviews = fx["recipe"]["reviews"]
        texts = [" ".join(clean(r["text"]).split()) for r in reviews]
        flagged = [i for i, r in enumerate(reviews) if r.get("has_modification")]
        runs = []
        for position, choice in enumerate(flagged or [None]):
            if runs and runs[-1]["selection_mode"] == "deterministic":
                break  # the implementation does not choose randomly; one run covers it
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                recipe_file = tmp / "recipe.json"
                recipe_file.write_text(json.dumps(fx["recipe"], ensure_ascii=False), encoding="utf-8")
                analysed = []

                def create(**kwargs):
                    prompt = kwargs["messages"][0]["content"]
                    review = re.search(
                        r'User Review: "(.*?)"\n\nExtract the recipe modifications', prompt, re.S
                    ).group(1)
                    index = texts.index(" ".join(clean(review).split()))
                    analysed.append(index)
                    response, _, _ = self.oracle(fx, index)
                    return NS(choices=[NS(message=NS(content=json.dumps(response)))])

                with capture_warnings(tmp) as warnings, mock.patch(
                    "random.choice", side_effect=lambda seq: seq[position]
                ) as random_choice:
                    pipeline = self.pkg.pipeline.LLMAnalysisPipeline(
                        openai_api_key="not-used", output_dir=str(tmp / "out")
                    )
                    pipeline.tweak_extractor.client = NS(chat=NS(completions=NS(create=create)))
                    pipeline.process_single_recipe(str(recipe_file), save_output=True)
                files = sorted((tmp / "out").glob("*.json"))
                output = json.loads(files[0].read_text(encoding="utf-8")) if files else None

            random_mode = random_choice.called
            if not random_mode:
                choice = analysed[0] if analysed else None  # what the implementation picked
            _, captured, edits = self.oracle(fx, choice) if choice is not None else (None, [], [])
            entries = []
            if output:
                output.pop("created_at", None)
                for m in output["modifications_applied"]:
                    entries.append({
                        "kind": "applied",
                        "tweaks": captured,
                        "type": m["modification_type"],
                        "has_review_pointer": "review_index" in m["source_review"],
                        "has_evidence_quote": "evidence_quote" in m,
                    })
            runs.append({
                "selection": "none" if choice is None else
                             f"reviews[{choice}] ({'random choice enumerated' if random_mode else 'deterministic'})",
                "selection_mode": "random" if random_mode else "deterministic",
                "reviews_analysed": sorted(set(analysed)),
                "captured": captured,
                "groups": [captured] if captured else [],
                "passed_on": captured,
                "edits_given": edits,
                "entries": entries,
                "output": output,
                "warnings": warnings,
            })
        return runs

    def run_edit_case(self, recipe, edit):
        models = self.pkg.models
        r = models.Recipe(recipe_id=recipe["recipe_id"], title=recipe["title"],
                          ingredients=recipe["ingredients"], instructions=recipe["instructions"])
        e = models.ModificationEdit(**{k: edit[k] for k in ("target", "operation", "find", "replace", "add") if k in edit})
        mod = models.ModificationObject(modification_type="technique_change", reasoning="baseline", edits=[e])
        with capture_warnings("<none>"):
            modified, records = self.pkg.recipe_modifier.RecipeModifier().apply_modification(r, mod)
        return {
            "result": {"ingredients": modified.ingredients, "instructions": modified.instructions},
            "records": [x.model_dump() for x in records],
        }


class capture_warnings:
    """Collect the implementation's warning/error log lines (loguru), sanitised."""

    def __init__(self, tmp):
        self.tmp = str(tmp)
        self.lines = []

    def __enter__(self):
        from loguru import logger

        logger.remove()
        self.sink = logger.add(
            lambda m: self.lines.append(m.record["message"].replace(self.tmp, "<tmp>")),
            level="WARNING",
        )
        return self.lines

    def __exit__(self, *exc):
        from loguru import logger

        logger.remove(self.sink)
        return False


class TweakAdapter(OriginalAdapter):
    """Drives implementations whose extractor returns every tweak per review (extract_tweaks)."""

    @staticmethod
    def oracle_tweaks(fx, review_index):
        """Ideal answer to the per-tweak prompt: every labelled tweak, suggestions included."""
        prefix = {"ingredients": "ing", "instructions": "step"}
        return [
            {
                "modification_type": t["type"], "quote": t["quote"], "status": t["status"],
                "reviewer_outcome": t["outcome"], "reasoning": "As stated by the reviewer.",
                "edits": [{"target": e["target"], "operation": e["operation"],
                           "line_id": f"{prefix[e['target']]}-{e['line']}", "find": e["find"],
                           "replace": e.get("replace"), "add": e.get("add")} for e in t["edits"]],
            }
            for t in fx["tweaks"] if t["review"] == review_index
        ]

    def run_fixture(self, fx):
        reviews = fx["recipe"]["reviews"]
        texts = [" ".join(clean(r["text"]).split()) for r in reviews]
        by_quote = {" ".join(t["quote"].split()).casefold(): t["id"] for t in fx["tweaks"]}
        labelled_edits = {t["id"]: t["edits"] for t in fx["tweaks"]}
        analysed = []

        def create(**kwargs):
            review = re.search(r'"""(.*)"""', kwargs["messages"][1]["content"], re.S).group(1)
            index = texts.index(" ".join(clean(review).split()))
            analysed.append(index)
            content = json.dumps({"tweaks": self.oracle_tweaks(fx, index)})
            return NS(choices=[NS(message=NS(content=content), finish_reason="stop")])

        def label(quote):
            return by_quote.get(" ".join((quote or "").split()).casefold())

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            recipe_file = tmp / "recipe.json"
            recipe_file.write_text(json.dumps(fx["recipe"], ensure_ascii=False), encoding="utf-8")
            with capture_warnings(tmp) as warnings:
                pipeline = self.pkg.pipeline.LLMAnalysisPipeline(
                    openai_api_key="not-used", output_dir=str(tmp / "out"))
                pipeline.tweak_extractor.client = NS(chat=NS(completions=NS(create=create)))
                pipeline.process_single_recipe(str(recipe_file), save_output=True)
                # Score extraction from what the implementation accepted, whether or
                # not the pipeline went on to write an output
                loaded = self.pkg.loader.load_recipe_file(str(recipe_file))
                extracted = pipeline.tweak_extractor.extract_all(loaded.reviews, loaded.recipe)
            files = sorted((tmp / "out").glob("*.json"))
            output = json.loads(files[0].read_text(encoding="utf-8")) if files else None

        captured = [label(t.evidence_quote) for _, r in extracted for t in r.tweaks]
        captured = [c for c in captured if c]
        entries, applied = [], []
        if output:
            output.pop("created_at", None)
            for m in output["modifications_applied"]:
                tid = label(m.get("evidence_quote"))
                applied.append(tid)
                entries.append({
                    "kind": "applied", "tweaks": [tid] if tid else [], "type": m["modification_type"],
                    "has_review_pointer": bool(m["source_review"].get("review_id")),
                    "has_evidence_quote": bool(m.get("evidence_quote")),
                })
            for result in output.get("extraction_results", []):
                for t in result["tweaks"]:
                    tid = label(t["evidence_quote"])
                    if tid not in applied:
                        entries.append({"kind": "recorded", "tweaks": [tid] if tid else [],
                                        "type": t["modification_type"], "has_review_pointer": True,
                                        "has_evidence_quote": bool(t["evidence_quote"])})
        applied = [a for a in applied if a]
        first = sorted(set(analysed))
        return [{
            "selection": "all reviews extracted (deterministic)",
            "selection_mode": "deterministic",
            "reviews_analysed": first,
            "captured": captured,
            "groups": [[c] for c in captured],
            "passed_on": applied,
            "edits_given": [e for tid in applied for e in labelled_edits[tid]],
            "entries": entries,
            "output": output,
            "warnings": sorted(set(warnings)),
        }]


class LineTargetAdapter(TweakAdapter):
    """Implementations whose modifier applies Tweaks by line ID (apply_tweaks)."""

    def run_edit_case(self, recipe, edit):
        models = self.pkg.models
        prefix = {"ingredients": "ing", "instructions": "step"}
        r = self.pkg.loader.parse_recipe(dict(recipe))
        rid, tid = "rev-0000000000", "rev-0000000000.t1"
        tweak = models.Tweak(id=tid, review_id=rid, modification_type="technique_change", reasoning="baseline",
                             edits=[models.TweakEdit(
                                 id=tid + ".e1", tweak_id=tid, target=edit["target"], operation=edit["operation"],
                                 line_id=f"{prefix[edit['target']]}-{edit['line']}", find=edit["find"],
                                 replace=edit.get("replace"), add=edit.get("add"))])
        with capture_warnings("<none>"):
            modified, [outcome] = self.pkg.recipe_modifier.RecipeModifier().apply_tweaks(r, [tweak], {rid: 1})
        return {
            "result": {"ingredients": modified.ingredients, "instructions": modified.instructions},
            "records": [c.model_dump() for c in self.pkg.recipe_modifier.change_records(outcome)],
        }


def adapter_for(pkg):
    if hasattr(pkg.recipe_modifier.RecipeModifier, "apply_tweaks"):
        return LineTargetAdapter(pkg)
    if hasattr(pkg.tweak_extractor.TweakExtractor, "extract_single_modification"):
        return OriginalAdapter(pkg)
    if hasattr(pkg.tweak_extractor.TweakExtractor, "extract_tweaks"):
        return TweakAdapter(pkg)
    raise SystemExit("No adapter for this implementation")


# --- Scoring ---------------------------------------------------------------------


def score_run(fx, run):
    tweaks = fx["tweaks"]
    reviews = fx["recipe"]["reviews"]
    original = {t: fx["recipe"][t] for t in TARGETS}
    out = run["output"]
    actual_recipe = {t: out[t] for t in TARGETS} if out else original
    actual = line_changes(original, actual_recipe)

    # Extraction: judged against all labelled tweaks
    analysed = set(run["reviews_analysed"])
    captured = set(run["captured"])
    missed = {}
    for t in tweaks:
        if t["id"] in captured:
            continue
        if t["review"] not in analysed:
            why = "review never analysed: " + (
                "no keyword flag" if not reviews[t["review"]].get("has_modification") else "another review was chosen"
            )
        elif t["status"] == "suggested":
            why = "suggestion has no representation in the output schema"
        else:
            why = "not represented in the extraction"
        missed[t["id"]] = why
    eligible = {t["id"] for t in tweaks if t["status"] == "tried" and t["outcome"] == "positive"}
    by_id = tweak_by_id(fx)
    extraction = {
        "reviews_analysed": f"{len(analysed)}/{len(reviews)}",
        "tweaks_captured": f"{len(captured)}/{len(tweaks)}",
        "tried_tweaks_captured": f"{len(captured & {t['id'] for t in tweaks if t['status'] == 'tried'})}"
                                 f"/{sum(t['status'] == 'tried' for t in tweaks)}",
        "captured_as_own_record": f"{sum(len(g) == 1 for g in run['groups'])}/{len(captured)}",
        "missed": missed,
        "ineligible_passed_to_modifier": {
            tid: f"{by_id[tid]['status']}, outcome {by_id[tid]['outcome']}"
            for tid in run["passed_on"] if tid not in eligible
        },
    }

    # Edits: judged only against the edits the modifier was given
    expected_recipe_for_edits = reference_apply(original, run["edits_given"])
    expected = line_changes(original, expected_recipe_for_edits)
    edits = {
        "edits_given": len(run["edits_given"]),
        "expected_line_changes": sum(expected.values()),
        "correct": sum((expected & actual).values()),
        "missing": describe(expected - actual),
        "unexpected": describe(actual - expected),
        "recipe_exact": normalized(actual_recipe) == normalized(expected_recipe_for_edits),
    }

    # Reporting: judged against what actually happened to the text and the labels
    if out:
        records = [c for m in out["modifications_applied"] for c in m["changes_made"]]
        noop = [r for r in records if r["operation"] == "replace" and r["from_text"] == r["to_text"]]
        net = net_record_effects(records)
        accounted = {tid for e in run["entries"] for tid in e["tweaks"]}
        reporting = {
            "output_written": True,
            "change_records": len(records),
            "noop_records": len(noop),
            "claimed_but_not_in_text": describe(net - actual),
            "in_text_but_not_recorded": describe(actual - net),
            "summary_total_changes": out["enhancement_summary"]["total_changes"],
            "title_claims_enhancement": ENHANCED_SUFFIX in out["title"],
            "text_actually_changed": sum(actual.values()) > 0,
            "tweaks_accounted_for": f"{len(accounted)}/{len(tweaks)}",
            "tweak_records_with_no_tweak": sum(1 for e in run["entries"] if not e["tweaks"]),
            "type_mislabelled": sorted(
                tid for e in run["entries"] for tid in e["tweaks"] if by_id[tid]["type"] != e["type"]
            ),
            "records_with_review_pointer": f"{sum(e['has_review_pointer'] for e in run['entries'])}/{len(run['entries'])}",
            "records_with_evidence_quote": f"{sum(e['has_evidence_quote'] for e in run['entries'])}/{len(run['entries'])}",
        }
    else:
        reporting = {"output_written": False, "tweaks_accounted_for": f"0/{len(tweaks)}"}
    reporting["title_honest"] = (
        reporting.get("title_claims_enhancement", False) == reporting.get("text_actually_changed", False)
    )

    # End to end: the agreed policy's expected recipe for this fixture
    want = expected_recipe(fx)
    end_to_end = {
        "recipe_exact": normalized(actual_recipe) == normalized(want),
        "missing": describe(line_changes(original, want) - actual),
        "unexpected": describe(actual - line_changes(original, want)),
    }
    return {"extraction": extraction, "edits": edits, "reporting": reporting, "end_to_end": end_to_end}


def score_edit_case(case, recipe, outcome):
    original = {t: recipe[t] for t in TARGETS}
    actual = line_changes(original, outcome["result"])
    expect = case["expect"]
    if expect["outcome"] == "apply":
        target = case["edit"]["target"]
        wanted = {t: list(original[t]) for t in TARGETS}
        wanted[target][expect["line"]] = expect["after"]
        expected = line_changes(original, wanted)
    else:
        expected = Counter()

    if not actual:
        result = "phantom record" if outcome["records"] else "refused"
    elif expect["outcome"] == "apply" and actual == expected:
        result = "applied correctly"
    else:
        result = "wrong edit"
    passed = result == "applied correctly" or (
        result == "refused"
        and (expect["outcome"] == "refuse" or bool(expect.get("refuse_acceptable")))
    )
    edit = case["edit"]
    if expect["outcome"] == "refuse":
        expected_text = "refuse"
    else:
        change = (f"-> '{expect['after']}'" if len(expect["after"]) <= 60
                  else f"'{edit['find']}' -> '{edit['replace']}'")
        expected_text = f"{edit['target']}[{expect['line']}] {change}" + (
            " (or refuse)" if expect.get("refuse_acceptable") else "")
    return {
        "id": case["id"],
        "why": case["why"],
        "expected": expected_text,
        "actual": result,
        "actual_changes": describe(actual),
        "records": len(outcome["records"]),
        "edits_pass": passed,
        "reporting_pass": not (result == "phantom record" or (actual and not outcome["records"])),
    }


def check_observation(obs):
    out = json.loads((ROOT / obs["output"]).read_text(encoding="utf-8"))
    src = json.loads((ROOT / obs["source"]["file"]).read_text(encoding="utf-8"))
    review = src["reviews"][obs["source"]["review_index"]]["text"]
    failures = []
    if not any(m["source_review"]["text"] == review for m in out["modifications_applied"]):
        failures.append("output has no modification from this review")
    for quote in obs["tweaks_in_review"].values():
        if quote not in review:
            failures.append(f"label quote not in review: {quote}")
    for c in obs["checks"]:
        if c["kind"] == "ingredient_present" and c["text"] not in out["ingredients"]:
            failures.append(f"missing ingredient: {c['text']}")
        elif c["kind"] == "ingredient_absent_substring" and any(c["text"] in x for x in out["ingredients"]):
            failures.append(f"unexpected ingredient containing: {c['text']}")
        elif c["kind"] == "instructions_unchanged" and out["instructions"] != src["instructions"]:
            failures.append("instructions changed")
        elif c["kind"] == "words_not_in_review":
            words = set(re.findall(r"[\w.]+", review.lower()))
            present = [w for w in c["words"] if w.lower() in words]
            if present:
                failures.append(f"words do appear in review: {present}")
    return {
        "id": obs["id"],
        "output": obs["output"],
        "code_version": obs["code_version"],
        "captured": f"{len(obs['captured'])}/{len(obs['tweaks_in_review'])}",
        "missed": sorted(set(obs["tweaks_in_review"]) - set(obs["captured"])),
        "defects": obs["defects"],
        "observation_holds": not failures,
        "check_failures": failures,
    }


# --- Report ----------------------------------------------------------------------


def render_markdown(results):
    lines = [
        f"# Baseline results: `{results['implementation']['label']}`",
        "",
        f"Implementation: `{results['implementation']['path']}` "
        f"(fingerprint `{results['implementation']['fingerprint']}`)",
        "",
        "Generated by `baseline/run_baseline.py` with an oracle LLM (labelled ideal extraction); "
        "see baseline/README.md.",
        "",
    ]
    for fx in results["fixtures"]:
        lines += [f"## {fx['id']}", "", fx["description"], ""]
        if "edit_cases" in fx:
            lines += ["| Case | Target | Expected | Actual | Edit OK | Report OK |", "|---|---|---|---|---|---|"]
            for c in fx["edit_cases"]:
                actual = c["actual"] + (f": {'; '.join(c['actual_changes'])}" if c["actual"] == "wrong edit" else "")
                lines.append(
                    f"| {c['id']} | {c['why']} | {c['expected']} | {actual} | "
                    f"{'yes' if c['edits_pass'] else '**no**'} | {'yes' if c['reporting_pass'] else '**no**'} |"
                )
            lines.append("")
            continue
        for run in fx["runs"]:
            s = run["scores"]
            ex, ed, rp, e2e = s["extraction"], s["edits"], s["reporting"], s["end_to_end"]
            lines += [
                f"### Run: {run['selection']}",
                "",
                "| Dimension | Check | Expected | Actual |",
                "|---|---|---|---|",
                f"| Extraction | reviews analysed | all | {ex['reviews_analysed']} |",
                f"| Extraction | tweaks captured | all | {ex['tweaks_captured']} |",
                f"| Extraction | tried tweaks captured | all | {ex['tried_tweaks_captured']} |",
                f"| Extraction | captured tweaks with their own record | all | {ex['captured_as_own_record']} |",
                f"| Extraction | ineligible tweaks passed to modifier | none | {fmt(ex['ineligible_passed_to_modifier'])} |",
                f"| Edits | line changes from the edits given | {ed['expected_line_changes']} | {ed['correct']} correct |",
                f"| Edits | missing changes | none | {fmt(ed['missing'])} |",
                f"| Edits | unexpected changes | none | {fmt(ed['unexpected'])} |",
                f"| Reporting | output written | yes | {'yes' if rp['output_written'] else 'no'} |",
            ]
            if rp["output_written"]:
                lines += [
                    f"| Reporting | no-op change records | 0 | {rp['noop_records']} |",
                    f"| Reporting | claimed but not in text | none | {fmt(rp['claimed_but_not_in_text'])} |",
                    f"| Reporting | in text but not recorded | none | {fmt(rp['in_text_but_not_recorded'])} |",
                    f"| Reporting | title claims enhancement | {rp['text_actually_changed']} | {rp['title_claims_enhancement']} |",
                    f"| Reporting | tweaks accounted for (applied or skipped) | all | {rp['tweaks_accounted_for']} |",
                    f"| Reporting | records describing no tweak | 0 | {rp['tweak_records_with_no_tweak']} |",
                    f"| Reporting | tweaks with wrong type label | none | {fmt(rp['type_mislabelled'])} |",
                    f"| Reporting | records with review pointer / evidence quote | all | "
                    f"{rp['records_with_review_pointer']} / {rp['records_with_evidence_quote']} |",
                ]
            lines += [
                f"| End to end | matches policy recipe | yes | {'yes' if e2e['recipe_exact'] else 'no'} |",
                "",
            ]
            if ex["missed"]:
                lines += ["Missed tweaks: " + "; ".join(f"{k} ({v})" for k, v in ex["missed"].items()), ""]
            if run["warnings"]:
                lines += ["Implementation warnings: " + "; ".join(f"`{w}`" for w in run["warnings"]), ""]
    lines += ["## Observed real-LLM outputs (committed files)", "",
              "| Obs | Output | Code version | Tweaks captured | Missed | Defects | Still holds |",
              "|---|---|---|---|---|---|---|"]
    for o in results["observed"]:
        lines.append(
            f"| {o['id']} | `{o['output']}` | {o['code_version']} | {o['captured']} | "
            f"{', '.join(o['missed']) or '-'} | {'; '.join(o['defects']) or '-'} | {'yes' if o['observation_holds'] else 'NO'} |"
        )
    lines += ["", "## Summary", "", "Ranges show worst to best over the runs (one per possible random review choice).", ""]
    rows = results["summary"]["per_fixture"]
    keys = list(rows[0])
    lines += ["| Metric | " + " | ".join(r["fixture"] for r in rows) + " |",
              "|---|" + "---|" * len(rows)]
    for k in keys[1:]:
        lines.append(f"| {k} | " + " | ".join(str(r[k]) for r in rows) + " |")
    ec = results["summary"]["edit_cases"]
    lines += ["", f"Edit-level cases (F4): {ec['passed']} passed; wrong edits: {', '.join(ec['wrong edits']) or 'none'}; "
              f"phantom or unrecorded changes: {', '.join(ec['phantom or unrecorded changes']) or 'none'}.", ""]
    return "\n".join(lines) + "\n"


def short(text, limit=70):
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fmt(value):
    if not value:
        return "none"
    if isinstance(value, dict):
        return "; ".join(f"{k} ({v})" for k, v in value.items())
    return "; ".join(short(v) for v in value)


def summarise(fixtures):
    """Per-fixture rows; where a fixture has several runs, show the worst-best range."""

    def span(values):
        values = sorted(values, key=lambda v: (int(v.split("/")[0]), v))
        return values[0] if values[0] == values[-1] else f"{values[0]} to {values[-1]}"

    def count(values):
        values = sorted(values)
        return str(values[0]) if values[0] == values[-1] else f"{values[0]} to {values[-1]}"

    rows = []
    for fx in fixtures:
        if "runs" not in fx:
            continue
        s = [r["scores"] for r in fx["runs"]]
        rows.append({
            "fixture": fx["id"],
            "runs": len(s),
            "distinct recipes across runs": fx["distinct_outputs_across_runs"],
            "reviews analysed": span([x["extraction"]["reviews_analysed"] for x in s]),
            "tweaks captured": span([x["extraction"]["tweaks_captured"] for x in s]),
            "captured with own record": span([x["extraction"]["captured_as_own_record"] for x in s]),
            "ineligible tweaks passed on": count([len(x["extraction"]["ineligible_passed_to_modifier"]) for x in s]),
            "edit line changes correct": span([f"{x['edits']['correct']}/{x['edits']['expected_line_changes']}" for x in s]),
            "unexpected line changes": count([len(x["edits"]["unexpected"]) for x in s]),
            "title misstates change": count([int(not x["reporting"]["title_honest"]) for x in s]),
            "tweaks accounted for": span([x["reporting"]["tweaks_accounted_for"] for x in s]),
            "type-mislabelled tweaks": count([len(x["reporting"].get("type_mislabelled", [])) for x in s]),
            "policy recipe matched": count([int(x["end_to_end"]["recipe_exact"]) for x in s]),
        })
    cases = [c for fx in fixtures for c in fx.get("edit_cases", [])]
    return {
        "per_fixture": rows,
        "edit_cases": {
            "passed": f"{sum(c['edits_pass'] for c in cases)}/{len(cases)}",
            "wrong edits": [c["id"] for c in cases if c["actual"] == "wrong edit"],
            "phantom or unrecorded changes": [c["id"] for c in cases if not c["reporting_pass"]],
        },
    }


# --- Main ------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", default="baseline/reference/original",
                        help="directory containing the llm_pipeline package to test")
    parser.add_argument("--label", default="original", help="name for the results files")
    parser.add_argument("--results-dir", default=str(RESULTS), help="where to write results")
    parser.add_argument("--fixtures", default="F*.json",
                        help="fixture glob under baseline/fixtures (held-out set: heldout/H*.json)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    fixtures = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(FIXTURES.glob(args.fixtures))]
    problems = {fx["id"]: check_labels(fx) for fx in fixtures}
    if any(problems.values()):
        print(json.dumps(problems, indent=2))
        raise SystemExit("Label self-check failed; the baseline would be invalid.")

    pkg, impl = load_impl(args.impl)
    from loguru import logger

    logger.remove()  # the implementation's own logs are captured per run instead
    adapter = adapter_for(pkg)
    # Line endings are normalized so a Windows (CRLF) checkout fingerprints the same code identically
    fingerprint = hashlib.sha256(
        b"".join(p.read_bytes().replace(b"\r\n", b"\n") for p in sorted((impl / "llm_pipeline").glob("*.py")))
    ).hexdigest()[:16]

    results = {
        "implementation": {"label": args.label, "path": args.impl, "fingerprint": fingerprint},
        "fixtures": [],
    }
    for fx in fixtures:
        entry = {"id": fx["id"], "covers": fx["covers"], "description": fx["description"]}
        if "edit_cases" in fx:
            entry["edit_cases"] = [
                score_edit_case(case, fx["recipes"][case["recipe"]],
                                adapter.run_edit_case(fx["recipes"][case["recipe"]], case["edit"]))
                for case in fx["edit_cases"]
            ]
        else:
            entry["runs"] = []
            for run in adapter.run_fixture(fx):
                run["scores"] = score_run(fx, run)
                entry["runs"].append(run)
            entry["distinct_outputs_across_runs"] = len({
                json.dumps(normalized(r["output"] if r["output"] else fx["recipe"]))
                for r in entry["runs"]
            })
        results["fixtures"].append(entry)

    observed = json.loads((FIXTURES / "observed_llm.json").read_text(encoding="utf-8"))
    results["observed"] = [check_observation(o) for o in observed["observations"]]
    results["summary"] = summarise(results["fixtures"])

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{args.label}.json"
    md_path = results_dir / f"{args.label}.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")

    print(md_path.read_text(encoding="utf-8").split("## Summary", 1)[1])
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
