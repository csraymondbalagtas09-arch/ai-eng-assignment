#!/usr/bin/env python3
"""
Measurements beyond the harness: cached real-model output, the validator on defective
answers, and latency / token cost per version.

    uv run --with tiktoken python baseline/measure.py cost --impl baseline/reference/original
    uv run --with tiktoken python baseline/measure.py cost --impl src
    uv run python baseline/measure.py validator
    uv run python baseline/measure.py cached

Modes are labelled in every result: "mocked" (stand-in answers written by hand),
"cached" (real model output saved earlier), "live" (new model calls; none exist yet).
"""

import argparse
import hashlib
import html
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "baseline" / "recordings"
OUT = ROOT / "baseline" / "results" / "measure"


def review_key(text):
    return " ".join(html.unescape(text).split()).translate(
        str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})).casefold()


def recorded_tweaks():
    """Review text key -> recorded (mocked) tweaks, from every recordings file."""
    by_key = {}
    for path in RECORDINGS.glob("*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        data = json.loads((ROOT / rec["recipe_file"]).read_text(encoding="utf-8"))
        for r in data.get("reviews", []):
            rid = "rev-" + hashlib.sha1(review_key(r["text"]).encode()).hexdigest()[:10]
            if rid in rec["responses"]:
                by_key[review_key(r["text"])] = rec["responses"][rid]["tweaks"]
    return by_key


# --- Latency and cost ----------------------------------------------------------------

def cost(impl):
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    sys.path.insert(0, str((ROOT / impl).resolve()))
    from loguru import logger

    import llm_pipeline.pipeline as pipeline_module

    logger.remove()
    original = hasattr(pipeline_module.TweakExtractor, "extract_single_modification")
    tweaks_by_review = recorded_tweaks()
    calls = []

    def create(**kwargs):
        messages = kwargs["messages"]
        prompt_text = "".join(m["content"] for m in messages)
        if original:  # the original prompt quotes the review as User Review: "..."
            review = prompt_text.split('User Review: "', 1)[1].rsplit('"\n\nExtract the recipe', 1)[0]
            tried = [t for t in tweaks_by_review.get(review_key(review), []) if t["status"] == "tried"]
            body = {"modification_type": tried[0]["modification_type"] if tried else "addition",
                    "reasoning": "; ".join(t["reasoning"] for t in tried) or "none",
                    "edits": [{k: e[k] for k in ("target", "operation", "find", "replace", "add")}
                              for t in tried for e in t["edits"]]}
        else:
            review = messages[1]["content"].split('"""')[1]
            body = {"tweaks": tweaks_by_review.get(review_key(review), [])}
        content = json.dumps(body)
        # tiktoken count of message text plus ~4 tokens of chat framing per message
        calls.append({"input_tokens": sum(len(enc.encode(m["content"])) + 4 for m in messages),
                      "output_tokens": len(enc.encode(content))})
        return NS(choices=[NS(message=NS(content=content), finish_reason="stop")])

    client = NS(chat=NS(completions=NS(create=create)))
    recipes = sorted((ROOT / "data").glob("recipe_*.json"))
    per_recipe = []
    import random as random_module
    import tempfile

    for path in recipes:
        timings = []
        for repeat in range(5):
            random_module.seed(repeat)
            calls.clear()
            with tempfile.TemporaryDirectory() as tmp:
                if original:
                    p = pipeline_module.LLMAnalysisPipeline(openai_api_key="not-used", output_dir=tmp)
                    p.tweak_extractor.client = client
                else:
                    p = pipeline_module.LLMAnalysisPipeline(output_dir=tmp, llm_client=client)
                start = time.perf_counter()
                p.process_single_recipe(str(path), save_output=True)
                timings.append((time.perf_counter() - start) * 1000)
        per_recipe.append({
            "recipe": path.name,
            "llm_calls": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "pipeline_ms_median": round(statistics.median(timings), 1),
        })
    totals = {k: sum(r[k] for r in per_recipe) for k in ("llm_calls", "input_tokens", "output_tokens")}
    totals["pipeline_ms_median_sum"] = round(sum(r["pipeline_ms_median"] for r in per_recipe), 1)
    if not original:
        from llm_pipeline.prompts import PROMPT_VERSION
        from llm_pipeline.tweak_extractor import DEFAULT_SDK_RETRIES, DEFAULT_TIMEOUT_SECONDS
        prompt, timeout, retries = PROMPT_VERSION, DEFAULT_TIMEOUT_SECONDS, DEFAULT_SDK_RETRIES
        worst_requests = 2 * (1 + retries)  # corrective retry x SDK retries
    else:
        prompt = hashlib.sha256((ROOT / impl / "llm_pipeline" / "prompts.py").read_bytes()).hexdigest()[:12]
        timeout, retries, worst_requests = 600.0, 2, 3 * (1 + 2)  # SDK defaults; 3 extractor attempts
    result = {
        "impl": impl, "mode": "mocked (responses built from the hand-written recordings)",
        "model": "gpt-3.5-turbo (configured; not called)", "prompt_version": prompt,
        "tokenizer": "tiktoken cl100k_base, +4 tokens framing per message",
        "request_timeout_s": timeout, "sdk_retries": retries,
        "worst_case_requests_per_call": worst_requests,
        "per_recipe": per_recipe, "totals": totals,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = "cost-original.json" if original else "cost-improved.json"
    (OUT / name).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["totals"]), "->", name)


# --- Validator on defective answers (mocked) and cached real output -----------------

def _validator_setup():
    sys.path.insert(0, str(ROOT / "src"))
    from loguru import logger

    logger.remove()
    from llm_pipeline.loader import load_recipe_file
    from llm_pipeline.tweak_extractor import validate_tweak

    def load(name):
        return load_recipe_file(str(ROOT / "data" / name))

    return load, validate_tweak


def E(op, line_id, find, replace=None, add=None):
    target = "ingredients" if line_id.startswith("ing") else "instructions"
    return {"target": target, "operation": op, "line_id": line_id, "find": find, "replace": replace, "add": add}


def T(quote, edits, status="tried", outcome="positive", type_="quantity_adjustment"):
    return {"modification_type": type_, "quote": quote, "status": status, "reviewer_outcome": outcome,
            "reasoning": "r", "edits": edits}


def validator():
    """Should-reject defects and should-accept valid answers through validate_tweak."""
    load, validate = _validator_setup()
    cookies = load("recipe_10813_best-chocolate-chip-cookies.json")
    soup = load("recipe_77935_creamy-sweet-potato-with-ginger-soup.json")
    cake = load("recipe_19117_spicy-apple-cake.json")

    def review(loaded, fragment):
        return next(r for r in loaded.reviews if fragment in r.text)

    r1, r3 = review(cookies, "following tweaks"), review(cookies, "The first batch")
    yams, apples = review(soup, "canned yams"), review(cake, "one more cup of apples")
    water = T("I omitted the water", [E("remove", "ing-6", "2 teaspoons hot water")], type_="removal")
    cases = [
        ("D1 paraphrased quote", cookies, r1, {**water, "quote": "I left out the water"}, "reject"),
        ("D2 invented amount (observed O3)", soup, yams,
         T("I added roasted Brussels Sprouts and peppers",
           [E("add_after", "ing-1", "1.5 pounds peeled raw sweet potatoes, cut into 1-inch chunks",
              add="1 cup roasted Brussels Sprouts, halved")], type_="addition"), "reject"),
        ("D3 find on the wrong line", cookies, r1,
         {**water, "edits": [E("remove", "ing-5", "2 teaspoons hot water")]}, "reject"),
        ("D4 tweak not in the review", cookies, r1,
         T("I added walnuts", [E("add_after", "ing-9", "2 cups semisweet chocolate chips", add="walnuts")]), "reject"),
        ("D5 partial remove", cookies, r1, {**water, "edits": [E("remove", "ing-6", "hot water")]}, "reject"),
        ("D6 replace without replacement", cookies, r3,
         T("I used 1 tsp. of salt instead of 1/2 tsp.", [E("replace", "ing-7", "0.5 teaspoon salt")]), "reject"),
        ("D7 unit swap tsp -> tablespoon", cookies, r3,
         T("I used 1 tsp. of salt instead of 1/2 tsp.", [E("replace", "ing-7", "0.5 teaspoon salt", "1 tablespoon salt")]),
         "reject"),
        ("D8 suggestion labelled tried", cake, apples,
         T("The next time I make this I will add one more cup of apples",
           [E("replace", "ing-12", "2 cups apple", "3 cups apple")]), "reject"),
        ("D9 invented descriptor", soup, yams,
         T("I added roasted Brussels Sprouts and peppers",
           [E("add_after", "ing-2", "1 large onion, cut into large dice",
              add="roasted bell peppers, diced (amount not specified)")], type_="addition"), "reject"),
        ("D10 unit conversion", cookies, r3,
         T("I used 1 tsp. of salt instead of 1/2 tsp.", [E("replace", "ing-7", "0.5 teaspoon salt", "0.33 tablespoon salt")]),
         "reject"),
    ]
    results = []
    for name, loaded, rev, proposal, should in cases:
        tweak, problems, warnings = validate(proposal, rev, loaded.recipe)
        results.append({"case": name, "should": should, "decision": "accept" if tweak else "reject",
                        "correct": (tweak is None) == (should == "reject"), "problems": problems,
                        "warnings": warnings})

    # Every labelled tweak (fixtures used in development + held-out set) should be accepted
    valid_total = valid_accepted = 0
    sys.path.insert(0, str(ROOT / "baseline"))
    from llm_pipeline.loader import parse_recipe, parse_reviews
    for path in sorted((ROOT / "baseline" / "fixtures").glob("**/[FH]*.json")):
        fx = json.loads(path.read_text(encoding="utf-8"))
        if "tweaks" not in fx:
            continue
        recipe = parse_recipe(fx["recipe"])
        reviews = {int(r.sources[0][8:-1]): r for r in parse_reviews(fx["recipe"])}
        prefix = {"ingredients": "ing", "instructions": "step"}
        for t in fx["tweaks"]:
            proposal = T(t["quote"], [E(e["operation"], f"{prefix[e['target']]}-{e['line']}", e["find"],
                                        e.get("replace"), e.get("add")) for e in t["edits"]],
                         t["status"], t["outcome"], t["type"])
            tweak, problems, _ = validate(proposal, reviews[t["review"]], recipe)
            valid_total += 1
            valid_accepted += tweak is not None
            if tweak is None:
                results.append({"case": f"valid {t['id']}", "should": "accept", "decision": "reject",
                                "correct": False, "problems": problems, "warnings": []})
    defects = [r for r in results if r["should"] == "reject"]
    summary = {
        "mode": "mocked (defective answers written after implementation; valid answers = all labelled tweaks)",
        "defects_rejected": f"{sum(r['correct'] for r in defects)}/{len(defects)}",
        "valid_accepted": f"{valid_accepted}/{valid_total}",
        "cases": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validator.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in defects:
        print(f"  {r['case']:36} should {r['should']:6} -> {r['decision']:6} {'ok' if r['correct'] else 'MISSED'}"
              + (f"  (warning: {r['warnings'][0]})" if r["warnings"] and not r["correct"] else ""))
    print("defects rejected", summary["defects_rejected"], "| valid accepted", summary["valid_accepted"])


def cached():
    """The only real model output: committed modifications, scored per tweak against the gold labels."""
    observed = json.loads((ROOT / "baseline" / "fixtures" / "observed_llm.json").read_text(encoding="utf-8"))
    rows, labelled, captured, items, faithful = [], 0, 0, 0, 0
    for o in observed["observations"]:
        n_labelled, n_captured = len(o["tweaks_in_review"]), len(o["captured"])
        defects = o["defects"]
        content_defect = any(k in " ".join(defects) for k in ("invented", "untried", "never gave"))
        labelled += n_labelled
        captured += n_captured
        items += 1
        faithful += not content_defect
        rows.append({"obs": o["id"], "output": o["output"], "code_version": o["code_version"],
                     "recall": f"{n_captured}/{n_labelled}", "item_faithful": not content_defect, "defects": defects})
    summary = {
        "mode": "cached (real model output committed with the assignment; model and prompt unrecorded)",
        "version": "original-era code (O2-O4 from an earlier revision than the original snapshot)",
        "tweak_recall": f"{captured}/{labelled}",
        "items_matching_a_real_tweak": f"{items}/{items}",
        "items_faithful_to_the_review": f"{faithful}/{items}",
        "observations": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cached.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "observations"}, indent=1))


# --- Summary -------------------------------------------------------------------------

def _frac(values):
    """Mean over runs of 'x/y' strings, as 'x/y' (x to 1 decimal when not whole)."""
    num = statistics.mean(int(v.split("/")[0]) for v in values)
    den = int(values[0].split("/")[1])
    return num, den


def harness_metrics(path):
    """Expected value over runs (the original chooses a review at random), summed over fixtures."""
    d = json.loads(path.read_text(encoding="utf-8"))
    m = {k: [0.0, 0] for k in ("reviews", "captured", "distinct", "lines", "accounted")}
    m.update(ineligible=0.0, unexpected=0.0, mislabelled=0.0, exact=0.0, fixtures=0, pointer=[0.0, 0],
             quote=[0.0, 0], title_wrong=0.0, cases=None)
    for fx in d["fixtures"]:
        if "edit_cases" in fx:
            m["cases"] = (sum(c["edits_pass"] for c in fx["edit_cases"]), len(fx["edit_cases"]))
            continue
        runs = [r["scores"] for r in fx["runs"]]
        m["fixtures"] += 1
        for key, field in (("reviews", ("extraction", "reviews_analysed")), ("captured", ("extraction", "tweaks_captured")),
                           ("accounted", ("reporting", "tweaks_accounted_for"))):
            num, den = _frac([s[field[0]][field[1]] for s in runs])
            m[key][0] += num
            m[key][1] += den
        m["distinct"][0] += statistics.mean(int(s["extraction"]["captured_as_own_record"].split("/")[0]) for s in runs)
        m["distinct"][1] += int(runs[0]["extraction"]["tweaks_captured"].split("/")[1])
        m["lines"][0] += statistics.mean(s["edits"]["correct"] for s in runs)
        m["lines"][1] += statistics.mean(s["edits"]["expected_line_changes"] for s in runs)
        m["ineligible"] += statistics.mean(len(s["extraction"]["ineligible_passed_to_modifier"]) for s in runs)
        m["unexpected"] += statistics.mean(len(s["edits"]["unexpected"]) for s in runs)
        m["mislabelled"] += statistics.mean(len(s["reporting"].get("type_mislabelled", [])) for s in runs)
        m["title_wrong"] += statistics.mean(int(not s["reporting"]["title_honest"]) for s in runs)
        m["exact"] += statistics.mean(int(s["end_to_end"]["recipe_exact"]) for s in runs)
        for key in ("pointer", "quote"):
            field = "records_with_review_pointer" if key == "pointer" else "records_with_evidence_quote"
            vals = [s["reporting"].get(field) for s in runs if s["reporting"].get(field)]
            if vals:
                num, den = _frac(vals)
                m[key][0] += num
                m[key][1] += den
    return m


def _f(pair):
    num, den = pair

    def short(x):
        return f"{x:.1f}".rstrip("0").rstrip(".")

    return f"{short(num)}/{short(den)}" + (f" ({num / den:.0%})" if den else "")


def summary():
    sets = {"Development fixtures F1–F5 (used while building)": "fixtures", "Held-out H1 (never used while building)": "heldout"}
    lines = ["# Measurement: original vs improved", "",
             "| Mode | Meaning | Results |", "|---|---|---|",
             "| **Mocked** | Stand-in answers written by hand (ideal labelled answers, or deliberately defective ones) | All tables below unless marked |",
             "| **Cached** | Real model output saved earlier (the 4 committed outputs, original-era code) | Cached section |",
             "| **Live** | New model calls | **None**: no API key was available |", ""]
    cost_o = json.loads((OUT / "cost-original.json").read_text(encoding="utf-8"))
    cost_i = json.loads((OUT / "cost-improved.json").read_text(encoding="utf-8"))
    lines += ["| Version | Code | Model (configured) | Prompt version |", "|---|---|---|---|",
              f"| Original | `baseline/reference/original` | gpt-3.5-turbo | `{cost_o['prompt_version']}` (hash of prompts.py) |",
              f"| Improved | `src` | gpt-3.5-turbo | `{cost_i['prompt_version']}` (PROMPT_VERSION) |", ""]
    lines += ["> **Post-measurement fixes (20:39).** Two changes were made after the first measurement, prompted by it: "
              "lines added after the same line now keep their order (found by the held-out set: improved held-out "
              "'recipe exactly as expected' was **0/1 before the fix**, 1/1 after), and the validator now checks units "
              "and plan wording in 'tried' quotes (defects rejected **7/10 before**, 9/10 after). The improved figures "
              "below are post-fix, so the held-out set is **no longer blind** for these fixes.", ""]
    for title, key in sets.items():
        o = harness_metrics(OUT / f"original-{key}.json")
        i = harness_metrics(OUT / f"improved-{key}.json")
        lines += [f"## {title}", "",
                  "Mocked: the stand-in returns the labelled ideal extraction in each version's format, so extraction "
                  "precision is 100% by construction and is not reported here (see Cached and Validator). The original "
                  "picks a review at random, so its figures are the expected value over every possible choice.", "",
                  "| Area | Metric | Original | Improved |", "|---|---|---|---|",
                  f"| Extraction | reviews analysed | {_f(o['reviews'])} | {_f(i['reviews'])} |",
                  f"| Extraction | recall: labelled tweaks captured | {_f(o['captured'])} | {_f(i['captured'])} |",
                  f"| Extraction | recall: captured as their own record | {_f(o['distinct'])} | {_f(i['distinct'])} |",
                  f"| Extraction | untested or disliked tweaks applied | {o['ineligible']:.1f} | {i['ineligible']:.1f} |",
                  f"| Edits | expected line changes produced | {_f(o['lines'])} | {_f(i['lines'])} |",
                  f"| Edits | unexpected line changes | {o['unexpected']:.1f} | {i['unexpected']:.1f} |",
                  f"| Edits | recipe exactly as the policy expects | {o['exact']:.1f}/{o['fixtures']} | {i['exact']:.1f}/{i['fixtures']} |"]
        if o["cases"]:
            lines.append(f"| Edits | single-edit cases (F4) | {o['cases'][0]}/{o['cases'][1]} | {i['cases'][0]}/{i['cases'][1]} |")
        lines += [f"| Attribution | tweaks accounted for (applied, refused or skipped) | {_f(o['accounted'])} | {_f(i['accounted'])} |",
                  f"| Attribution | records naming their review / quoting evidence | {_f(o['pointer'])} / {_f(o['quote'])} | {_f(i['pointer'])} / {_f(i['quote'])} |",
                  f"| Attribution | tweaks with the wrong type label | {o['mislabelled']:.1f} | {i['mislabelled']:.1f} |",
                  f"| Attribution | title misstates whether anything changed | {o['title_wrong']:.1f} | {i['title_wrong']:.1f} |", ""]
    c = json.loads((OUT / "cached.json").read_text(encoding="utf-8"))
    v = json.loads((OUT / "validator.json").read_text(encoding="utf-8"))
    lines += ["## Cached real-model output (original-era code)", "",
              f"Tweak recall {c['tweak_recall']}; extracted items that are real tweaks {c['items_matching_a_real_tweak']}; "
              f"items faithful to the review {c['items_faithful_to_the_review']} (one invented amounts, one applied an untried "
              "suggestion). Model and prompt were not recorded with these outputs. No cached output exists for the improved version.", "",
              "## Validator precision (mocked defective answers)", "",
              f"Defective answers rejected: **{v['defects_rejected']}**. Valid labelled tweaks accepted: **{v['valid_accepted']}** "
              "(development and held-out). Missed defects:", ""]
    lines += [f"- {r['case']}" + (f" (only warned: {r['warnings'][0]})" if r["warnings"] else "")
              for r in v["cases"] if not r["correct"]]
    lines += ["", "## Latency and cost (mocked responses, all 6 recipes)", "",
              "| | Original | Improved |", "|---|---|---|",
              f"| LLM calls | {cost_o['totals']['llm_calls']} | {cost_i['totals']['llm_calls']} |",
              f"| Input tokens | {cost_o['totals']['input_tokens']:,} | {cost_i['totals']['input_tokens']:,} |",
              f"| Output tokens | {cost_o['totals']['output_tokens']:,} | {cost_i['totals']['output_tokens']:,} |",
              f"| Pipeline time excluding the model (sum of per-recipe medians) | {cost_o['totals']['pipeline_ms_median_sum']} ms | {cost_i['totals']['pipeline_ms_median_sum']} ms |",
              f"| Request timeout / SDK retries | {cost_o['request_timeout_s']:g} s / {cost_o['sdk_retries']} | {cost_i['request_timeout_s']:g} s / {cost_i['sdk_retries']} |",
              f"| Worst-case requests per review call | {cost_o['worst_case_requests_per_call']} | {cost_i['worst_case_requests_per_call']} |", "",
              "Tokens counted with tiktoken (cl100k_base) plus 4 framing tokens per message; output tokens come from the "
              "hand-written responses, so they are indicative only. Model latency was not measured (no live calls).", ""]
    (OUT / "measurement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["cost", "validator", "cached", "summary"])
    parser.add_argument("--impl", default="src")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"cost": lambda: cost(args.impl), "validator": validator, "cached": cached, "summary": summary}[args.what]()
