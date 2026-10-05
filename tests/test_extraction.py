"""Tests for tweak extraction: validation, support checks, retry, empty vs failed."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from llm_pipeline.loader import load_recipe_file, parse_recipe, parse_reviews
from llm_pipeline.models import Review
from llm_pipeline.support import quantities, quote_in_text, unsupported_amounts, unsupported_words
from llm_pipeline.tweak_extractor import TweakExtractor, validate_tweak

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "baseline" / "fixtures"
SOUP_FILE = ROOT / "data" / "recipe_77935_creamy-sweet-potato-with-ginger-soup.json"


def fixture(name):
    fx = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    recipe = parse_recipe(fx["recipe"])
    reviews = {r.sources[0]: r for r in parse_reviews(fx["recipe"])}
    return fx, recipe, [reviews[f"reviews[{i}]"] for i in range(len(fx["recipe"]["reviews"]))]


def as_llm(tweak):
    """A labelled tweak in the JSON format the LLM must return."""
    prefix = {"ingredients": "ing", "instructions": "step"}
    return {
        "modification_type": tweak["type"], "quote": tweak["quote"], "status": tweak["status"],
        "reviewer_outcome": tweak["outcome"], "reasoning": "As stated by the reviewer.",
        "edits": [{"target": e["target"], "operation": e["operation"],
                   "line_id": f"{prefix[e['target']]}-{e['line']}", "find": e["find"],
                   "replace": e.get("replace"), "add": e.get("add")} for e in tweak["edits"]],
    }


def client(*responses):
    """Fake OpenAI client returning each response in turn (dict -> JSON, str as is, Exception raised)."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        r = responses[len(calls) - 1]
        if isinstance(r, Exception):
            raise r
        content = json.dumps(r) if isinstance(r, dict) else r
        return NS(choices=[NS(message=NS(content=content), finish_reason="stop")])

    return NS(chat=NS(completions=NS(create=create))), calls


def extractor(*responses):
    c, calls = client(*responses)
    return TweakExtractor(client=c), calls


# --- Support checks ---------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [("1/2 c less flour", {0.5}), ("one-and-a-half cups", {1.5}), ("a tiny dash", {1}),
     ("1lb of meat", {1}), ("2% milk", {2}), ("8-9 minutes", {8, 9}), ("½ cup", {0.5}),
     ("1 1/2 cups", {1.5}), ("(1) I used (2) I omitted", set()), ("no amounts here", set())],
)
def test_quantities(text, expected):
    assert quantities(text) == expected


@pytest.mark.parametrize(
    "new, old, review, unsupported",
    [
        ("1 cup roasted Brussels Sprouts", "", "I added roasted Brussels Sprouts", [1]),  # observed O3
        ("2.5 cups all-purpose flour", "3 cups all-purpose flour", "1/2 c less flour", []),  # 3 - 0.5
        ("3 cups apple", "2 cups apple", "add one more cup of apples", []),  # 2 + 1
        ("1 teaspoon cream of tartar", "", "I added a teaspoon of cream of tartar", []),
        ("0.33 tablespoon salt", "0.5 teaspoon salt", "I used 1 tsp. of salt", [0.33]),  # conversion
        ("1.5 cups 2% milk", "1.5 cups half-and-half", "I substituted 2% milk", []),
        ("375 degrees F (190 degrees C)", "350 degrees F (175 degrees C)", "baked at 375", [190]),
    ],
)
def test_unsupported_amounts(new, old, review, unsupported):
    assert unsupported_amounts(new, old, review) == unsupported


def test_unsupported_words_flags_invented_detail_not_units_or_filler():
    words = unsupported_words("1 cup roasted bell peppers, diced (amount not specified)", "",
                              "I added roasted Brussels Sprouts and peppers", "")
    assert words == ["bell", "diced"]


# --- The labelled fixtures: supported changes pass, invented ones do not -----------

@pytest.mark.parametrize("name", ["F1_multi_tweak", "F2_untested_suggestions",
                                  "F3_conflicting_reviews", "F5_unflagged_reviews"])
def test_every_labelled_tweak_is_accepted(name):
    fx, recipe, reviews = fixture(name)
    for t in fx["tweaks"]:
        tweak, problems, _ = validate_tweak(as_llm(t), reviews[t["review"]], recipe)
        assert tweak is not None, (t["id"], problems)


def test_observed_invented_amounts_are_rejected():
    soup = load_recipe_file(str(SOUP_FILE))
    review = next(r for r in soup.reviews if "canned yams" in r.text)
    proposed = {  # what the real model produced for this review (committed output O3)
        "modification_type": "addition", "quote": "I added roasted Brussels Sprouts and peppers",
        "status": "tried", "reviewer_outcome": "positive", "reasoning": "Contrast.",
        "edits": [{"target": "ingredients", "operation": "add_after", "line_id": "ing-1",
                   "find": "1.5 pounds peeled raw sweet potatoes, cut into 1-inch chunks",
                   "add": "1 cup roasted Brussels Sprouts, halved; 1 cup roasted bell peppers, diced"}],
    }
    tweak, problems, _ = validate_tweak(proposed, review, soup.recipe)
    assert tweak is None
    assert any("amount 1" in p for p in problems)


# --- Field and line-reference validation --------------------------------------------

@pytest.fixture
def f1():
    fx, recipe, reviews = fixture("F1_multi_tweak")
    return as_llm(fx["tweaks"][1]), reviews[0], recipe  # "I omitted the water"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: t.pop("status"), "status: Field required"),
        (lambda t: t.update(status="maybe"), "status: Input should be 'tried' or 'suggested'"),
        (lambda t: t.update(quote="I left out the water"), "does not appear in the review"),
        (lambda t: t.update(reasoning=" "), "reasoning is empty"),
        (lambda t: t["edits"][0].update(line_id="ing-99"), "'ing-99' does not exist"),
        (lambda t: t["edits"][0].update(line_id="step-3"), "step-3 is not an ingredients line"),
        (lambda t: t["edits"][0].update(line_id="ing-5"), "is not in ing-5 (it is in ing-6)"),
        (lambda t: t["edits"][0].update(find="hot water"), "remove must use the whole line"),
        (lambda t: t["edits"][1].update(replace=None), "replace needs replacement text"),
        (lambda t: t["edits"][1].update(replace="Dissolve baking soda in hot water. Add to batter"),
         "same as the found text"),
        (lambda t: t["edits"].append({"target": "ingredients", "operation": "add_after", "line_id": "ing-7",
                                      "find": "salt", "add": None}), "add_after needs the new line"),
        (lambda t: t["edits"][1].update(replace="Add 2 teaspoons baking soda to batter"),
         "amount 2 in 'Add 2 teaspoons"),
    ],
)
def test_invalid_proposals_are_rejected_with_a_usable_message(f1, mutate, message):
    raw, review, recipe = f1
    mutate(raw)
    tweak, problems, _ = validate_tweak(raw, review, recipe)
    assert tweak is None
    assert any(message in p for p in problems), problems


def test_paraphrased_wording_is_a_warning_not_a_rejection():
    fx, recipe, reviews = fixture("F3_conflicting_reviews")
    s12 = next(t for t in fx["tweaks"] if t["id"] == "F3.S12")
    tweak, problems, warnings = validate_tweak(as_llm(s12), reviews[1], recipe)
    assert tweak is not None and not problems
    assert warnings == ["ing-11: words in neither review nor recipe: plenty"]


# --- Extraction flow: ok, empty, failed, one corrective retry -----------------------

def test_every_tweak_is_returned_with_ids_quotes_status_and_line_targets():
    fx, recipe, reviews = fixture("F1_multi_tweak")
    ex, calls = extractor({"tweaks": [as_llm(t) for t in fx["tweaks"]]})
    result = ex.extract_tweaks(reviews[0], recipe)
    assert (result.status, result.attempts, len(calls)) == ("ok", 1, 1)
    assert [t.evidence_quote for t in result.tweaks] == [t["quote"] for t in fx["tweaks"]]
    assert [t.id for t in result.tweaks] == [f"{reviews[0].id}.t{n}" for n in range(1, 5)]
    assert {t.status for t in result.tweaks} == {"tried"}
    assert result.tweaks[1].edits[0].line_id == "ing-6"
    assert result.tweaks[1].edits[1].id == f"{reviews[0].id}.t2.e2"
    user_prompt = calls[0]["messages"][1]["content"]
    assert "[ing-6] 2 teaspoons hot water" in user_prompt and "[step-3]" in user_prompt


def test_no_tweaks_is_a_valid_empty_result_without_retry():
    fx, recipe, reviews = fixture("F1_multi_tweak")
    ex, calls = extractor({"tweaks": []})
    result = ex.extract_tweaks(Review(text="It is a great recipe", id="rev-0000000000"), recipe)
    assert (result.status, result.tweaks, result.errors, result.attempts) == ("empty", [], [], 1)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "first, second, error",
    [
        (RuntimeError("timeout"), RuntimeError("timeout"), "the API call failed (timeout)"),
        ("not json", "still not json", "not valid JSON"),
        ({"items": []}, {"items": []}, "must be a JSON object with a 'tweaks' list"),
        ("", "", "the response was empty"),
    ],
)
def test_unusable_responses_fail_after_one_retry(first, second, error):
    fx, recipe, reviews = fixture("F1_multi_tweak")
    ex, calls = extractor(first, second)
    result = ex.extract_tweaks(reviews[0], recipe)
    assert result.status == "failed" and result.attempts == 2 and len(calls) == 2
    assert any(error in e for e in result.errors), result.errors


def test_corrective_retry_shows_the_problems_and_accepts_the_fix():
    fx, recipe, reviews = fixture("F1_multi_tweak")
    good = [as_llm(t) for t in fx["tweaks"]]
    broken = json.loads(json.dumps(good))
    broken[2]["edits"][0]["line_id"] = "ing-99"
    ex, calls = extractor({"tweaks": broken}, {"tweaks": good})
    result = ex.extract_tweaks(reviews[0], recipe)
    assert (result.status, result.attempts, len(result.tweaks), result.rejected) == ("ok", 2, 4, [])
    followup = calls[1]["messages"]
    assert followup[2]["role"] == "assistant"
    assert "tweak 3: edit 1: line_id 'ing-99' does not exist" in followup[3]["content"]


def test_only_one_corrective_retry_and_failures_are_kept_as_rejected():
    fx, recipe, reviews = fixture("F1_multi_tweak")
    bad = as_llm(fx["tweaks"][0])
    bad["quote"] = "I used less sugar"
    ex, calls = extractor({"tweaks": [bad]}, {"tweaks": [bad]})
    result = ex.extract_tweaks(reviews[0], recipe)
    assert len(calls) == 2
    assert result.status == "failed" and not result.tweaks
    assert result.rejected[0].quote == "I used less sugar"
    assert "does not appear in the review" in result.rejected[0].problems[0]
    assert result.rejected[0].proposed["quote"] == "I used less sugar"


def test_valid_tweaks_survive_a_retry_that_drops_them():
    fx, recipe, reviews = fixture("F1_multi_tweak")
    good = [as_llm(t) for t in fx["tweaks"]]
    bad = dict(good[3], quote="I chilled it overnight")
    ex, _ = extractor({"tweaks": good[:3] + [bad]}, {"tweaks": [good[3]]})
    result = ex.extract_tweaks(reviews[0], recipe)
    assert result.status == "ok" and len(result.tweaks) == 4 and result.rejected == []


def test_a_tweak_left_out_of_the_correction_is_recorded():
    fx, recipe, reviews = fixture("F1_multi_tweak")
    good = [as_llm(t) for t in fx["tweaks"]]
    bad = dict(good[3], quote="I chilled it overnight")
    ex, _ = extractor({"tweaks": good[:3] + [bad]}, {"tweaks": good[:3]})
    result = ex.extract_tweaks(reviews[0], recipe)
    assert len(result.tweaks) == 3
    assert result.rejected[0].problems[-1] == "left out of the corrected answer"


def test_retry_after_api_error_reasks_the_same_question():
    fx, recipe, reviews = fixture("F1_multi_tweak")
    ex, calls = extractor(RuntimeError("timeout"), {"tweaks": []})
    result = ex.extract_tweaks(reviews[0], recipe)
    assert result.status == "empty" and result.attempts == 2
    assert calls[1]["messages"] == calls[0]["messages"]


def test_duplicate_tweaks_are_rejected():
    fx, recipe, reviews = fixture("F1_multi_tweak")
    t = as_llm(fx["tweaks"][0])
    ex, _ = extractor({"tweaks": [t, t]}, {"tweaks": [t]})
    result = ex.extract_tweaks(reviews[0], recipe)
    assert len(result.tweaks) == 1 and result.status == "ok"


def test_suggestions_are_extracted_and_labelled():
    fx, recipe, reviews = fixture("F2_untested_suggestions")
    ex, _ = extractor({"tweaks": [as_llm(t) for t in fx["tweaks"] if t["review"] == 0]})
    result = ex.extract_tweaks(reviews[0], recipe)
    assert [(t.status, bool(t.edits)) for t in result.tweaks] == [("tried", False), ("suggested", True)]


# --- Validator gaps found by the stress set ---------------------------------------------

from llm_pipeline.support import plan_language, unsupported_units  # noqa: E402


@pytest.mark.parametrize(
    "new, old, review, problems",
    [
        ("1 tablespoon salt", "0.5 teaspoon salt", "I used 1 tsp. of salt instead of 1/2 tsp.", ["1 tbsp"]),
        ("1 teaspoon salt", "0.5 teaspoon salt", "I used 1 tsp. of salt instead of 1/2 tsp.", []),
        ("0.5 cup white sugar", "1 cup white sugar", "I used a half cup of sugar", []),
        ("1.5 cups packed brown sugar", "1 cup packed brown sugar", "one-and-a-half cups of brown sugar", []),
        ("2.5 cups all-purpose flour", "3 cups all-purpose flour", "1/2 c less flour", []),
        ("3 cups apple", "2 cups apple", "I will add one more cup of apples", []),
        ("1 pound sirloin steak", "0.25 pound sirloin steak", "I used a lil over 1lb", []),
        ("1 dash cinnamon", "", "added a tiny dash of cinnamon", []),
        ("2 cups flour", "1 cup flour", "I doubled the flour", []),
        ("375 degrees F (190 degrees C)", "350 degrees F (175 degrees C)", "baked at 375", []),  # not units
    ],
)
def test_unsupported_units(new, old, review, problems):
    assert unsupported_units(new, old, review) == problems


@pytest.mark.parametrize(
    "quote, flagged",
    [("The next time I make this I will add one more cup of apples", True),
     ("I would prefer some more apple chunks in it", True),
     ("when I make it again I'll use fresh ginger", True),
     ("I did add an additional egg yolk", False),
     ("if I pressed them down slightly before baking that worked the best", False),
     ("I used a lil over 1lb", False)],
)
def test_plan_language(quote, flagged):
    assert bool(plan_language(quote)) == flagged


def test_unit_swap_and_plan_quoted_as_tried_are_rejected():
    fx, recipe, reviews = fixture("F1_multi_tweak")
    salt = {"modification_type": "quantity_adjustment", "quote": "I used a half cup of sugar", "status": "tried",
            "reviewer_outcome": "positive", "reasoning": "r",
            "edits": [{"target": "ingredients", "operation": "replace", "line_id": "ing-1",
                       "find": "1 cup white sugar", "replace": "0.5 tablespoon white sugar"}]}
    _, problems, _ = validate_tweak(salt, reviews[0], recipe)
    assert any("0.5 tbsp" in p and "unit" in p for p in problems), problems

    fx2, cake, cake_reviews = fixture("F2_untested_suggestions")
    apples = as_llm(next(t for t in fx2["tweaks"] if t["id"] == "F2.A2"))
    apples["status"] = "tried"
    _, problems, _ = validate_tweak(apples, cake_reviews[0], cake)
    assert any("describes a plan" in p for p in problems), problems
