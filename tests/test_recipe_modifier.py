"""Regression tests for predictable recipe edits."""

import json
from pathlib import Path

import pytest

from llm_pipeline.loader import load_recipe_file, normalize_text, parse_recipe, parse_reviews
from llm_pipeline.models import Tweak, TweakEdit
from llm_pipeline.pipeline import is_applicable
from llm_pipeline.recipe_modifier import RecipeModifier, change_records

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "baseline" / "fixtures"
COOKIES = load_recipe_file(str(ROOT / "data" / "recipe_10813_best-chocolate-chip-cookies.json")).recipe
PREFIX = {"ingredients": "ing", "instructions": "step"}
A, B = "rev-aaaaaaaaaa", "rev-bbbbbbbbbb"


def T(review, n, *edits):
    tid = f"{review}.t{n}"
    return Tweak(
        id=tid, review_id=review, modification_type="quantity_adjustment", reasoning="r",
        status="tried", reviewer_outcome="positive",
        edits=[TweakEdit(id=f"{tid}.e{i}", tweak_id=tid, **e) for i, e in enumerate(edits, start=1)],
    )


def R(line_id, find, replace):
    target = "ingredients" if line_id.startswith("ing") else "instructions"
    return dict(target=target, operation="replace", line_id=line_id, find=find, replace=replace)


def ADD(line_id, find, add):
    target = "ingredients" if line_id.startswith("ing") else "instructions"
    return dict(target=target, operation="add_after", line_id=line_id, find=find, add=add)


def RM(line_id, find):
    target = "ingredients" if line_id.startswith("ing") else "instructions"
    return dict(target=target, operation="remove", line_id=line_id, find=find)


def apply(*tweaks, order=None, recipe=COOKIES):
    return RecipeModifier().apply_tweaks(recipe, list(tweaks), order or {A: 1, B: 2})


def reasons(outcome):
    return [(o.status, o.reason) for o in outcome.edit_outcomes]


# --- Verification before editing -----------------------------------------------------

@pytest.mark.parametrize(
    "edit, reason",
    [
        (R("ing-99", "salt", "sea salt"), "invalid"),                     # no such line
        (R("ing-6", "salt", "sea salt"), "not_found"),                    # text is on ing-7
        (R("step-1", "sugar", "honey"), "ambiguous"),                     # twice on the line
        (RM("step-4", "walnuts"), "invalid"),                             # partial remove
        (R("ing-3", "2 eggs", "2 eggs"), "no_op"),
    ],
)
def test_edits_are_verified_against_the_named_line(edit, reason):
    result, [outcome] = apply(T(A, 1, edit))
    assert outcome.status == "refused" and reasons(outcome) == [("refused", reason)]
    assert result.ingredients == COOKIES.ingredients and result.instructions == COOKIES.instructions


def test_line_id_resolves_text_that_appears_on_several_lines():
    recipe = load_recipe_file(str(ROOT / "data" / "recipe_19117_spicy-apple-cake.json")).recipe
    result, [outcome] = apply(T(A, 1, R("ing-14", "shortening", "butter")), recipe=recipe)
    assert outcome.status == "applied"
    assert (result.ingredients[7], result.ingredients[14]) == ("0.5 cup shortening", "0.25 cup butter")


# --- Stale targets, conflicts and review order ----------------------------------------

def test_same_review_editing_text_it_already_changed_is_stale():
    first = T(A, 1, R("ing-1", "1 cup white sugar", "0.5 cup white sugar"))
    second = T(A, 2, R("ing-1", "1 cup white sugar", "0.75 cup white sugar"))
    result, outcomes = apply(first, second)
    assert [o.status for o in outcomes] == ["applied", "refused"]
    assert reasons(outcomes[1]) == [("refused", "stale")]
    assert result.ingredients[1] == "0.5 cup white sugar"


def test_editing_a_line_the_same_review_removed_is_stale():
    _, outcomes = apply(T(A, 1, RM("ing-6", "2 teaspoons hot water")),
                        T(A, 2, R("ing-6", "2 teaspoons", "1 teaspoon")))
    assert reasons(outcomes[1]) == [("refused", "stale")]


def test_higher_ranked_review_wins_whatever_the_input_order():
    a = T(A, 1, R("ing-2", "1 cup packed brown sugar", "1.5 cups packed brown sugar"))
    b = T(B, 1, R("ing-2", "1 cup packed brown sugar", "0.5 cup packed brown sugar"))
    for order, winner in (({A: 1, B: 2}, "1.5 cups packed brown sugar"), ({A: 2, B: 1}, "0.5 cup packed brown sugar")):
        result, outcomes = apply(b, a, order=order)
        assert result.ingredients[2] == winner
        assert [o.status for o in outcomes] == ["applied", "refused"]
        assert reasons(outcomes[1]) == [("refused", "conflict")]
        assert outcomes[0].tweak_id in outcomes[1].edit_outcomes[0].detail


def test_another_review_may_not_touch_a_changed_ingredient_even_where_text_remains():
    a = T(A, 1, R("ing-2", "packed brown sugar", "packed dark brown sugar"))
    b = T(B, 1, R("ing-2", "1 cup", "1.25 cups"))
    _, outcomes = apply(a, b)
    assert reasons(outcomes[1]) == [("refused", "conflict")]


def test_another_review_may_edit_a_different_part_of_a_step():
    a = T(A, 1, R("step-1", "350 degrees F", "375 degrees F"))
    b = T(B, 1, R("step-1", "in a large bowl", "in a stand mixer"))
    result, outcomes = apply(a, b)
    assert [o.status for o in outcomes] == ["applied", "applied"]
    assert "375 degrees F" in result.instructions[1] and "in a stand mixer" in result.instructions[1]


@pytest.mark.parametrize(
    "later",
    [
        R("step-6", "about 10 minutes", "about 12 minutes"),                # same span
        RM("step-6", COOKIES.instructions[6]),                              # remove a changed step
    ],
)
def test_another_review_conflicts_on_a_changed_step(later):
    a = T(A, 1, R("step-6", "about 10 minutes", "about 8 minutes"))
    _, outcomes = apply(a, T(B, 1, later))
    assert reasons(outcomes[1]) == [("refused", "conflict")]


def test_adding_after_a_line_another_review_removed_is_a_conflict():
    _, outcomes = apply(T(A, 1, RM("ing-6", "2 teaspoons hot water")),
                        T(B, 1, ADD("ing-6", "2 teaspoons hot water", "1 teaspoon milk")))
    assert reasons(outcomes[1]) == [("refused", "conflict")]


def test_adding_after_a_line_another_review_changed_is_allowed():
    result, outcomes = apply(T(A, 1, R("ing-7", "0.5 teaspoon salt", "1 teaspoon salt")),
                             T(B, 1, ADD("ing-7", "0.5 teaspoon salt", "1 teaspoon cream of tartar")))
    assert [o.status for o in outcomes] == ["applied", "applied"]
    assert result.ingredients[7:9] == ["1 teaspoon salt", "1 teaspoon cream of tartar"]


# --- Duplicate changes ------------------------------------------------------------------

@pytest.mark.parametrize(
    "first, second",
    [
        (ADD("ing-3", "2 eggs", "1 egg yolk"), ADD("ing-3", "2 eggs", "1 Egg yolk")),
        (R("ing-1", "1 cup white sugar", "0.5 cup white sugar"), R("ing-1", "1 cup", "0.5 cup")),
        (RM("ing-10", "1 cup chopped walnuts"), RM("ing-10", "1 cup chopped walnuts")),
    ],
)
def test_the_same_change_from_two_reviews_is_a_duplicate(first, second):
    result, outcomes = apply(T(A, 1, first), T(B, 1, second))
    assert reasons(outcomes[1]) == [("refused", "duplicate")]
    assert outcomes[0].tweak_id in outcomes[1].edit_outcomes[0].detail or second["operation"] == "add_after"
    assert result == apply(T(A, 1, first))[0]


def test_adding_a_line_the_recipe_already_has_is_a_duplicate():
    _, [outcome] = apply(T(A, 1, ADD("ing-3", "2 eggs", "2 eggs")))
    assert reasons(outcome) == [("refused", "duplicate")]


def test_a_repeated_edit_inside_one_tweak_rejects_the_tweak():
    edit = R("ing-1", "1 cup white sugar", "0.5 cup white sugar")
    result, [outcome] = apply(T(A, 1, edit, edit))
    assert reasons(outcome) == [("rolled_back", None), ("refused", "duplicate")]
    assert result.ingredients == COOKIES.ingredients


# --- Atomic tweaks ----------------------------------------------------------------------

def test_a_failing_edit_rolls_back_the_whole_tweak():
    half_valid = T(A, 1, RM("ing-6", "2 teaspoons hot water"),
                   R("step-3", "Dissolve baking soda in cold water", "Add baking soda"))
    later = T(B, 1, R("ing-7", "0.5 teaspoon salt", "1 teaspoon salt"))
    result, outcomes = apply(half_valid, later)
    assert outcomes[0].status == "refused"
    assert reasons(outcomes[0]) == [("rolled_back", None), ("refused", "not_found")]
    assert "2 teaspoons hot water" in result.ingredients  # the valid removal was rolled back
    assert outcomes[1].status == "applied" and result.ingredients[7] == "1 teaspoon salt"


def test_tweak_without_edits_is_skipped_and_original_is_not_mutated():
    before = (list(COOKIES.ingredients), list(COOKIES.instructions))
    result, outcomes = apply(T(A, 1), T(B, 1, R("ing-3", "2 eggs", "3 eggs")))
    assert [(o.status, o.reason) for o in outcomes] == [("skipped", "no_edits"), ("applied", None)]
    assert (COOKIES.ingredients, COOKIES.instructions) == before
    assert (result.prep_time, result.total_time) == (COOKIES.prep_time, COOKIES.total_time)


def test_change_records_describe_exactly_the_applied_edits():
    _, [outcome] = apply(T(A, 1, R("ing-7", "0.5", "1"), ADD("ing-7", "0.5 teaspoon salt", "1 dash cinnamon"),
                           RM("ing-10", "1 cup chopped walnuts")))
    assert [(c.operation, c.from_text, c.to_text) for c in change_records(outcome)] == [
        ("replace", "0.5 teaspoon salt", "1 teaspoon salt"),
        ("add", "", "1 dash cinnamon"),
        ("remove", "1 cup chopped walnuts", ""),
    ]


# --- Exact outputs on the labelled fixtures --------------------------------------------

def fixture_run(name, reverse=False):
    fx = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    recipe = parse_recipe(fx["recipe"])
    reviews = {int(r.sources[0][8:-1]): r for r in parse_reviews(fx["recipe"])}
    tweaks, label_of, count = [], {}, {}
    for t in fx["tweaks"]:
        review = reviews[t["review"]]
        count[review.id] = count.get(review.id, 0) + 1
        tid = f"{review.id}.t{count[review.id]}"
        tweak = Tweak(
            id=tid, review_id=review.id, modification_type=t["type"], reasoning="r",
            evidence_quote=t["quote"], status=t["status"], reviewer_outcome=t["outcome"],
            edits=[TweakEdit(id=f"{tid}.e{i}", tweak_id=tid, target=e["target"], operation=e["operation"],
                             line_id=f"{PREFIX[e['target']]}-{e['line']}", find=e["find"],
                             replace=e.get("replace"), add=e.get("add"))
                   for i, e in enumerate(t["edits"], start=1)],
        )
        label_of[tid] = t["id"]
        if is_applicable(tweak):
            tweaks.append(tweak)
    if reverse:
        tweaks.reverse()
    result, outcomes = RecipeModifier().apply_tweaks(recipe, tweaks, {r.id: r.rank for r in reviews.values()})
    expected = fx["expected"]["recipe"]
    want = {k: [normalize_text(x) for x in (fx["recipe"][k] if expected == "unchanged" or expected[k] == "unchanged"
                                             else expected[k])] for k in ("ingredients", "instructions")}
    return fx, result, {label_of[o.tweak_id]: o for o in outcomes}, want


@pytest.mark.parametrize("name", ["F1_multi_tweak", "F3_conflicting_reviews", "F5_unflagged_reviews"])
@pytest.mark.parametrize("reverse", [False, True])
def test_fixture_output_is_exactly_the_expected_recipe(name, reverse):
    fx, result, outcomes, want = fixture_run(name, reverse)
    assert result.ingredients == want["ingredients"]
    assert result.instructions == want["instructions"]
    applied = sorted(k for k, o in outcomes.items() if o.status == "applied")
    assert applied == sorted(fx["expected"]["applied"])


def test_conflict_in_fixture_is_recorded_against_the_winner():
    _, _, outcomes, _ = fixture_run("F3_conflicting_reviews")
    heavy_cream = outcomes["F3.S13"]
    assert heavy_cream.status == "refused"
    assert {o.reason for o in heavy_cream.edit_outcomes} == {"conflict"}
    assert outcomes["F3.S7"].tweak_id in heavy_cream.edit_outcomes[0].detail


# --- Ingredient-instruction consistency -------------------------------------------------

def mentions(lines, term):
    return [line for line in lines if term.casefold() in line.casefold()]


def test_removed_and_added_ingredients_are_reflected_in_the_steps():
    _, result, _, _ = fixture_run("F1_multi_tweak")
    assert not mentions(result.ingredients, "hot water") and not mentions(result.instructions, "hot water")
    assert mentions(result.ingredients, "cream of tartar") and mentions(result.instructions, "cream of tartar")


def test_substitution_reaches_every_step_and_the_conflicting_one_leaves_no_trace():
    _, result, _, _ = fixture_run("F3_conflicting_reviews")
    text = result.ingredients + result.instructions
    assert not mentions(text, "half-and-half")
    assert len(mentions(result.instructions, "2% milk")) == 2 and mentions(result.ingredients, "2% milk")
    assert not mentions(text, "heavy cream")


def test_a_substitution_whose_step_edit_fails_changes_neither_list():
    soup = load_recipe_file(str(ROOT / "data" / "recipe_77935_creamy-sweet-potato-with-ginger-soup.json")).recipe
    swap = T(A, 1, R("ing-10", "1.5 cups half-and-half (or whole milk)", "1.5 cups heavy cream"),
             R("step-6", "add enough cream", "add enough heavy cream"))  # wrong expected text
    result, [outcome] = apply(swap, recipe=soup)
    assert outcome.status == "refused"
    assert result.ingredients == soup.ingredients and result.instructions == soup.instructions


# --- Edit-level baseline cases (F4), now targeted by line ID ----------------------------

F4 = json.loads((FIXTURES / "F4_ambiguous_targets.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", F4["edit_cases"], ids=[c["id"] for c in F4["edit_cases"]])
def test_f4_edit_cases(case):
    recipe = parse_recipe(F4["recipes"][case["recipe"]])
    e = case["edit"]
    edit = dict(target=e["target"], operation=e["operation"], line_id=f"{PREFIX[e['target']]}-{e['line']}",
                find=e["find"], replace=e.get("replace"), add=e.get("add"))
    result, [outcome] = apply(T(A, 1, edit), recipe=recipe)
    lines = getattr(result, e["target"])
    if case["expect"]["outcome"] == "apply":
        assert outcome.status == "applied" or case["expect"].get("refuse_acceptable")
        if outcome.status == "applied":
            assert lines[case["expect"]["line"]] == normalize_text(case["expect"]["after"])
    else:
        assert outcome.status == "refused"
        assert result.ingredients == recipe.ingredients and result.instructions == recipe.instructions


# --- Several lines added after the same line (held-out finding) --------------------------

def test_lines_added_after_the_same_line_keep_their_order():
    # Regression (held-out H1): each insert went straight after the anchor, so a second
    # addition landed above the first and the lines came out reversed
    result, _ = apply(T(A, 1, ADD("ing-3", "2 eggs", "1 egg yolk"), ADD("ing-3", "2 eggs", "1 tablespoon milk")))
    assert result.ingredients[3:6] == ["2 eggs", "1 egg yolk", "1 tablespoon milk"]


def test_additions_from_several_tweaks_follow_application_order():
    result, _ = apply(T(A, 1, ADD("ing-7", "0.5 teaspoon salt", "1 teaspoon cream of tartar")),
                      T(B, 1, ADD("ing-7", "0.5 teaspoon salt", "1 dash cinnamon")))
    assert result.ingredients[7:10] == ["0.5 teaspoon salt", "1 teaspoon cream of tartar", "1 dash cinnamon"]
