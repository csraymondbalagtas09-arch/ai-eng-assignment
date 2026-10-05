"""Pipeline-level checks of input handling and extraction, with a fake LLM client."""

import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

from llm_pipeline.pipeline import LLMAnalysisPipeline

DATA = Path(__file__).resolve().parent.parent / "data"
COOKIES = DATA / "recipe_10813_best-chocolate-chip-cookies.json"
CAKE = DATA / "recipe_19117_spicy-apple-cake.json"
JAM = DATA / "recipe_284494_spiced-purple-plum-jam.json"

EGG_YOLK = {"modification_type": "addition", "quote": "I did add an additional egg yolk",
            "status": "tried", "reviewer_outcome": "positive", "reasoning": "Keeps the cookies chewy.",
            "edits": [{"target": "ingredients", "operation": "add_after", "line_id": "ing-3",
                       "find": "2 eggs", "replace": None, "add": "1 egg yolk"}]}
MORE_APPLES = {"modification_type": "quantity_adjustment", "status": "suggested", "reviewer_outcome": "unclear",
               "quote": "The next time I make this I will add one more cup of apples", "reasoning": "Plan.",
               "edits": [{"target": "ingredients", "operation": "replace", "line_id": "ing-12",
                          "find": "2 cups apple", "replace": "3 cups apple", "add": None}]}


def pipeline(tmp_path, answers):
    """answers maps a fragment of the review text to the tweaks to return."""
    seen = []

    def create(**kwargs):
        review = re.search(r'"""(.*)"""', kwargs["messages"][1]["content"], re.S).group(1)
        seen.append(review)
        tweaks = next((t for frag, t in answers.items() if frag in review), [])
        return NS(choices=[NS(message=NS(content=json.dumps({"tweaks": tweaks})), finish_reason="stop")])

    p = LLMAnalysisPipeline(openai_api_key="not-used", output_dir=str(tmp_path))
    p.tweak_extractor.client = NS(chat=NS(completions=NS(create=create)))
    return p, seen


def test_every_review_is_extracted_and_recorded(tmp_path):
    p, seen = pipeline(tmp_path, {"ice cream scoop": [EGG_YOLK]})
    result = p.process_single_recipe(str(COOKIES), save_output=False)
    assert len(seen) == 9  # all unique reviews, flagged or not; featured duplicates merged
    assert len(result.extraction_results) == 9
    statuses = sorted(r.status for r in result.extraction_results)
    assert statuses == ["empty"] * 8 + ["ok"]
    assert "1 egg yolk" in result.ingredients


def test_applied_tweak_is_attributed(tmp_path):
    p, _ = pipeline(tmp_path, {"ice cream scoop": [EGG_YOLK]})
    result = p.process_single_recipe(str(COOKIES), save_output=False)
    [applied] = result.modifications_applied
    src = applied.source_review
    assert src.sources == ["reviews[0]", "featured_tweaks[0]"] and src.rank == 1
    assert applied.tweak_id == f"{src.review_id}.t1"
    assert applied.evidence_quote == "I did add an additional egg yolk"
    assert result.source_recipe_hash and (result.prep_time, result.total_time) == ("PT20M", "PT30M")


def test_run_is_deterministic(tmp_path):
    a = pipeline(tmp_path, {"ice cream scoop": [EGG_YOLK]})[0].process_single_recipe(str(COOKIES), save_output=False)
    b = pipeline(tmp_path, {"ice cream scoop": [EGG_YOLK]})[0].process_single_recipe(str(COOKIES), save_output=False)
    assert a.model_dump(exclude={"created_at"}) == b.model_dump(exclude={"created_at"})


def test_suggestions_are_recorded_but_not_applied(tmp_path):
    p, seen = pipeline(tmp_path, {"one more cup of apples": [MORE_APPLES]})
    run = p.run_recipe(str(CAKE), save_output=True)
    assert len(seen) == 2
    assert (run.status, run.reason) == ("unchanged", "nothing_applicable")
    assert run.enhanced.title == "Spicy Apple Cake"  # no enhancement claimed
    assert [(o.status, o.reason) for o in run.enhanced.tweak_outcomes] == [("skipped", "not_tried")]
    assert [Path(o).suffix for o in run.outputs] == [".json", ".md"]  # the record is still written


def test_recipe_without_reviews_makes_no_llm_call(tmp_path):
    p, seen = pipeline(tmp_path, {})
    assert p.process_single_recipe(str(JAM), save_output=False) is None
    assert seen == []


def test_invalid_recipe_file_fails_cleanly(tmp_path):
    bad = tmp_path / "recipe_bad.json"
    bad.write_text(json.dumps({"recipe_id": "1", "title": "T", "ingredients": [], "instructions": ["x"]}),
                   encoding="utf-8")
    p, seen = pipeline(tmp_path, {})
    assert p.process_single_recipe(str(bad), save_output=False) is None
    assert seen == []


def test_change_types_are_listed_in_a_stable_order():
    # list(set(...)) ordering varies with the per-process hash seed; the output must not
    from llm_pipeline.enhanced_recipe_generator import EnhancedRecipeGenerator
    from llm_pipeline.models import ModificationApplied, SourceReview

    def mod(type_):
        return ModificationApplied(source_review=SourceReview(text="t", reviewer=None, rating=5),
                                   modification_type=type_, reasoning="r", changes_made=[])

    types = ["technique_change", "addition", "removal", "quantity_adjustment"]
    summary = EnhancedRecipeGenerator().calculate_enhancement_summary([mod(t) for t in types])
    assert summary.change_types == sorted(types)


SOUP = DATA / "recipe_77935_creamy-sweet-potato-with-ginger-soup.json"


def swap(quote, to, outcome="positive"):
    return {"modification_type": "ingredient_substitution", "quote": quote, "status": "tried",
            "reviewer_outcome": outcome, "reasoning": "As stated.",
            "edits": [{"target": "ingredients", "operation": "replace", "line_id": "ing-10",
                       "find": "1.5 cups half-and-half (or whole milk)", "replace": f"1.5 cups {to}", "add": None},
                      {"target": "instructions", "operation": "replace", "line_id": "step-6",
                       "find": "add enough half-and-half", "replace": f"add enough {to}", "add": None}]}


def test_tweaks_from_all_reviews_apply_in_rank_order_and_every_outcome_is_recorded(tmp_path):
    p, _ = pipeline(tmp_path, {
        "2% milk instead of half-and-half": [swap("substituted 2% milk instead of half-and-half", "2% milk")],
        "heavy cream instead of 1/2 & 1/2": [swap("used heavy cream instead of 1/2 & 1/2", "heavy cream")],
        "Used fresh grated ginger": [{
            "modification_type": "ingredient_substitution", "quote": "Used fresh grated ginger", "status": "tried",
            "reviewer_outcome": "positive", "reasoning": "Hearty.",
            "edits": [{"target": "ingredients", "operation": "replace", "line_id": "ing-6",
                       "find": "ground ginger", "replace": "fresh grated ginger (amount not specified)", "add": None}]}],
        "I will use fresh ginger": [{**swap("when I make it again I will use fresh ginger", "x"), "status": "suggested"}],
    })
    result = p.process_single_recipe(str(SOUP), save_output=False)
    assert result.ingredients[10] == "1.5 cups 2% milk"  # the 5-star review outranks the 4-star one
    assert "fresh grated ginger" in result.ingredients[6]  # a second review's tweak also applies
    outcomes = {o.tweak_id: o for o in result.tweak_outcomes}
    statuses = sorted((o.status, o.reason) for o in outcomes.values())
    assert statuses == [("applied", None), ("applied", None), ("refused", None), ("skipped", "not_tried")]
    refused = next(o for o in outcomes.values() if o.status == "refused")
    assert {e.reason for e in refused.edit_outcomes} == {"conflict"}
    assert len(result.modifications_applied) == 2
