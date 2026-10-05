"""End-to-end workflow: one complete recipe, explicit failures, verified diffs."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from llm_pipeline.models import EnhancedRecipe, EditOutcome, Recipe, TweakOutcome
from llm_pipeline.pipeline import LLMAnalysisPipeline
from llm_pipeline.replay import ReplayClient
from llm_pipeline.report import attribute_diff, compute_diff
from llm_pipeline.tweak_extractor import LLMConfigurationError

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RECORDINGS = ROOT / "baseline" / "recordings"
COOKIES = DATA / "recipe_10813_best-chocolate-chip-cookies.json"

EXPECTED_INGREDIENTS = [
    "1 cup butter, softened", "0.5 cup white sugar", "1.5 cups packed brown sugar", "2 eggs", "1 egg yolk",
    # Both additions follow ing-7 in application order: reviews[1] (rank 2) before reviews[6] (rank 3)
    "2 teaspoons vanilla extract", "1 teaspoon baking soda", "1 teaspoon salt", "1 teaspoon cream of tartar",
    "1 dash cinnamon", "2.5 cups all-purpose flour", "2 cups semisweet chocolate chips",
]
EXPECTED_INSTRUCTIONS = [
    "Gather your ingredients, making sure your butter is softened, and your eggs are room temperature.",
    "Preheat the oven to 350 degrees F (175 degrees C). Beat butter, white sugar, and brown sugar with an "
    "electric mixer in a large bowl until smooth.",
    "Beat in eggs and the egg yolk, one at a time, then stir in vanilla.",
    "Add baking soda to batter along with salt and cream of tartar.",
    "Stir in flour, cinnamon, and chocolate chips.",
    "Cover and refrigerate the dough for at least 1 hour.",
    "Drop scoops of dough (an ice cream scoop makes 16 big cookies) 2 inches apart onto ungreased baking "
    "sheets and press them down slightly.",
    "Bake in the preheated oven until edges are nicely browned, about 10 minutes.",
    "Cool on the baking sheets briefly before removing to a wire rack to cool completely.",
    "Store in an airtight container or serve immediately and enjoy!",
]


def replay_pipeline(tmp_path, *names):
    client = ReplayClient(*(RECORDINGS / n for n in names))
    return LLMAnalysisPipeline(output_dir=str(tmp_path), llm_client=client)


@pytest.fixture(scope="module")
def cookie_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("cookies")
    return replay_pipeline(out, "recipe_10813.json").run_recipe(str(COOKIES))


# --- One complete recipe -----------------------------------------------------------

def test_cookie_recipe_end_to_end_exact_output(cookie_run):
    r = cookie_run.enhanced
    assert (cookie_run.status, cookie_run.reason) == ("enhanced", None)
    assert r.ingredients == EXPECTED_INGREDIENTS
    assert r.instructions == EXPECTED_INSTRUCTIONS
    assert r.title == "Best Chocolate Chip Cookies (Community Enhanced)"


def test_every_review_and_tweak_is_accounted_for(cookie_run):
    r = cookie_run.enhanced
    assert len(r.extraction_results) == 9 and len(r.reviews) == 9
    assert sorted(x.status for x in r.extraction_results) == ["empty"] * 5 + ["ok"] * 4
    tweaks = [t for x in r.extraction_results for t in x.tweaks]
    assert len(tweaks) == 12 and len(r.tweak_outcomes) == 12
    statuses = sorted(o.status for o in r.tweak_outcomes)
    assert statuses == ["applied"] * 11 + ["refused"]
    assert len(r.modifications_applied) == 11


def test_conflict_is_resolved_by_rank_and_explained(cookie_run):
    r = cookie_run.enhanced
    rank = {x.review_id: x.rank for x in r.reviews}
    [refused] = [o for o in r.tweak_outcomes if o.status == "refused"]
    [edit] = refused.edit_outcomes
    assert edit.reason == "conflict" and edit.line_id == "ing-2"
    winner = edit.detail.split("changed by ")[1]
    assert rank[winner.split(".")[0]] < rank[refused.tweak_id.split(".")[0]]


def test_diff_is_verified_and_fully_attributed(cookie_run):
    r = cookie_run.enhanced
    assert r.diff_verified is True and r.diff_problems == []
    assert len(r.recipe_diff) == 14
    assert all(d.tweak_ids for d in r.recipe_diff)
    applied = {o.tweak_id for o in r.tweak_outcomes if o.status == "applied"}
    assert {t for d in r.recipe_diff for t in d.tweak_ids} == applied
    kinds = {(d.kind, d.line_id, d.after) for d in r.recipe_diff if d.kind != "changed"}
    assert kinds == {
        ("added", "ing-3", "1 egg yolk"), ("removed", "ing-6", None), ("added", "ing-7", "1 dash cinnamon"),
        ("added", "ing-7", "1 teaspoon cream of tartar"), ("removed", "ing-10", None),
        ("added", "step-4", "Cover and refrigerate the dough for at least 1 hour."),
    }


def test_attribution_points_back_to_the_source_file(cookie_run):
    r = cookie_run.enhanced
    data = json.loads(COOKIES.read_text(encoding="utf-8"))
    for applied in r.modifications_applied:
        src = applied.source_review
        index = int(src.sources[0][len("reviews["):-1])
        assert data["reviews"][index]["text"] == src.text
        assert applied.evidence_quote in src.text
        assert applied.tweak_id.startswith(src.review_id + ".t")


def test_json_and_markdown_reports_are_written_and_consistent(cookie_run):
    json_path, md_path = map(Path, cookie_run.outputs)
    saved = EnhancedRecipe.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert saved.model_dump(exclude={"created_at"}) == cookie_run.enhanced.model_dump(exclude={"created_at"})
    md = md_path.read_text(encoding="utf-8")
    assert "Extraction answers: **replay of recorded responses (recipe_10813.json)**" in md
    for heading in ("## What changed", "## Applied tweaks", "## Not applied", "## Extraction", "## Recipe"):
        assert heading in md
    assert "verified against the applied edits" in md
    assert "conflict: ing-2 was already changed by" in md
    assert md.count("| ingredients |") + md.count("| instructions |") == 14


def test_run_is_reproducible(tmp_path, cookie_run):
    again = replay_pipeline(tmp_path, "recipe_10813.json").run_recipe(str(COOKIES))
    exclude = {"created_at"}
    assert again.enhanced.model_dump(exclude=exclude) == cookie_run.enhanced.model_dump(exclude=exclude)


# --- Explicit failures ---------------------------------------------------------------

def test_missing_api_key_fails_before_any_call(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("llm_pipeline.pipeline.load_dotenv", lambda: None)
    with pytest.raises(LLMConfigurationError, match="No OpenAI API key"):
        LLMAnalysisPipeline(output_dir=str(tmp_path))


def test_built_client_has_bounded_timeout_and_retries(tmp_path):
    p = LLMAnalysisPipeline(openai_api_key="sk-test", output_dir=str(tmp_path), timeout=12.5, max_retries=0)
    assert p.tweak_extractor.client.timeout == 12.5 and p.tweak_extractor.client.max_retries == 0


def test_model_timeouts_end_in_an_explicit_failure(tmp_path):
    def create(**kwargs):
        raise TimeoutError("request timed out after 30s")

    p = LLMAnalysisPipeline(output_dir=str(tmp_path), llm_client=NS(chat=NS(completions=NS(create=create))))
    run = p.run_recipe(str(COOKIES))
    assert (run.status, run.reason) == ("failed", "extraction_failed")
    assert "request timed out after 30s" in run.detail and run.outputs == []


def test_replay_gaps_are_recorded_as_failed_extractions(tmp_path):
    soup = DATA / "recipe_77935_creamy-sweet-potato-with-ginger-soup.json"
    run = replay_pipeline(tmp_path, "recipe_77935.json").run_recipe(str(soup))
    assert run.status == "enhanced"
    failed = [x for x in run.enhanced.extraction_results if x.status == "failed"]
    assert len(failed) == 4
    assert all("no recorded response" in x.errors[0] for x in failed)
    assert "Extraction failed for" in Path(run.outputs[1]).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "data, status, reason",
    [
        ({"recipe_id": "1", "title": "T", "ingredients": [], "instructions": ["x"]}, "failed", "invalid_input"),
        ({"recipe_id": "1", "title": "T", "ingredients": ["1 egg"], "instructions": ["Mix."]}, "skipped", "no_reviews"),
    ],
)
def test_input_problems_are_explicit(tmp_path, data, status, reason):
    path = tmp_path / "recipe_x.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    run = replay_pipeline(tmp_path, "recipe_10813.json").run_recipe(str(path))
    assert (run.status, run.reason) == (status, reason)


def test_unexpected_errors_are_reported_not_swallowed(tmp_path, monkeypatch):
    p = replay_pipeline(tmp_path, "recipe_10813.json")
    monkeypatch.setattr(p.recipe_modifier, "apply_tweaks", lambda *a: (_ for _ in ()).throw(KeyError("boom")))
    run = p.run_recipe(str(COOKIES))
    assert (run.status, run.reason) == ("failed", "error") and "KeyError" in run.detail


# --- Diff accuracy -------------------------------------------------------------------

def R(ingredients, instructions=("Mix.",)):
    return Recipe(recipe_id="1", title="T", ingredients=list(ingredients), instructions=list(instructions))


def test_diff_pairs_lines_by_similarity_not_position():
    # Regression: positional pairing reported "hot water -> 1 teaspoon salt"
    before = R(["2 teaspoons hot water", "0.5 teaspoon salt", "3 cups all-purpose flour"])
    after = R(["1 teaspoon salt", "1 dash cinnamon", "1 teaspoon cream of tartar", "2.5 cups all-purpose flour"])
    diff = [(d.kind, d.line_id, d.before, d.after) for d in compute_diff(before, after)]
    assert diff == [
        ("removed", "ing-0", "2 teaspoons hot water", None),
        ("changed", "ing-1", "0.5 teaspoon salt", "1 teaspoon salt"),
        ("added", "ing-1", None, "1 dash cinnamon"),
        ("added", "ing-1", None, "1 teaspoon cream of tartar"),
        ("changed", "ing-2", "3 cups all-purpose flour", "2.5 cups all-purpose flour"),
    ]


def test_diff_that_does_not_match_the_outcomes_is_flagged():
    before, after = R(["1 cup flour"]), R(["2 cups flour"])
    claimed = TweakOutcome(tweak_id="rev-0123456789.t1", status="applied", edit_outcomes=[EditOutcome(
        edit_id="rev-0123456789.t1.e1", status="applied", line_id="ing-0", before="1 cup flour", after="3 cups flour")])
    _, problems = attribute_diff(compute_diff(before, after), [claimed])
    assert problems == ["ingredients ing-0: the text change is not what the applied edits produce"]
    _, problems = attribute_diff(compute_diff(before, after), [])
    assert any("not explained by any applied tweak" in p for p in problems)


def test_long_lines_pair_despite_difflib_autojunk():
    # Regression: on 200+ character lines difflib's default autojunk under-scored
    # similarity, so a one-word change showed as a removed line plus an added line
    long_step = ("Heat oil over medium-high heat in a large, deep saute pan until shimmering. Add sweet potatoes, "
                 "then onion; saute, stirring very little at first, then more frequently, until vegetables start "
                 "to turn golden brown, 7 to 8 minutes.")
    before, after = R(["x"], [long_step]), R(["x"], [long_step.replace("sweet potatoes", "potatoes")])
    assert [d.kind for d in compute_diff(before, after)] == ["changed"]


def test_dissimilar_replacement_is_one_changed_line_only_when_an_edit_explains_it():
    before, after = R(["1.5 cups half-and-half (or whole milk)"]), R(["1.5 cups 2% milk"])
    assert sorted(d.kind for d in compute_diff(before, after)) == ["added", "removed"]
    swap = TweakOutcome(tweak_id="rev-0123456789.t1", status="applied", edit_outcomes=[EditOutcome(
        edit_id="rev-0123456789.t1.e1", status="applied", line_id="ing-0",
        before="1.5 cups half-and-half (or whole milk)", after="1.5 cups 2% milk")])
    diffs, problems = attribute_diff(compute_diff(before, after), [swap])
    assert [(d.kind, d.line_id, d.after, d.tweak_ids) for d in diffs] == [
        ("changed", "ing-0", "1.5 cups 2% milk", ["rev-0123456789.t1"])]
    assert problems == []
    _, problems = attribute_diff(compute_diff(before, after), [])
    assert len(problems) == 2  # without the edit, both halves stay unexplained


def test_soup_run_matches_the_labelled_conflict_fixture(tmp_path):
    soup = DATA / "recipe_77935_creamy-sweet-potato-with-ginger-soup.json"
    run = replay_pipeline(tmp_path, "recipe_77935.json").run_recipe(str(soup))
    fx = json.loads((ROOT / "baseline" / "fixtures" / "F3_conflicting_reviews.json").read_text(encoding="utf-8"))
    assert run.enhanced.ingredients == fx["expected"]["recipe"]["ingredients"]
    assert run.enhanced.instructions == fx["expected"]["recipe"]["instructions"]
    assert run.enhanced.diff_verified is True and len(run.enhanced.recipe_diff) == 8
