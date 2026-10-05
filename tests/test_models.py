"""Tests for the tweak, edit and outcome record contracts."""

import pytest
from pydantic import ValidationError

from llm_pipeline.models import (
    EditOutcome,
    ModificationEdit,
    ModificationObject,
    Recipe,
    RecipeLine,
    Review,
    Tweak,
    TweakEdit,
    TweakOutcome,
)

RID = "rev-0123456789"
TID = RID + ".t1"


def edit(n=1, **kw):
    fields = dict(id=f"{TID}.e{n}", tweak_id=TID, target="ingredients", operation="replace",
                  find="1 cup flour", replace="2 cups flour")
    fields.update(kw)
    return TweakEdit(**fields)


def outcome(n=1, status="applied", **kw):
    fields = dict(edit_id=f"{TID}.e{n}", status=status)
    if status == "applied":
        fields.update(before="1 cup flour", after="2 cups flour", line_id="ing-0")
    fields.update(kw)
    return EditOutcome(**fields)


# --- IDs ------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["rev-123", "rev-0123456789x", "REV-0123456789", "0123456789"])
def test_review_id_format(bad):
    Review(text="x", id=RID)
    with pytest.raises(ValidationError):
        Review(text="x", id=bad)


def test_line_id_must_match_its_position():
    RecipeLine(id="step-2", section="instructions", index=2, text="Mix.", raw_text="Mix.")
    with pytest.raises(ValidationError, match="does not match"):
        RecipeLine(id="ing-2", section="instructions", index=2, text="Mix.", raw_text="Mix.")


def test_recipe_lines_must_mirror_recipe_text():
    line = RecipeLine(id="ing-0", section="ingredients", index=0, text="1 cup flour", raw_text="1 cup flour")
    Recipe(recipe_id="1", title="T", ingredients=["1 cup flour"], instructions=[], lines=[line])
    with pytest.raises(ValidationError, match="do not match ingredients"):
        Recipe(recipe_id="1", title="T", ingredients=["2 cups flour"], instructions=[], lines=[line])


# --- Edits ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kw, message",
    [
        (dict(id="rev-0123456789.t2.e1"), "does not belong to tweak"),
        (dict(replace=None), "needs replacement text"),
        (dict(operation="add_after", replace=None, add="  "), "needs text to add"),
        (dict(line_id="step-1"), "is not in ingredients"),
        (dict(find=""), "at least 1 character"),
        (dict(line_id="ingredient-1"), "pattern"),
    ],
)
def test_invalid_edits_are_rejected(kw, message):
    with pytest.raises(ValidationError, match=message):
        edit(**kw)


def test_empty_replacement_is_a_valid_deletion():
    assert edit(replace="").replace == ""


@pytest.mark.parametrize(
    "kw",
    [dict(operation="replace", replace=None), dict(operation="add_after", add=""), dict(find="  ")],
)
def test_llm_edits_with_missing_parts_fail_validation_so_they_are_retried(kw):
    fields = dict(target="ingredients", operation="replace", find="1 cup flour", replace="x")
    fields.update(kw)
    with pytest.raises(ValidationError):
        ModificationEdit(**fields)


# --- Tweaks -----------------------------------------------------------------------

def test_tweak_from_modification_assigns_stable_ids():
    mod = ModificationObject(
        modification_type="quantity_adjustment", reasoning="r",
        edits=[ModificationEdit(target="ingredients", operation="replace", find="a", replace="b"),
               ModificationEdit(target="instructions", operation="remove", find="Mix.")],
    )
    tweak = Tweak.from_modification(mod, RID, 2)
    assert tweak.id == RID + ".t2"
    assert [e.id for e in tweak.edits] == [RID + ".t2.e1", RID + ".t2.e2"]
    assert tweak.status is None and tweak.evidence_quote is None  # not determined yet


def test_tweak_rejects_foreign_or_duplicate_edits():
    with pytest.raises(ValidationError, match="does not belong to review"):
        Tweak(id="rev-9999999999.t1", review_id=RID, modification_type="addition", reasoning="r")
    other = TweakEdit(id=RID + ".t2.e1", tweak_id=RID + ".t2", target="ingredients",
                      operation="remove", find="x")
    with pytest.raises(ValidationError, match="another tweak's edit"):
        Tweak(id=TID, review_id=RID, modification_type="addition", reasoning="r", edits=[other])
    with pytest.raises(ValidationError, match="duplicate edit IDs"):
        Tweak(id=TID, review_id=RID, modification_type="addition", reasoning="r", edits=[edit(), edit()])


# --- Outcomes ---------------------------------------------------------------------

def test_edit_outcome_rules():
    outcome()
    EditOutcome(edit_id=f"{TID}.e1", status="refused", reason="ambiguous", detail="2 lines")
    EditOutcome(edit_id=f"{TID}.e1", status="rolled_back")
    with pytest.raises(ValidationError, match="needs a reason"):
        EditOutcome(edit_id=f"{TID}.e1", status="refused")
    with pytest.raises(ValidationError, match="no refusal reason"):
        EditOutcome(edit_id=f"{TID}.e1", status="rolled_back", reason="conflict")
    with pytest.raises(ValidationError, match="must change the text"):
        outcome(after="1 cup flour")
    with pytest.raises(ValidationError):
        outcome(match_score=1.5)


def test_tweak_outcome_is_all_or_nothing():
    TweakOutcome(tweak_id=TID, status="applied", edit_outcomes=[outcome(1), outcome(2)])
    TweakOutcome(tweak_id=TID, status="refused", edit_outcomes=[
        outcome(1, status="rolled_back"), outcome(2, status="refused", reason="not_found")])
    with pytest.raises(ValidationError, match="every edit applied"):
        TweakOutcome(tweak_id=TID, status="applied", edit_outcomes=[
            outcome(1), outcome(2, status="refused", reason="no_op")])
    with pytest.raises(ValidationError, match="every edit applied"):
        TweakOutcome(tweak_id=TID, status="applied")
    with pytest.raises(ValidationError, match="no applied edits"):
        TweakOutcome(tweak_id=TID, status="refused", edit_outcomes=[
            outcome(1), outcome(2, status="refused", reason="not_found")])


def test_skipped_tweak_needs_reason_and_no_attempts():
    TweakOutcome(tweak_id=TID, status="skipped", reason="not_tried", detail="next time I will…")
    with pytest.raises(ValidationError, match="needs a reason"):
        TweakOutcome(tweak_id=TID, status="skipped")
    with pytest.raises(ValidationError, match="no edit outcomes"):
        TweakOutcome(tweak_id=TID, status="skipped", reason="not_tried", edit_outcomes=[outcome()])
    with pytest.raises(ValidationError, match="only a skipped tweak"):
        TweakOutcome(tweak_id=TID, status="applied", reason="not_tried", edit_outcomes=[outcome()])


def test_outcome_cannot_contain_another_tweaks_edit():
    with pytest.raises(ValidationError, match="another tweak's edit"):
        TweakOutcome(tweak_id=RID + ".t2", status="applied", edit_outcomes=[outcome()])
