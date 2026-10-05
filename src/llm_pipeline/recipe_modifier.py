"""
Step 2: Recipe Modification

This module applies verified tweaks to a recipe. Every result is predictable from
the rules below, and every refusal is recorded with its reason:

- Each edit names a line by ID. The line must exist, and the line as it was in the
  original recipe must contain the edit's find text exactly once; edits are written
  against the original recipe.
- If an earlier tweak has since changed or removed the target, the edit is stale
  (same review) or a conflict (different review). Tweaks are applied in the
  documented review order, so the higher-ranked review wins a conflict.
  A different review may still edit another part of an instruction step, but not
  an ingredient line another review changed, nor remove a line another review changed.
- A change that has already been made is a duplicate: adding a line that is already
  there, a replacement whose result is already on the line, removing a removed line.
- A tweak is applied all-or-nothing. If any edit is refused, the recipe stays as it
  was before the tweak and the tweak's other edits are recorded as rolled back.
"""

import copy
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .models import ChangeRecord, EditOutcome, Recipe, Tweak, TweakEdit, TweakOutcome


class _Refusal(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass
class _Line:
    line_id: Optional[str]  # original line ID; None for a line added by a tweak
    text: str
    original: Optional[str]  # text in the original recipe; None for an added line
    changed_by: Optional[str] = None  # tweak that last changed or added the line
    review_id: Optional[str] = None  # review of that tweak
    anchor: Optional[str] = None  # for an added line: the original line it follows


@dataclass
class _State:
    lines: Dict[str, List[_Line]]  # section -> lines in order
    removed: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # line ID -> (tweak, review)


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _spans(find: str, text: str) -> Tuple[List[Tuple[int, int]], str]:
    """Where find occurs in text: exact matches, else case/space-insensitive ones."""
    exact = [m.span() for m in re.finditer(re.escape(find), text)]
    if exact:
        return exact, "exact"
    loose = r"\s+".join(re.escape(token) for token in find.split())
    return [m.span() for m in re.finditer(loose, text, re.IGNORECASE)], "normalized"


def _replace_once(text: str, span: Tuple[int, int], replacement: str) -> str:
    return " ".join((text[: span[0]] + replacement + text[span[1]:]).split())


def change_records(outcome: TweakOutcome) -> List[ChangeRecord]:
    """The applied edits of a tweak in the original output format."""
    records = []
    for o in outcome.edit_outcomes:
        if o.status != "applied":
            continue
        kind = "ingredient" if (o.line_id or "").startswith("ing-") else "instruction"
        operation = "add" if o.before is None else "remove" if o.after is None else "replace"
        records.append(ChangeRecord(type=kind, from_text=o.before or "", to_text=o.after or "",
                                    operation=operation))
    return records


class RecipeModifier:
    """Applies verified tweaks to recipes, all-or-nothing, in review order."""

    def __init__(self):
        logger.info("Initialized RecipeModifier (line-ID targeting, atomic tweaks)")

    @staticmethod
    def _initial_state(recipe: Recipe) -> _State:
        return _State(lines={
            "ingredients": [_Line(f"ing-{i}", t, t) for i, t in enumerate(recipe.ingredients)],
            "instructions": [_Line(f"step-{i}", t, t) for i, t in enumerate(recipe.instructions)],
        })

    def _apply_edit(self, state: _State, edit: TweakEdit, tweak: Tweak) -> EditOutcome:
        """Verify one edit against the current state and apply it, or raise _Refusal."""
        lines = state.lines[edit.target]
        if edit.line_id is None:
            raise _Refusal("invalid", "the edit does not name a line")
        if edit.line_id in state.removed:
            by_tweak, by_review = state.removed[edit.line_id]
            if edit.operation == "remove":
                raise _Refusal("duplicate", f"{edit.line_id} was already removed by {by_tweak}")
            reason = "stale" if by_review == tweak.review_id else "conflict"
            raise _Refusal(reason, f"{edit.line_id} was removed by {by_tweak}")
        position = next((i for i, l in enumerate(lines) if l.line_id == edit.line_id), None)
        if position is None:
            raise _Refusal("invalid", f"{edit.line_id} does not exist in the recipe")
        line = lines[position]

        # The edit must match the line as it was in the original recipe
        spans, method = _spans(edit.find, line.original)
        if not spans:
            raise _Refusal("not_found", f"'{edit.find}' is not on {edit.line_id}: '{line.original}'")
        if len(spans) > 1:
            raise _Refusal("ambiguous", f"'{edit.find}' occurs {len(spans)} times on {edit.line_id}")

        other_review = line.changed_by is not None and line.review_id != tweak.review_id
        changed = line.text != line.original

        if edit.operation == "add_after":
            if any(_norm(l.text) == _norm(edit.add) for l in lines):
                raise _Refusal("duplicate", f"'{edit.add}' is already in the {edit.target}")
            # After any lines already added after this anchor, so additions keep
            # their order: edit order within a tweak, application order across tweaks
            insert_at = position + 1
            while insert_at < len(lines) and lines[insert_at].anchor == edit.line_id:
                insert_at += 1
            lines.insert(insert_at, _Line(None, edit.add, None, tweak.id, tweak.review_id, anchor=edit.line_id))
            return EditOutcome(edit_id=edit.id, status="applied", line_id=edit.line_id,
                               before=None, after=edit.add, match_method=method, match_score=1.0)

        if edit.operation == "remove":
            if _norm(edit.find) != _norm(line.original):
                raise _Refusal("invalid", f"remove must name the whole line '{line.original}'")
            if other_review:
                raise _Refusal("conflict", f"{edit.line_id} was changed by {line.changed_by}")
            lines.pop(position)
            state.removed[edit.line_id] = (tweak.id, tweak.review_id)
            return EditOutcome(edit_id=edit.id, status="applied", line_id=edit.line_id,
                               before=line.text, after=None, match_method=method, match_score=1.0)

        # replace
        intended = _replace_once(line.original, spans[0], edit.replace)
        if changed:
            if _norm(line.text) == _norm(intended):
                raise _Refusal("duplicate", f"{line.changed_by} already made this change to {edit.line_id}")
            if other_review and edit.target == "ingredients":
                raise _Refusal("conflict", f"{edit.line_id} was already changed by {line.changed_by}")
            spans, method = _spans(edit.find, line.text)
            if not spans:
                reason = "conflict" if other_review else "stale"
                raise _Refusal(reason, f"'{edit.find}' is no longer on {edit.line_id}; "
                                       f"{line.changed_by} changed it")
            if len(spans) > 1:
                raise _Refusal("ambiguous", f"'{edit.find}' occurs {len(spans)} times on {edit.line_id}")
        new_text = _replace_once(line.text, spans[0], edit.replace)
        if _norm(new_text) == _norm(line.text):
            raise _Refusal("no_op", f"the edit leaves {edit.line_id} unchanged")
        if not new_text:
            raise _Refusal("invalid", f"the edit would empty {edit.line_id}; use remove")
        before = line.text
        line.text, line.changed_by, line.review_id = new_text, tweak.id, tweak.review_id
        return EditOutcome(edit_id=edit.id, status="applied", line_id=edit.line_id,
                           before=before, after=new_text, match_method=method, match_score=1.0)

    def _apply_tweak(self, state: _State, tweak: Tweak) -> Tuple[_State, TweakOutcome]:
        """Try every edit on a copy; keep the copy only if all of them applied."""
        trial = copy.deepcopy(state)
        outcomes = []
        for edit in tweak.edits:
            try:
                outcomes.append(self._apply_edit(trial, edit, tweak))
            except _Refusal as r:
                outcomes.append(EditOutcome(edit_id=edit.id, status="refused", reason=r.reason,
                                            detail=r.detail, line_id=edit.line_id))
        refused = [o for o in outcomes if o.status == "refused"]
        if not refused:
            return trial, TweakOutcome(tweak_id=tweak.id, status="applied", edit_outcomes=outcomes)
        rolled_back = [
            o if o.status == "refused" else o.model_copy(update={"status": "rolled_back"})
            for o in outcomes
        ]
        detail = "; ".join(f"{o.edit_id}: {o.reason}: {o.detail}" for o in refused)
        logger.warning(f"Tweak {tweak.id} not applied: {detail}")
        return state, TweakOutcome(tweak_id=tweak.id, status="refused", detail=detail,
                                   edit_outcomes=rolled_back)

    def apply_tweaks(
        self,
        recipe: Recipe,
        tweaks: List[Tweak],
        review_order: Optional[Dict[str, int]] = None,
    ) -> Tuple[Recipe, List[TweakOutcome]]:
        """
        Apply tweaks to a recipe, each all-or-nothing.

        Args:
            recipe: Original recipe (as loaded; not modified)
            tweaks: Tweaks with edits targeting the original recipe's line IDs
            review_order: Review ID -> rank (1 = first). Tweaks are applied by review
                rank, then in the order given, so higher-ranked reviews win conflicts.

        Returns:
            Tuple of (modified recipe, one TweakOutcome per tweak in application order)
        """
        rank = review_order or {}
        ordered = sorted(enumerate(tweaks), key=lambda p: (rank.get(p[1].review_id, float("inf")), p[0]))
        state = self._initial_state(recipe)
        outcomes = []
        for _, tweak in ordered:
            if not tweak.edits:
                outcomes.append(TweakOutcome(tweak_id=tweak.id, status="skipped", reason="no_edits"))
                continue
            state, outcome = self._apply_tweak(state, tweak)
            outcomes.append(outcome)

        applied = sum(o.status == "applied" for o in outcomes)
        logger.info(f"Applied {applied}/{len(outcomes)} tweaks")
        modified = Recipe(
            recipe_id=f"{recipe.recipe_id}_modified",
            title=recipe.title,
            ingredients=[l.text for l in state.lines["ingredients"]],
            instructions=[l.text for l in state.lines["instructions"]],
            description=recipe.description,
            servings=recipe.servings,
            rating=recipe.rating,
            prep_time=recipe.prep_time,
            cook_time=recipe.cook_time,
            total_time=recipe.total_time,
        )
        return modified, outcomes
