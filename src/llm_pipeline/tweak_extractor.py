"""
Step 1: Tweak Extraction & Parsing

This module extracts structured modifications from review text using LLM processing.
It asks for every tweak in a review and accepts only what it can verify:

- the response is a JSON object with a "tweaks" list
- each tweak has every required field with an allowed value
- its quote appears in the review
- each edit points at an existing line of the right section, its find text is on
  that line, and it is complete for its operation
- every amount an edit introduces is supported by the review (see support.py)

If anything fails, the model gets one corrective retry listing exactly what was
wrong. Tweaks that still fail are kept as rejected, with their problems. A review
with no tweaks is a valid "empty" result, not a failure.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from openai import OpenAI
from pydantic import ValidationError

from .loader import normalize_text, parse_recipe, review_id
from .models import (
    ExtractedTweak,
    ExtractionResult,
    Recipe,
    RejectedTweak,
    Review,
    Tweak,
    TweakEdit,
)
from .prompts import PROMPT_VERSION, build_correction_message, build_extraction_messages
from .support import plan_language, quote_in_text, unsupported_amounts, unsupported_units, unsupported_words

Valid = Tuple[ExtractedTweak, List[str]]  # tweak and its warnings


def _norm(text: str) -> str:
    return normalize_text(text).casefold()


def _recipe_lines(recipe: Recipe) -> Dict[str, Tuple[str, str]]:
    lines = {f"ing-{i}": ("ingredients", t) for i, t in enumerate(recipe.ingredients)}
    lines.update({f"step-{i}": ("instructions", t) for i, t in enumerate(recipe.instructions)})
    return lines


def _targets(proposed: Dict[str, Any]) -> frozenset:
    """The (target, line, find) triples a proposed tweak edits; empty if it has none."""
    edits = proposed.get("edits") if isinstance(proposed, dict) else None
    return frozenset(
        (str(e.get("target")), str(e.get("line_id")), _norm(str(e.get("find") or "")))
        for e in (edits if isinstance(edits, list) else []) if isinstance(e, dict)
    )


def parse_response(content: str) -> Tuple[Optional[List[Any]], List[str]]:
    """Return the raw tweak list, or None and the problem with the response."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return None, [f"the response is not valid JSON ({e})"]
    if not isinstance(data, dict) or not isinstance(data.get("tweaks"), list):
        return None, ["the response must be a JSON object with a 'tweaks' list"]
    return data["tweaks"], []


def validate_tweak(raw: Any, review: Review, recipe: Recipe) -> Tuple[Optional[ExtractedTweak], List[str], List[str]]:
    """
    Check one proposed tweak against the review and the recipe.

    Returns:
        (tweak if valid else None, problems, warnings)
    """
    try:
        tweak = ExtractedTweak.model_validate(raw)
    except ValidationError as e:
        return None, [f"{'.'.join(map(str, err['loc'])) or 'tweak'}: {err['msg']}" for err in e.errors()], []

    problems: List[str] = []
    warnings: List[str] = []
    if not tweak.quote.strip():
        problems.append("quote is empty")
    elif not quote_in_text(tweak.quote, review.text):
        problems.append(f"quote '{tweak.quote}' does not appear in the review; copy it exactly")
    elif tweak.status == "tried" and plan_language(tweak.quote):
        problems.append(
            f"status is 'tried' but the quote describes a plan ('{plan_language(tweak.quote)}'); "
            "use 'suggested', or quote where the reviewer says they did it"
        )
    if not tweak.reasoning.strip():
        problems.append("reasoning is empty")

    lines = _recipe_lines(recipe)
    recipe_text = " ".join(recipe.ingredients + recipe.instructions)
    for n, edit in enumerate(tweak.edits, start=1):
        where = f"edit {n}"
        if edit.line_id not in lines:
            problems.append(f"{where}: line_id '{edit.line_id}' does not exist in the recipe")
            continue
        section, line = lines[edit.line_id]
        if section != edit.target:
            problems.append(f"{where}: {edit.line_id} is not an {edit.target} line")
            continue
        if not edit.find.strip() or _norm(edit.find) not in _norm(line):
            elsewhere = [
                lid for lid, (s, text) in lines.items()
                if s == edit.target and edit.find.strip() and _norm(edit.find) in _norm(text)
            ]
            hint = f" (it is in {', '.join(elsewhere)})" if elsewhere else ""
            problems.append(f"{where}: find '{edit.find}' is not in {edit.line_id}{hint}")
            continue

        if edit.operation == "remove":
            if _norm(edit.find) != _norm(line):
                problems.append(f"{where}: remove must use the whole line as find: '{line}'")
            continue
        if edit.operation == "replace":
            if edit.replace is None:
                problems.append(f"{where}: replace needs replacement text ('' deletes)")
                continue
            if _norm(edit.replace) == _norm(edit.find):
                problems.append(f"{where}: the replacement is the same as the found text")
                continue
            new_text, old_text = edit.replace, line
        else:
            if not (edit.add or "").strip():
                problems.append(f"{where}: add_after needs the new line in 'add'")
                continue
            new_text, old_text = edit.add, ""

        amounts = unsupported_amounts(new_text, old_text, review.text)
        if amounts:
            shown = ", ".join(f"{a:g}" for a in amounts)
            problems.append(
                f"{where}: amount {shown} in '{new_text}' is not stated in the review "
                "or derived from it and the recipe"
            )
        elif units := unsupported_units(new_text, old_text, review.text):
            problems.append(
                f"{where}: amount {', '.join(units)} in '{new_text}' is not stated in the review "
                "with that unit; do not convert units"
            )
        if edit.target == "ingredients":
            words = unsupported_words(new_text, old_text, review.text, recipe_text)
            if words:
                warnings.append(f"{edit.line_id}: words in neither review nor recipe: {', '.join(words)}")

    return (None if problems else tweak), problems, warnings


class LLMConfigurationError(RuntimeError):
    """The model cannot be reached as configured (e.g. no API key)."""


DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_SDK_RETRIES = 1


class TweakExtractor:
    """Extracts structured modifications from review text using LLM processing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        client=None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_SDK_RETRIES,
    ):
        """
        Initialize the TweakExtractor.

        Model access is injectable: pass any client with an OpenAI-style
        chat.completions.create(). Otherwise an OpenAI client is built with a bounded
        timeout. Per review there are at most 2 requests (one corrective retry), each
        retried at most max_retries times by the SDK on transient errors.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use for extraction (must support JSON mode)
            client: OpenAI-compatible client to use instead of building one
            timeout: Seconds before a single request is abandoned
            max_retries: SDK retries per request for transient errors

        Raises:
            LLMConfigurationError: If no client is given and no API key is available
        """
        if client is None:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise LLMConfigurationError(
                    "No OpenAI API key: set OPENAI_API_KEY (e.g. in .env) or inject a client"
                )
            client = OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)
        self.client = client
        self.model = model
        self.source = getattr(client, "description", None) or f"model {model}"
        self.prompt_version = PROMPT_VERSION
        logger.info(f"Initialized TweakExtractor with model: {model}")

    def _ask(self, messages: List[dict]) -> Tuple[Optional[str], Optional[str]]:
        """One LLM call. Returns (content, None) or (None, error)."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=4000,
            )
        except Exception as e:
            return None, f"the API call failed ({e})"
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            return None, "the response was cut off at the token limit"
        if not choice.message.content:
            return None, "the response was empty"
        return choice.message.content, None

    def _check(self, content: str, review: Review, recipe: Recipe) -> Tuple[List[Valid], List[RejectedTweak], List[str]]:
        """Validate a response: (valid tweaks, rejected tweaks, response-level problems)."""
        raw_tweaks, problems = parse_response(content)
        if raw_tweaks is None:
            return [], [], problems
        valid: List[Valid] = []
        rejected: List[RejectedTweak] = []
        seen = set()
        for n, raw in enumerate(raw_tweaks, start=1):
            tweak, tweak_problems, warnings = validate_tweak(raw, review, recipe)
            if tweak and _norm(tweak.quote) in seen:
                tweak, tweak_problems = None, ["it repeats an earlier tweak with the same quote"]
            if tweak:
                seen.add(_norm(tweak.quote))
                valid.append((tweak, warnings))
            else:
                quote = raw.get("quote") if isinstance(raw, dict) else None
                rejected.append(RejectedTweak(
                    quote=quote if isinstance(quote, str) else "",
                    problems=[f"tweak {n}: {p}" for p in tweak_problems],
                    proposed=raw if isinstance(raw, dict) else {"value": raw},
                ))
        return valid, rejected, []

    def extract_tweaks(self, review: Review, recipe: Recipe) -> ExtractionResult:
        """
        Extract every tweak in a review, with at most one corrective retry.

        Args:
            review: Review to analyze (the has_modification flag is not required)
            recipe: Original recipe, as normalized by the loader

        Returns:
            ExtractionResult: ok (tweaks found), empty (none in the review) or failed
        """
        rid = review.id or review_id(review.text)
        messages = build_extraction_messages(review.text, recipe, review.rating)
        logger.debug(f"Extracting tweaks from review {rid}: {review.text[:100]}...")

        content, error = self._ask(messages)
        valid, rejected, problems = ([], [], [error]) if error else self._check(content, review, recipe)
        if not rejected and not problems:
            return self._result(rid, valid, [], [], attempts=1)

        # One corrective retry: show the model exactly what failed
        all_problems = problems + [p for r in rejected for p in r.problems]
        logger.warning(f"Review {rid}: retrying after {len(all_problems)} problem(s): {all_problems}")
        followup = messages if content is None else messages + [
            {"role": "assistant", "content": content}, build_correction_message(all_problems)]
        content2, error2 = self._ask(followup)
        valid2, rejected2, problems2 = ([], [], [error2]) if error2 else self._check(content2, review, recipe)
        if problems2:  # the retry produced nothing usable: keep what the first answer had
            return self._result(rid, valid, rejected, [f"retry: {p}" for p in problems2] + problems, attempts=2)

        # The corrected answer is authoritative, but keep valid first-answer tweaks it
        # dropped, and record rejected ones it silently left out. A tweak counts as
        # answered if the correction has one with the same quote or the same edit
        # targets, since a fix usually changes one of the two.
        quotes = {_norm(t.quote) for t, _ in valid2} | {_norm(r.quote) for r in rejected2 if r.quote}
        targets = {_targets(t.model_dump()) for t, _ in valid2} | {_targets(r.proposed) for r in rejected2}

        def answered(quote: str, proposed: Dict[str, Any]) -> bool:
            return _norm(quote) in quotes or (bool(_targets(proposed)) and _targets(proposed) in targets)

        kept = [(t, w) for t, w in valid if not answered(t.quote, t.model_dump())]
        withdrawn = [
            r.model_copy(update={"problems": r.problems + ["left out of the corrected answer"]})
            for r in rejected if not answered(r.quote, r.proposed)
        ]
        return self._result(rid, kept + valid2, rejected2 + withdrawn, [], attempts=2)

    @staticmethod
    def _result(rid: str, valid: List[Valid], rejected: List[RejectedTweak], errors: List[str], attempts: int) -> ExtractionResult:
        tweaks = []
        for n, (t, warnings) in enumerate(valid, start=1):
            tid = f"{rid}.t{n}"
            tweaks.append(Tweak(
                id=tid,
                review_id=rid,
                modification_type=t.modification_type,
                reasoning=t.reasoning,
                evidence_quote=t.quote,
                status=t.status,
                reviewer_outcome=t.reviewer_outcome,
                warnings=warnings,
                edits=[
                    TweakEdit(id=f"{tid}.e{i}", tweak_id=tid, target=e.target, operation=e.operation,
                              find=e.find, replace=e.replace, add=e.add, line_id=e.line_id)
                    for i, e in enumerate(t.edits, start=1)
                ],
            ))
        status = "ok" if tweaks else ("failed" if rejected or errors else "empty")
        result = ExtractionResult(review_id=rid, status=status, tweaks=tweaks, rejected=rejected,
                                  errors=errors, attempts=attempts)
        logger.info(f"Review {rid}: {status}, {len(tweaks)} tweaks, {len(rejected)} rejected, {attempts} attempt(s)")
        return result

    def extract_all(self, reviews: List[Review], recipe: Recipe) -> List[Tuple[Review, ExtractionResult]]:
        """Extract every review, in the order given (best-ranked first from the loader)."""
        return [(review, self.extract_tweaks(review, recipe)) for review in reviews]

    def test_extraction(self, review_text: str, recipe_data: dict) -> ExtractionResult:
        """
        Test extraction with raw text and recipe data.

        Args:
            review_text: Raw review text
            recipe_data: Raw recipe dictionary (recipe_id, title, ingredients, instructions)

        Returns:
            ExtractionResult
        """
        review = Review(text=review_text, id=review_id(review_text), has_modification=True)
        return self.extract_tweaks(review, parse_recipe(recipe_data))
