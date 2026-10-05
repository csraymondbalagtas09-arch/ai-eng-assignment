"""
Checks that a proposed tweak is supported by the review it came from.

Two kinds of check, both deterministic and explainable:

- Evidence: the tweak's quote must appear in the review (ignoring case, spacing,
  curly quotes and punctuation at its ends; "..." may join several spans).
- Amounts: every amount an edit introduces must be stated in the review, or follow
  from one review amount and one amount already on the line by a single +, -, x or /
  (e.g. "1/2 c less" flour on a "3 cups" line supports "2.5 cups"). Invented
  amounts and unit conversions fail.

Wording is only a warning: new words in an ingredient line that appear neither in
the review nor in the recipe are reported, because paraphrase ("lots of" -> "plenty
of") is common and harmless while invented details ("bell" peppers) are not, and a
word list cannot tell them apart.
"""

import re
from itertools import product
from typing import List, Set

from .loader import normalize_text

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125}
_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12,
    "half": 0.5, "quarter": 0.25, "third": 1 / 3,
    "double": 2, "doubled": 2, "twice": 2, "triple": 3, "tripled": 3,
}
_UNITS = (
    r"cups?|c|tablespoons?|tbsps?|tbs|teaspoons?|tsps?|pounds?|lbs?|ounces?|oz|pinch(?:es)?|"
    r"dash(?:es)?|cloves?|cans?|sticks?|hours?|hrs?|minutes?|mins?|quarts?|pints?|grams?|g|"
    r"kg|ml|liters?|handfuls?|slices?|eggs?|yolks?|pieces?"
)
_FILLER = {
    "a", "an", "and", "or", "of", "the", "to", "taste", "more", "less", "amount", "not",
    "specified", "plus", "with", "for", "in", "into", "about", "as", "needed", "optional",
    "extra", "additional", "little", "over", "few", "some", "if", "desired", "at", "least",
}
_TOLERANCE = 0.01


def _norm(text: str) -> str:
    return normalize_text(text).translate(_QUOTES).casefold()


def quote_in_text(quote: str, text: str) -> bool:
    """True if the quote (or every '...'-separated part of it) appears in the text."""
    haystack = _norm(text)
    parts = [_norm(p).strip(" \"'.,;:!?()") for p in re.split(r"\.\.\.|…", quote)]
    parts = [p for p in parts if p]
    return bool(parts) and all(p in haystack for p in parts)


def quantities(text: str) -> Set[float]:
    """Every amount a text states: digits, fractions, number words, 'a cup'."""
    t = _norm(text)
    for char, value in _FRACTIONS.items():
        t = t.replace(char, f" {value} ")
    t = re.sub(r"\(\d+\)", " ", t)  # list markers such as "(2)" are not amounts
    found: Set[float] = set()

    def take(pattern, value):
        # Record each match, then blank it so its parts are not counted again
        nonlocal t
        for m in re.finditer(pattern, t):
            found.add(value(m))
        t = re.sub(pattern, " ", t)

    take(r"\b(" + "|".join(_WORDS) + r")[\s-]+and[\s-]+a[\s-]+half\b", lambda m: _WORDS[m.group(1)] + 0.5)
    take(r"(?<![\d/.])(\d+)\s+(\d+)/([1-9]\d*)(?![\d/])", lambda m: int(m.group(1)) + int(m.group(2)) / int(m.group(3)))
    for m in re.finditer(r"(?<![\d.])(\d+)/(\d+)(?![\d/])", t):
        if int(m.group(2)):
            found.add(int(m.group(1)) / int(m.group(2)))
    for m in re.finditer(r"(?<![\d/.])\d+(?:\.\d+)?(?![\d/])", t):
        found.add(float(m.group()))
    for word in re.findall(r"[a-z]+", t):
        if word in _WORDS:
            found.add(_WORDS[word])
    for _ in re.finditer(r"\b(?:a|an)\s+(?:[a-z]+\s+)?(?:" + _UNITS + r")\b", t):
        found.add(1)
    return found


def unsupported_amounts(new_text: str, old_text: str, review_text: str) -> List[float]:
    """Amounts in new_text that neither the review nor the old line accounts for."""
    old, review = quantities(old_text), quantities(review_text)
    allowed = old | review
    for o, r in product(old, review):
        allowed |= {o + r, o - r, r - o, o * r}
        if r:
            allowed.add(o / r)
    return sorted(
        n for n in quantities(new_text) - old
        if not any(abs(n - a) < _TOLERANCE for a in allowed)
    )


_UNIT_NAMES = {
    "cup": "cup", "cups": "cup", "tablespoon": "tbsp", "tablespoons": "tbsp", "tbsp": "tbsp", "tbsps": "tbsp",
    "tbs": "tbsp", "teaspoon": "tsp", "teaspoons": "tsp", "tsp": "tsp", "tsps": "tsp", "pound": "lb",
    "pounds": "lb", "lb": "lb", "lbs": "lb", "ounce": "oz", "ounces": "oz", "oz": "oz", "pinch": "pinch",
    "pinches": "pinch", "dash": "dash", "dashes": "dash", "clove": "clove", "cloves": "clove", "can": "can",
    "cans": "can", "stick": "stick", "sticks": "stick", "hour": "hour", "hours": "hour", "hr": "hour",
    "hrs": "hour", "minute": "minute", "minutes": "minute", "min": "minute", "mins": "minute",
    "quart": "quart", "quarts": "quart", "pint": "pint", "pints": "pint", "gram": "g", "grams": "g",
    "kg": "kg", "ml": "ml", "liter": "l", "liters": "l",
}
_SHORT_UNITS = {"c": "cup", "g": "g", "l": "l"}  # only directly after the number ("1/2 c")
_COUNTING_WORDS = {k: v for k, v in _WORDS.items() if k not in ("double", "doubled", "twice", "triple", "tripled")}
_NUMBER = r"\d+(?:\.\d+)?"
_PLAN = re.compile(r"\b(next time|will|would|i'll|i'd|going to|plan to|want to|might|should)\b")


def plan_language(quote: str) -> str:
    """The wording that marks a quote as a plan rather than something done ('' if none)."""
    match = _PLAN.search(_norm(quote))
    return match.group(1) if match else ""


def _amounts_with_units(text: str) -> List[tuple]:
    """(amount, unit) pairs a text states, e.g. 'a half cup' -> (0.5, 'cup')."""
    t = _norm(text)
    for char, value in _FRACTIONS.items():
        t = t.replace(char, f" {value} ")
    t = re.sub(r"\(\d+\)", " ", t)
    t = re.sub(r"\b(" + "|".join(_WORDS) + r")[\s-]+and[\s-]+a[\s-]+half\b",
               lambda m: f" {_WORDS[m.group(1)] + 0.5} ", t)
    t = re.sub(r"(?<![\d/.])(\d+)\s+(\d+)/([1-9]\d*)(?![\d/])",
               lambda m: f" {int(m.group(1)) + int(m.group(2)) / int(m.group(3))} ", t)
    t = re.sub(r"(?<![\d.])(\d+)/([1-9]\d*)(?![\d/])", lambda m: f" {int(m.group(1)) / int(m.group(2))} ", t)
    t = re.sub(r"\b(?:a|an)\b(?=\s+(?:[a-z]+\s+)?(?:" + "|".join(_UNIT_NAMES) + r")\b)", " 1 ", t)
    t = re.sub(r"\b(" + "|".join(_COUNTING_WORDS) + r")\b", lambda m: f" {_COUNTING_WORDS[m.group(1)]} ", t)
    tokens = re.findall(_NUMBER + r"|[a-z%]+", t)
    pairs = []
    for i, token in enumerate(tokens):
        if not re.fullmatch(_NUMBER, token):
            continue
        for j, following in enumerate(tokens[i + 1:i + 4]):
            if re.fullmatch(_NUMBER, following):
                break
            unit = _UNIT_NAMES.get(following) or (_SHORT_UNITS.get(following) if j == 0 else None)
            if unit:
                pairs.append((float(token), unit))
                break
    return pairs


def unsupported_units(new_text: str, old_text: str, review_text: str) -> List[str]:
    """
    Amounts with a unit in new_text whose unit the review does not support.

    An amount and unit is supported if the old line already has it, the review
    states it, or it follows from an old amount in the same unit and a review amount
    in that unit (+, -) or any review number (x, /, e.g. 'doubled').
    """
    old, review = _amounts_with_units(old_text), _amounts_with_units(review_text)
    factors = quantities(review_text)

    def close(a, b):
        return abs(a - b) < _TOLERANCE

    problems = []
    for value, unit in _amounts_with_units(new_text):
        if any(u == unit and close(v, value) for v, u in old + review):
            continue
        derived = any(
            close(value, o + r) or close(value, o - r)
            for o, ou in old if ou == unit for r, ru in review if ru == unit
        ) or any(
            close(value, o * f) or (f and close(value, o / f))
            for o, ou in old if ou == unit for f in factors
        )
        if not derived:
            problems.append(f"{value:g} {unit}")
    return problems


def _stems(text: str) -> Set[str]:
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z]+", _norm(text))}


def unsupported_words(new_text: str, old_text: str, review_text: str, recipe_text: str) -> List[str]:
    """Words new_text adds that appear in neither the review nor the recipe."""
    known = _stems(old_text) | _stems(review_text) | _stems(recipe_text)
    units = _stems(" ".join(re.findall(r"[a-z]+", _UNITS.replace("?", " "))))
    return sorted(w for w in _stems(new_text) - known - _FILLER - units if len(w) > 1)
