"""
Input loading and normalization.

Turns a scraped recipe file into a validated Recipe and a deduplicated, ranked list
of Reviews with stable IDs. Text quirks in the scrape are handled once, here, so
every later step (prompting, matching, attribution) sees the same text.

Normalization (recipe lines, titles and review text):
- decode HTML entities ("&#39;" -> "'"), NFC Unicode, collapse whitespace, trim
- drop empty lines; the scraped text is kept as RecipeLine.raw_text
- amounts are not rewritten ("1/2 cup" stays as scraped)
- durations stay ISO 8601 ("PT20M"); a zero duration ("PT0M") is a scrape artifact
  and becomes None

Stable IDs:
- recipe lines: "ing-<n>" / "step-<n>", the 0-based position after normalization.
  Recipe.content_hash identifies the text those positions refer to.
- reviews: "rev-" + 10 hex chars of SHA-1 over the normalized, case-folded text.
  A review gets the same ID in reviews and featured_tweaks, after re-ordering,
  and across re-scrapes; identical text is one review.

Review ranking (best first). Each signal only breaks ties left by the one before:
  1. featured          listed in featured_tweaks (shown with photos on AllRecipes)
  2. helpful_votes     vote count if the scrape has one (missing counts as 0).
                       The supplied data and current scraper have no vote data,
                       so today this level never separates reviews.
  3. has_modification  the scraper's keyword flag (weak: it misses real tweaks)
  4. rating            star rating (missing counts as 0)
  5. file order        reviews first, then featured-only entries, by position

Every review is extracted, flagged or not; the ranking sets the order, and the
pipeline applies tweaks from the best-ranked review that has any it can apply.
"""

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .models import Recipe, RecipeLine, Review

VOTE_KEYS = ("helpful_votes", "helpful_count")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


class RecipeDataError(ValueError):
    """The recipe file does not satisfy the input contract."""


@dataclass
class LoadedRecipe:
    """A normalized recipe and its ranked, deduplicated reviews."""

    recipe: Recipe
    reviews: List[Review]
    source_file: Optional[str] = None


def normalize_text(text: str) -> str:
    """Decode HTML entities, NFC-normalize, collapse whitespace and trim."""
    previous = None
    while previous != text:  # handles double-escaped entities such as "&amp;#39;"
        previous, text = text, html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def review_key(text: str) -> str:
    """Key under which two review texts count as the same review."""
    return normalize_text(text).translate(_QUOTES).casefold()


def review_id(text: str) -> str:
    return "rev-" + hashlib.sha1(review_key(text).encode("utf-8")).hexdigest()[:10]


def normalize_duration(value: Any) -> Optional[str]:
    """Keep an ISO 8601 duration, treating missing or zero durations as None."""
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if re.fullmatch(r"P(T(\d+H)?(\d+M)?(\d+S)?)?", value) and not re.search(r"[1-9]", value):
        return None
    return value


def _rating(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        rating = int(value)
    except (TypeError, ValueError):
        return None
    return rating if 1 <= rating <= 5 and rating == float(value) else None


def _votes(item: Dict[str, Any]) -> Optional[int]:
    for key in VOTE_KEYS:
        value = item.get(key)
        if isinstance(value, bool):
            continue
        try:
            votes = int(value)
        except (TypeError, ValueError):
            continue
        if votes >= 0:
            return votes
    return None


def _lines(data: Dict[str, Any], section: str) -> List[RecipeLine]:
    raw = data.get(section)
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise RecipeDataError(f"'{section}' must be a list of strings")
    lines = []
    for raw_text in raw:
        text = normalize_text(raw_text)
        if not text:
            logger.warning(f"Dropping empty {section} line")
            continue
        prefix = "ing" if section == "ingredients" else "step"
        lines.append(
            RecipeLine(
                id=f"{prefix}-{len(lines)}", section=section, index=len(lines),
                text=text, raw_text=raw_text,
            )
        )
    if not lines:
        raise RecipeDataError(f"recipe has no {section}")
    return lines


def parse_recipe(data: Dict[str, Any]) -> Recipe:
    """Validate and normalize the recipe part of a scraped recipe file."""
    if not isinstance(data, dict):
        raise RecipeDataError("recipe data must be a JSON object")
    recipe_id = data.get("recipe_id")
    title = data.get("title")
    if recipe_id in (None, "") or not isinstance(recipe_id, (str, int)):
        raise RecipeDataError("recipe has no recipe_id")
    if not isinstance(title, str) or not normalize_text(title):
        raise RecipeDataError("recipe has no title")

    lines = _lines(data, "ingredients") + _lines(data, "instructions")
    ingredients = [x.text for x in lines if x.section == "ingredients"]
    instructions = [x.text for x in lines if x.section == "instructions"]
    content = json.dumps([ingredients, instructions], ensure_ascii=False)
    description = data.get("description")
    servings = data.get("servings")

    return Recipe(
        recipe_id=str(recipe_id),
        title=normalize_text(title),
        ingredients=ingredients,
        instructions=instructions,
        description=normalize_text(description) if isinstance(description, str) else None,
        servings=str(servings) if servings not in (None, "") else None,
        rating=data.get("rating") if isinstance(data.get("rating"), dict) else None,
        prep_time=normalize_duration(data.get("preptime")),
        cook_time=normalize_duration(data.get("cooktime")),
        total_time=normalize_duration(data.get("totaltime")),
        lines=lines,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
    )


def parse_reviews(data: Dict[str, Any]) -> List[Review]:
    """
    Merge reviews and featured_tweaks into unique reviews, ranked best first.

    A review listed in both places becomes one Review that is featured, keeps every
    source position, and takes the first known rating and username and the highest
    vote count.
    """
    merged: Dict[str, Review] = {}
    order: Dict[str, tuple] = {}

    for source_rank, source in enumerate(("reviews", "featured_tweaks")):
        items = data.get(source) or []
        if not isinstance(items, list):
            raise RecipeDataError(f"'{source}' must be a list")
        for index, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                logger.warning(f"Skipping {source}[{index}]: no review text")
                continue
            text = normalize_text(item["text"])
            if not text:
                logger.warning(f"Skipping {source}[{index}]: empty review text")
                continue
            rid = review_id(text)
            rating = _rating(item.get("rating"))
            username = normalize_text(item["username"]) if isinstance(item.get("username"), str) else None
            featured = source == "featured_tweaks" or item.get("is_featured") is True
            votes = _votes(item)
            pointer = f"{source}[{index}]"

            if rid not in merged:
                merged[rid] = Review(
                    id=rid, text=text, rating=rating, username=username or None,
                    has_modification=item.get("has_modification") is True,
                    featured=featured, helpful_votes=votes, sources=[pointer],
                )
                order[rid] = (source_rank, index)
                continue

            review = merged[rid]
            if rating is not None and review.rating is not None and rating != review.rating:
                logger.warning(f"{pointer} rates review {rid} {rating}, kept {review.rating}")
            review.rating = review.rating if review.rating is not None else rating
            review.username = review.username or username or None
            review.has_modification = review.has_modification or item.get("has_modification") is True
            review.featured = review.featured or featured
            if votes is not None:
                review.helpful_votes = max(votes, review.helpful_votes or 0)
            review.sources.append(pointer)

    return rank_reviews(list(merged.values()), order)


def rank_reviews(reviews: List[Review], order: Optional[Dict[str, tuple]] = None) -> List[Review]:
    """Sort reviews best first by the documented signals and set Review.rank."""
    order = order or {}
    ranked = sorted(
        enumerate(reviews),
        key=lambda pair: (
            not pair[1].featured,
            -(pair[1].helpful_votes or 0),
            not pair[1].has_modification,
            -(pair[1].rating or 0),
            order.get(pair[1].id, (0, pair[0])),
        ),
    )
    result = []
    for rank, (_, review) in enumerate(ranked, start=1):
        review.rank = rank
        result.append(review)
    return result


def load_recipe_file(path: str) -> LoadedRecipe:
    """Load, validate and normalize one scraped recipe file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RecipeDataError(f"cannot read {path}: {e}") from e
    recipe = parse_recipe(data)
    reviews = parse_reviews(data)
    return LoadedRecipe(recipe=recipe, reviews=reviews, source_file=str(path))
