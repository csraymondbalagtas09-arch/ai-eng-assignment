"""Tests for input loading: normalization, deduplication, stable IDs and ranking."""

import json
import random
from pathlib import Path

import pytest

from llm_pipeline.loader import (
    RecipeDataError,
    load_recipe_file,
    normalize_duration,
    normalize_text,
    parse_recipe,
    parse_reviews,
    rank_reviews,
    review_id,
)
from llm_pipeline.models import Review

DATA = Path(__file__).resolve().parent.parent / "data"
COOKIES = DATA / "recipe_10813_best-chocolate-chip-cookies.json"
SOUP = DATA / "recipe_77935_creamy-sweet-potato-with-ginger-soup.json"
NIKUJAGA = DATA / "recipe_144299_nikujaga-japanese-style-meat-and-potatoes-.json"


def raw(path):
    return json.loads(path.read_text(encoding="utf-8"))


def minimal(**overrides):
    data = {"recipe_id": "1", "title": "T", "ingredients": ["1 cup flour"], "instructions": ["Mix."]}
    data.update(overrides)
    return data


# --- Normalization ----------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("lid&#39;s pop-out", "lid's pop-out"),
        ("double &amp;#39;escaped", "double 'escaped"),
        ("  two   spaces\tand\ttabs\n", "two spaces and tabs"),
        ("non breaking", "non breaking"),
        ("café", "café"),  # NFC
        ("1/2 cup shortening", "1/2 cup shortening"),  # amounts are not rewritten
    ],
)
def test_normalize_text(text, expected):
    assert normalize_text(text) == expected


def test_recipe_lines_are_normalized_with_raw_text_kept():
    recipe = load_recipe_file(str(SOUP)).recipe
    line = recipe.line("step-5")
    assert "lid's pop-out" in line.text and "&#39;" not in line.text
    assert "&#39;" in line.raw_text
    assert recipe.instructions[5] == line.text


@pytest.mark.parametrize(
    "value, expected",
    [("PT20M", "PT20M"), ("PT1H5M", "PT1H5M"), ("PT0M", None), ("P", None), (None, None), ("", None), (30, None),
     ("about 20 minutes", "about 20 minutes")],
)
def test_normalize_duration(value, expected):
    assert normalize_duration(value) == expected


def test_times_are_mapped_from_scraped_keys():
    recipe = load_recipe_file(str(COOKIES)).recipe
    assert (recipe.prep_time, recipe.cook_time, recipe.total_time) == ("PT20M", "PT10M", "PT30M")
    assert load_recipe_file(str(SOUP)).recipe.total_time is None  # scraped as PT0M


# --- Stable line IDs ---------------------------------------------------------------

def test_line_ids_follow_position_in_each_section():
    recipe = load_recipe_file(str(COOKIES)).recipe
    assert [l.id for l in recipe.lines[:3]] == ["ing-0", "ing-1", "ing-2"]
    assert recipe.line("ing-10").text == "1 cup chopped walnuts"
    assert recipe.line("step-0").section == "instructions"
    with pytest.raises(KeyError):
        recipe.line("ing-99")


def test_empty_lines_are_dropped_and_ids_stay_contiguous():
    recipe = parse_recipe(minimal(ingredients=["1 cup flour", "   ", "2 eggs"]))
    assert [(l.id, l.text) for l in recipe.lines if l.section == "ingredients"] == [
        ("ing-0", "1 cup flour"), ("ing-1", "2 eggs")]


def test_content_hash_is_stable_and_tracks_text():
    a = parse_recipe(raw(COOKIES))
    b = parse_recipe(raw(COOKIES))
    assert a.content_hash == b.content_hash
    changed = raw(COOKIES)
    changed["ingredients"][0] = "2 cups butter, softened"
    assert parse_recipe(changed).content_hash != a.content_hash
    spaced = raw(COOKIES)
    spaced["ingredients"][0] = "  1 cup  butter, softened "
    assert parse_recipe(spaced).content_hash == a.content_hash  # whitespace is not content


# --- Input contract ----------------------------------------------------------------

@pytest.mark.parametrize(
    "data, message",
    [
        ([], "JSON object"),
        (minimal(recipe_id=None), "recipe_id"),
        (minimal(title="  "), "title"),
        (minimal(ingredients="1 cup flour"), "list of strings"),
        (minimal(ingredients=["1 cup flour", 3]), "list of strings"),
        (minimal(instructions=[]), "no instructions"),
        (minimal(instructions=["   "]), "no instructions"),
        (minimal(reviews={"text": "x"}), "'reviews' must be a list"),
    ],
)
def test_invalid_recipe_data_is_rejected(data, message):
    with pytest.raises(RecipeDataError, match=message):
        parse_recipe(data) if "reviews" not in data else parse_reviews(data)


def test_unreadable_file_is_a_data_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(RecipeDataError, match="cannot read"):
        load_recipe_file(str(bad))


def test_every_supplied_recipe_satisfies_the_contract():
    counts = {}
    for path in sorted(DATA.glob("recipe_*.json")):
        loaded = load_recipe_file(str(path))
        counts[loaded.recipe.recipe_id] = len(loaded.reviews)
    assert counts == {"10813": 9, "144299": 2, "19117": 2, "284494": 0, "45613": 0, "77935": 6}


# --- Deduplication and review IDs ---------------------------------------------------

def test_featured_duplicates_merge_into_one_featured_review():
    reviews = parse_reviews(raw(COOKIES))
    assert len(reviews) == 9  # 9 reviews + 4 featured duplicates
    featured = [r for r in reviews if r.featured]
    assert len(featured) == 4
    for r in featured:
        assert any(s.startswith("reviews[") for s in r.sources)
        assert any(s.startswith("featured_tweaks[") for s in r.sources)


def test_review_ids_are_content_based():
    reviews = parse_reviews(raw(COOKIES))
    ids = [r.id for r in reviews]
    assert len(set(ids)) == len(ids)
    shuffled = raw(COOKIES)
    random.Random(0).shuffle(shuffled["reviews"])
    assert {r.id for r in parse_reviews(shuffled)} == set(ids)
    assert review_id("It is a  great recipe") == review_id("it is a great recipe")


def test_near_duplicate_text_merges_and_keeps_first_known_values():
    data = {"reviews": [
        {"text": "I’d add  MORE salt.", "rating": None},
        {"text": "i'd add more salt.", "rating": 4, "username": "cook1", "has_modification": True},
    ]}
    [review] = parse_reviews(data)
    assert review.text == "I’d add MORE salt."
    assert (review.rating, review.username, review.has_modification) == (4, "cook1", True)
    assert review.sources == ["reviews[0]", "reviews[1]"]


@pytest.mark.parametrize("value, expected", [(5, 5), ("4", 4), (4.0, 4), (4.5, None), (0, None), (7, None),
                                             ("abc", None), (True, None), (None, None)])
def test_ratings_are_coerced_or_dropped(value, expected):
    [review] = parse_reviews({"reviews": [{"text": "x", "rating": value}]})
    assert review.rating == expected


def test_reviews_without_text_are_skipped():
    data = {"reviews": [{"rating": 5}, {"text": "  "}, "not a dict", {"text": "kept"}]}
    assert [r.text for r in parse_reviews(data)] == ["kept"]


# --- Ranking and selection ----------------------------------------------------------

def R(text, **kw):
    return Review(text=text, id=review_id(text), **kw)


def test_ranking_signals_apply_in_documented_order():
    reviews = [
        R("plain 5 star", rating=5),
        R("flagged 3 star", rating=3, has_modification=True),
        R("voted", rating=1, helpful_votes=12),
        R("featured", rating=2, featured=True, has_modification=True),
        R("voted less", rating=5, helpful_votes=3),
    ]
    ranked = rank_reviews(reviews)
    assert [r.text for r in ranked] == ["featured", "voted", "voted less", "flagged 3 star", "plain 5 star"]
    assert [r.rank for r in ranked] == [1, 2, 3, 4, 5]


def test_missing_votes_and_ratings_count_as_zero_and_file_order_breaks_ties():
    ranked = rank_reviews([R("b"), R("a", helpful_votes=0), R("c", rating=None)])
    assert [r.text for r in ranked] == ["b", "a", "c"]


def test_votes_are_read_when_the_scrape_has_them():
    data = {"reviews": [{"text": "few", "helpful_count": 2}, {"text": "many", "helpful_votes": "9"},
                        {"text": "bad", "helpful_votes": -1}]}
    reviews = {r.text: r for r in parse_reviews(data)}
    assert (reviews["few"].helpful_votes, reviews["many"].helpful_votes, reviews["bad"].helpful_votes) == (2, 9, None)
    assert reviews["many"].rank == 1


def test_ranking_on_supplied_data_without_vote_data():
    # Cookies: featured reviews come first; ties fall back to flag, rating, file order
    cookies = parse_reviews(raw(COOKIES))
    assert [r.sources[0] for r in cookies[:4]] == ["reviews[0]", "reviews[1]", "reviews[6]", "reviews[3]"]
    # Nikujaga: the unflagged review ranks after the featured one
    nikujaga = parse_reviews(raw(NIKUJAGA))
    assert [r.sources[0] for r in nikujaga] == ["reviews[1]", "reviews[0]"]


def test_ranking_is_deterministic():
    assert [r.id for r in parse_reviews(raw(SOUP))] == [r.id for r in parse_reviews(raw(SOUP))]
