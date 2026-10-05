"""
LLM prompts and examples for recipe modification extraction.

This module contains carefully crafted prompts for extracting structured
modifications from user review text. The example is synthetic so that it does not
leak any review from data/ into the prompt.
"""

import hashlib
import json
from typing import List, Optional

from .models import Recipe

SYSTEM_PROMPT = """You are an expert recipe analyst. Extract every recipe tweak a reviewer describes and express each one as precise edits to the original recipe.

A tweak is any change to ingredients (substitution, amount, addition, omission), technique, temperature, timing, equipment or portioning that the reviewer made or proposes. Praise, complaints and comments about taste or yield that describe no change are not tweaks.

Return a JSON object: {"tweaks": [...]}. Each tweak has:
- "modification_type": one of "ingredient_substitution", "quantity_adjustment", "technique_change", "addition", "removal"
- "quote": the shortest span of the review that states the tweak, copied character for character
- "status": "tried" if the reviewer says they made the change; "suggested" for plans, wishes and advice they did not carry out ("next time I will", "I would prefer", "if you can't find X, Y works")
- "reviewer_outcome": the reviewer's reported result of this change: "positive", "mixed", "negative" or "unclear"
- "reasoning": one sentence giving the reviewer's own reason or result; do not invent benefits
- "edits": a list of edits to the ORIGINAL recipe, each with:
  - "target": "ingredients" or "instructions"
  - "operation": "replace" (change part of a line), "add_after" (insert a new line after a line) or "remove" (delete a whole line)
  - "line_id": the ID of the line, exactly as shown in the recipe (e.g. "ing-3", "step-1")
  - "find": text copied exactly from that line. For "replace" use the shortest span that identifies what changes; for "remove" use the whole line; for "add_after" any text from the line
  - "replace": the new text for the found span (replace only; "" deletes it), otherwise null
  - "add": the complete new line (add_after only), otherwise null

Rules:
1. One entry per distinct tweak. Never merge tweaks and never drop one because it seems minor. If the review has no tweak, return {"tweaks": []}.
2. Amounts: use only amounts the reviewer states, or that follow directly from a stated amount and the recipe's amount (e.g. "1/2 cup less" than 3 cups is 2.5 cups). Never invent an amount or convert units. If no amount is given, write "(amount not specified)"; for "more"/"extra" of an existing ingredient keep its amount and add ", or more to taste".
3. Use the reviewer's words for what is added or substituted. Write amounts in the recipe's style (e.g. "0.5 cup").
4. Keep ingredients and steps consistent: when an ingredient is added, removed or swapped, also edit the step that uses it.
5. Give edits for suggested tweaks too. Use an empty edits list only when the tweak cannot be expressed as an edit (e.g. no details given).

Example
Recipe:
Ingredients:
[ing-0] 1.5 cups all-purpose flour
[ing-1] 1.25 cups milk
[ing-2] 1 egg
Instructions:
[step-0] Whisk the flour in a bowl.
[step-1] Stir in milk and egg until just combined.
Review: "I swapped half the milk for buttermilk and they were so fluffy! Next time I'll add blueberries."
Output:
""" + json.dumps(
    {
        "tweaks": [
            {
                "modification_type": "ingredient_substitution",
                "quote": "I swapped half the milk for buttermilk",
                "status": "tried",
                "reviewer_outcome": "positive",
                "reasoning": "The reviewer says the pancakes came out very fluffy.",
                "edits": [
                    {"target": "ingredients", "operation": "replace", "line_id": "ing-1",
                     "find": "1.25 cups milk", "replace": "0.625 cup milk", "add": None},
                    {"target": "ingredients", "operation": "add_after", "line_id": "ing-1",
                     "find": "1.25 cups milk", "replace": None, "add": "0.625 cup buttermilk"},
                    {"target": "instructions", "operation": "replace", "line_id": "step-1",
                     "find": "Stir in milk and egg", "replace": "Stir in milk, buttermilk and egg", "add": None},
                ],
            },
            {
                "modification_type": "addition",
                "quote": "Next time I'll add blueberries.",
                "status": "suggested",
                "reviewer_outcome": "unclear",
                "reasoning": "The reviewer plans to add blueberries but has not tried it.",
                "edits": [
                    {"target": "ingredients", "operation": "add_after", "line_id": "ing-2",
                     "find": "1 egg", "replace": None, "add": "blueberries (amount not specified)"},
                    {"target": "instructions", "operation": "replace", "line_id": "step-1",
                     "find": "until just combined.", "replace": "until just combined. Fold in blueberries.", "add": None},
                ],
            },
        ]
    }
)


USER_TEMPLATE = (
    "Original Recipe: {title}\n{recipe}\n\n"
    'User Review (rating {rating}):\n"""{review}"""\n\n'
    "Extract every tweak in this review as JSON."
)

# Changes whenever the system prompt or the request template changes
PROMPT_VERSION = hashlib.sha256((SYSTEM_PROMPT + USER_TEMPLATE).encode("utf-8")).hexdigest()[:12]


def format_recipe(recipe: Recipe) -> str:
    """Render the recipe with the line IDs that edits must refer to."""
    ingredients = "\n".join(f"[ing-{i}] {x}" for i, x in enumerate(recipe.ingredients))
    instructions = "\n".join(f"[step-{i}] {x}" for i, x in enumerate(recipe.instructions))
    return f"Ingredients:\n{ingredients}\nInstructions:\n{instructions}"


def build_extraction_messages(review_text: str, recipe: Recipe, rating: Optional[int] = None) -> List[dict]:
    """Messages asking for every tweak in one review."""
    rating_text = f"{rating}/5" if rating is not None else "not given"
    user = USER_TEMPLATE.format(
        title=recipe.title, recipe=format_recipe(recipe), rating=rating_text, review=review_text
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_correction_message(problems: List[str]) -> dict:
    """The single corrective follow-up, listing exactly what failed validation."""
    listed = "\n".join(f"- {p}" for p in problems)
    return {
        "role": "user",
        "content": (
            "Your answer could not be accepted:\n"
            f"{listed}\n\n"
            "Return the complete corrected JSON with every tweak, including the ones that were "
            "fine. Fix only the problems listed. If the review does not support a tweak, leave it out."
        ),
    }
