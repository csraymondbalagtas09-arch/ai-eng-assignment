"""
Pydantic data models for the LLM Analysis Pipeline.

These models define the structure for recipe modifications, enhanced recipes,
and all intermediate data formats used throughout the pipeline.

Stable IDs (assigned by loader.py and Tweak.from_modification):
- recipe line:  "ing-<n>" / "step-<n>"   0-based position in the normalized list
- review:       "rev-<10 hex>"           hash of the review's normalized text
- tweak:        "<review id>.t<n>"
- edit:         "<tweak id>.e<n>"
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

LINE_ID = r"^(ing|step)-\d+$"
REVIEW_ID = r"^rev-[0-9a-f]{10}$"
TWEAK_ID = r"^rev-[0-9a-f]{10}\.t\d+$"
EDIT_ID = r"^rev-[0-9a-f]{10}\.t\d+\.e\d+$"
LINE_PREFIX = {"ingredients": "ing", "instructions": "step"}

ModificationType = Literal[
    "ingredient_substitution",
    "quantity_adjustment",
    "technique_change",
    "addition",
    "removal",
]


class ModificationEdit(BaseModel):
    """Individual atomic edit operation for a recipe modification."""

    target: Literal["ingredients", "instructions"] = Field(
        description="Whether this edit applies to ingredients or instructions"
    )
    operation: Literal["replace", "add_after", "remove"] = Field(
        default="replace",
        description="Type of operation: replace text, add after target, or remove",
    )
    find: str = Field(description="Text to find in the recipe")
    replace: Optional[str] = Field(
        default=None, description="Replacement text (required for replace operations)"
    )
    add: Optional[str] = Field(
        default=None, description="Text to add (required for add_after operations)"
    )

    @model_validator(mode="after")
    def _complete(self):
        # Invalid LLM output fails validation here, so the extractor retries it
        if not self.find.strip():
            raise ValueError("edit has no find text")
        if self.operation == "replace" and self.replace is None:
            raise ValueError("replace edit needs replacement text ('' deletes)")
        if self.operation == "add_after" and not (self.add or "").strip():
            raise ValueError("add_after edit needs text to add")
        return self


class ModificationObject(BaseModel):
    """Structured modification parsed from a review."""

    modification_type: Literal[
        "ingredient_substitution",
        "quantity_adjustment",
        "technique_change",
        "addition",
        "removal",
    ] = Field(description="Category of modification")

    reasoning: str = Field(description="Why this modification improves the recipe")

    edits: List[ModificationEdit] = Field(description="List of atomic edits to apply")


class SourceReview(BaseModel):
    """Reference to the original review that suggested the modification."""

    text: str = Field(description="Full text of the original review")
    reviewer: Optional[str] = Field(description="Username of the reviewer")
    rating: Optional[int] = Field(description="Star rating given by reviewer")
    review_id: Optional[str] = Field(default=None, description="Stable review ID")
    sources: List[str] = Field(
        default_factory=list,
        description="Where the review appears in the recipe file, e.g. reviews[1]",
    )
    featured: bool = Field(default=False, description="Listed in featured_tweaks")
    rank: Optional[int] = Field(default=None, description="Selection rank (1 = first)")


class ChangeRecord(BaseModel):
    """Record of a specific change made to the recipe."""

    type: Literal["ingredient", "instruction"] = Field(
        description="Type of element that was changed"
    )
    from_text: str = Field(description="Original text before modification")
    to_text: str = Field(description="New text after modification")
    operation: Literal["replace", "add", "remove"] = Field(
        description="Type of operation performed"
    )


class ModificationApplied(BaseModel):
    """Full record of a modification that was applied to a recipe."""

    source_review: SourceReview = Field(
        description="Review that suggested this modification"
    )
    modification_type: str = Field(description="Category of modification")
    reasoning: str = Field(description="Why this modification was applied")
    changes_made: List[ChangeRecord] = Field(
        description="Detailed list of changes made"
    )
    tweak_id: Optional[str] = Field(default=None, description="Stable tweak ID")
    evidence_quote: Optional[str] = Field(
        default=None, description="Verbatim span of the review that states the tweak"
    )


class EnhancementSummary(BaseModel):
    """Summary of all modifications applied to a recipe."""

    total_changes: int = Field(description="Total number of changes made")
    change_types: List[str] = Field(description="Types of modifications applied")
    expected_impact: str = Field(
        description="Expected improvement from these modifications"
    )


class EnhancedRecipe(BaseModel):
    """Recipe with community modifications applied and full attribution."""

    recipe_id: str = Field(description="Enhanced recipe ID")
    original_recipe_id: str = Field(description="ID of the original recipe")
    title: str = Field(description="Enhanced recipe title")

    # Enhanced recipe content
    ingredients: List[str] = Field(description="Modified ingredients list")
    instructions: List[str] = Field(description="Modified instructions list")

    # Attribution and tracking
    modifications_applied: List[ModificationApplied] = Field(
        description="Full record of all modifications applied"
    )
    enhancement_summary: EnhancementSummary = Field(
        description="Summary of all enhancements"
    )

    # Optional metadata
    description: Optional[str] = Field(description="Enhanced recipe description")
    servings: Optional[str] = Field(description="Number of servings")
    prep_time: Optional[str] = Field(description="Preparation time")
    cook_time: Optional[str] = Field(description="Cooking time")
    total_time: Optional[str] = Field(description="Total time")

    # Generation metadata
    created_at: str = Field(description="When this enhanced recipe was created")
    pipeline_version: str = Field(
        default="1.0.0", description="Version of the pipeline that created this"
    )
    source_recipe_hash: Optional[str] = Field(
        default=None,
        description="Content hash of the normalized source recipe that line IDs refer to",
    )
    extraction_results: List["ExtractionResult"] = Field(
        default_factory=list,
        description="Every review's extraction: tweaks found, rejected proposals, failures",
    )
    tweak_outcomes: List["TweakOutcome"] = Field(
        default_factory=list,
        description="What happened to every extracted tweak: applied, refused or skipped",
    )
    status: Literal["enhanced", "unchanged"] = Field(
        default="enhanced", description="Whether any tweak changed the recipe"
    )
    reviews: List[SourceReview] = Field(
        default_factory=list, description="Every review considered, in rank order"
    )
    recipe_diff: List["LineDiff"] = Field(
        default_factory=list,
        description="Line changes between the original and this recipe, computed from the text",
    )
    diff_verified: Optional[bool] = Field(
        default=None,
        description="True if every line change is explained by an applied edit and vice versa",
    )
    diff_problems: List[str] = Field(default_factory=list)
    extraction_source: Optional[str] = Field(
        default=None, description="Where extraction answers came from: a live model or a replay"
    )
    model: Optional[str] = Field(default=None, description="Model name configured for extraction")
    prompt_version: Optional[str] = Field(default=None, description="Hash of the extraction prompt")


class LineDiff(BaseModel):
    """One line that differs between the original and the enhanced recipe."""

    section: Literal["ingredients", "instructions"]
    kind: Literal["changed", "added", "removed"]
    line_id: Optional[str] = Field(
        default=None,
        description="Original line (for an added line: the line it follows; None at the top)",
    )
    before: Optional[str] = None
    after: Optional[str] = None
    tweak_ids: List[str] = Field(default_factory=list, description="Applied tweaks that explain it")


class RecipeRun(BaseModel):
    """The result of running the pipeline on one recipe file, including failures."""

    recipe_file: str
    status: Literal["enhanced", "unchanged", "skipped", "failed"]
    reason: Optional[
        Literal[
            "invalid_input", "no_reviews", "extraction_failed", "nothing_applicable",
            "nothing_applied", "error",
        ]
    ] = None
    detail: str = ""
    recipe_id: Optional[str] = None
    enhanced: Optional[EnhancedRecipe] = None
    outputs: List[str] = Field(default_factory=list, description="Files written")


class RecipeLine(BaseModel):
    """One normalized ingredient or instruction line with a stable ID."""

    id: str = Field(pattern=LINE_ID, description="ing-<n> or step-<n>")
    section: Literal["ingredients", "instructions"]
    index: int = Field(ge=0, description="0-based position in its section")
    text: str = Field(min_length=1, description="Normalized text")
    raw_text: str = Field(description="Text exactly as scraped")

    @model_validator(mode="after")
    def _id_matches_position(self):
        if self.id != f"{LINE_PREFIX[self.section]}-{self.index}":
            raise ValueError(f"line id {self.id} does not match {self.section}[{self.index}]")
        return self


class Recipe(BaseModel):
    """Base recipe model for input data."""

    recipe_id: str
    title: str
    ingredients: List[str]
    instructions: List[str]
    description: Optional[str] = None
    servings: Optional[str] = None
    rating: Optional[Dict[str, Any]] = None
    prep_time: Optional[str] = None
    cook_time: Optional[str] = None
    total_time: Optional[str] = None
    # Set by the loader for the original recipe; modified copies leave them empty
    lines: List[RecipeLine] = Field(default_factory=list)
    content_hash: Optional[str] = None

    @model_validator(mode="after")
    def _lines_mirror_text(self):
        if self.lines:
            for section in ("ingredients", "instructions"):
                texts = [line.text for line in self.lines if line.section == section]
                if texts != getattr(self, section):
                    raise ValueError(f"lines do not match {section}")
        return self

    def line(self, line_id: str) -> RecipeLine:
        """Look up a line of the original recipe by its ID."""
        for line in self.lines:
            if line.id == line_id:
                return line
        raise KeyError(line_id)


class Review(BaseModel):
    """Review model for input data."""

    text: str
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    username: Optional[str] = None
    has_modification: bool = False
    id: Optional[str] = Field(default=None, pattern=REVIEW_ID)
    featured: bool = Field(default=False, description="Listed in featured_tweaks")
    helpful_votes: Optional[int] = Field(default=None, ge=0)
    sources: List[str] = Field(
        default_factory=list, description="e.g. ['reviews[1]', 'featured_tweaks[1]']"
    )
    rank: Optional[int] = Field(default=None, ge=1, description="Selection rank (1 = first)")


# --- Tweak, edit and outcome records ----------------------------------------------
#
# A Tweak is one change a reviewer describes; its Edits are the concrete text
# operations that express it. Outcomes record what happened when they were applied:
# a tweak is applied all-or-nothing, so if any edit is refused, the edits that could
# have been applied are recorded as rolled back.


class TweakEdit(BaseModel):
    """One concrete text operation belonging to a tweak."""

    id: str = Field(pattern=EDIT_ID)
    tweak_id: str = Field(pattern=TWEAK_ID)
    target: Literal["ingredients", "instructions"]
    operation: Literal["replace", "add_after", "remove"]
    find: str = Field(min_length=1, description="Text to locate in the target line")
    replace: Optional[str] = Field(default=None, description="'' deletes the found span")
    add: Optional[str] = None
    line_id: Optional[str] = Field(
        default=None, pattern=LINE_ID, description="Intended target line, if known"
    )

    @model_validator(mode="after")
    def _consistent(self):
        if not self.id.startswith(self.tweak_id + ".e"):
            raise ValueError(f"edit {self.id} does not belong to tweak {self.tweak_id}")
        if self.operation == "replace" and self.replace is None:
            raise ValueError("replace edit needs replacement text")
        if self.operation == "add_after" and not (self.add or "").strip():
            raise ValueError("add_after edit needs text to add")
        if self.line_id and not self.line_id.startswith(LINE_PREFIX[self.target] + "-"):
            raise ValueError(f"line {self.line_id} is not in {self.target}")
        return self


class Tweak(BaseModel):
    """One change a reviewer describes, with the edits that express it."""

    id: str = Field(pattern=TWEAK_ID)
    review_id: str = Field(pattern=REVIEW_ID)
    modification_type: ModificationType
    reasoning: str
    # Not produced by the current extractor; None means "not determined"
    evidence_quote: Optional[str] = None
    status: Optional[Literal["tried", "suggested"]] = None
    reviewer_outcome: Optional[Literal["positive", "mixed", "negative", "unclear"]] = None
    edits: List[TweakEdit] = Field(default_factory=list)
    warnings: List[str] = Field(
        default_factory=list, description="Support checks that flagged but did not reject"
    )

    @model_validator(mode="after")
    def _consistent(self):
        if not self.id.startswith(self.review_id + ".t"):
            raise ValueError(f"tweak {self.id} does not belong to review {self.review_id}")
        if any(edit.tweak_id != self.id for edit in self.edits):
            raise ValueError(f"tweak {self.id} contains another tweak's edit")
        if len({edit.id for edit in self.edits}) != len(self.edits):
            raise ValueError(f"tweak {self.id} has duplicate edit IDs")
        return self

    def to_modification(self) -> "ModificationObject":
        """The tweak in the form the recipe modifier applies."""
        return ModificationObject(
            modification_type=self.modification_type,
            reasoning=self.reasoning,
            edits=[
                ModificationEdit(target=e.target, operation=e.operation, find=e.find,
                                 replace=e.replace, add=e.add)
                for e in self.edits
            ],
        )

    @classmethod
    def from_modification(
        cls, modification: "ModificationObject", review_id: str, number: int = 1
    ) -> "Tweak":
        """Give an extracted modification and its edits stable IDs."""
        tweak_id = f"{review_id}.t{number}"
        return cls(
            id=tweak_id,
            review_id=review_id,
            modification_type=modification.modification_type,
            reasoning=modification.reasoning,
            edits=[
                TweakEdit(
                    id=f"{tweak_id}.e{i}",
                    tweak_id=tweak_id,
                    target=edit.target,
                    operation=edit.operation,
                    find=edit.find,
                    replace=edit.replace,
                    add=edit.add,
                )
                for i, edit in enumerate(modification.edits, start=1)
            ],
        )


# --- Extraction ---------------------------------------------------------------------
#
# ExtractedTweak / ExtractedEdit describe the JSON the LLM must return. They are
# deliberately loose about content: tweak_extractor.py checks content (quotes, line
# references, support) itself so it can tell the model exactly what to correct.


class ExtractedEdit(BaseModel):
    """An edit as returned by the LLM, before validation."""

    target: Literal["ingredients", "instructions"]
    operation: Literal["replace", "add_after", "remove"]
    line_id: str
    find: str
    replace: Optional[str] = None
    add: Optional[str] = None


class ExtractedTweak(BaseModel):
    """A tweak as returned by the LLM, before validation."""

    modification_type: ModificationType
    quote: str
    status: Literal["tried", "suggested"]
    reviewer_outcome: Literal["positive", "mixed", "negative", "unclear"]
    reasoning: str
    edits: List[ExtractedEdit]


class RejectedTweak(BaseModel):
    """A proposed tweak that failed validation, kept for the record."""

    quote: str = ""
    problems: List[str] = Field(min_length=1)
    proposed: Dict[str, Any] = Field(description="The tweak exactly as the LLM returned it")


class ExtractionResult(BaseModel):
    """
    Result of extracting one review.

    ok:     at least one valid tweak (some proposals may also have been rejected)
    empty:  the model found no tweaks, a valid answer
    failed: no usable answer, or every proposed tweak was rejected
    """

    review_id: str = Field(pattern=REVIEW_ID)
    status: Literal["ok", "empty", "failed"]
    tweaks: List[Tweak] = Field(default_factory=list)
    rejected: List[RejectedTweak] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list, description="Response-level problems")
    attempts: int = Field(ge=1, le=2)

    @model_validator(mode="after")
    def _consistent(self):
        if any(t.review_id != self.review_id for t in self.tweaks):
            raise ValueError("extraction result contains another review's tweak")
        if self.status == "ok" and not self.tweaks:
            raise ValueError("an ok result needs at least one tweak")
        if self.status == "empty" and (self.tweaks or self.rejected or self.errors):
            raise ValueError("an empty result has no tweaks, rejections or errors")
        if self.status == "failed" and (self.tweaks or not (self.errors or self.rejected)):
            raise ValueError("a failed result has no tweaks and says why it failed")
        return self


EditRefusal = Literal[
    "not_found",   # the expected text is not on the line in the original recipe
    "ambiguous",   # the expected text occurs more than once on the line
    "stale",       # an earlier tweak of the same review changed or removed the target
    "conflict",    # a higher-ranked review's tweak already changed or removed the target
    "duplicate",   # the same change has already been made
    "no_op",       # the edit would not change the text
    "invalid",     # the edit cannot be applied as written (e.g. no such line)
]
TweakSkip = Literal[
    "not_selected", "not_tried", "outcome_not_positive", "no_edits", "evidence_not_in_review"
]


class EditOutcome(BaseModel):
    """What happened to one edit."""

    edit_id: str = Field(pattern=EDIT_ID)
    status: Literal["applied", "refused", "rolled_back"]
    reason: Optional[EditRefusal] = None
    detail: str = ""
    line_id: Optional[str] = Field(
        default=None, pattern=LINE_ID, description="Line in the original recipe it resolved to"
    )
    before: Optional[str] = Field(default=None, description="Line text before (None for an add)")
    after: Optional[str] = Field(default=None, description="Line text after (None for a remove)")
    match_method: Optional[Literal["exact", "normalized", "fuzzy"]] = None
    match_score: Optional[float] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _consistent(self):
        if self.status == "refused" and self.reason is None:
            raise ValueError("a refused edit needs a reason")
        if self.status != "refused" and self.reason is not None:
            raise ValueError(f"a {self.status} edit has no refusal reason")
        if self.status == "applied" and self.before == self.after:
            raise ValueError("an applied edit must change the text")
        return self


class TweakOutcome(BaseModel):
    """What happened to one tweak: applied whole, refused whole, or not attempted."""

    tweak_id: str = Field(pattern=TWEAK_ID)
    status: Literal["applied", "refused", "skipped"]
    reason: Optional[TweakSkip] = Field(
        default=None, description="Why a skipped tweak was not attempted"
    )
    detail: str = ""
    edit_outcomes: List[EditOutcome] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self):
        if any(not o.edit_id.startswith(self.tweak_id + ".e") for o in self.edit_outcomes):
            raise ValueError(f"outcome for tweak {self.tweak_id} contains another tweak's edit")
        statuses = {o.status for o in self.edit_outcomes}
        if self.status == "applied" and (not statuses or statuses != {"applied"}):
            raise ValueError("an applied tweak needs every edit applied")
        if self.status == "refused" and ("refused" not in statuses or "applied" in statuses):
            raise ValueError("a refused tweak needs a refused edit and no applied edits")
        if self.status == "skipped" and (self.reason is None or self.edit_outcomes):
            raise ValueError("a skipped tweak needs a reason and no edit outcomes")
        if self.status != "skipped" and self.reason is not None:
            raise ValueError("only a skipped tweak has a skip reason")
        return self


# EnhancedRecipe refers to ExtractionResult, which is defined after it
EnhancedRecipe.model_rebuild()
RecipeRun.model_rebuild()
