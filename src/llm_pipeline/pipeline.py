"""
LLM Analysis Pipeline - Main Orchestrator

This module coordinates the complete pipeline for one recipe file:

0. Load       validate and normalize the recipe; deduplicate and rank reviews (loader.py)
1. Select     reviews to analyse, in rank order (optionally capped)
2. Extract    every tweak from each review, validated against review and recipe
              (tweak_extractor.py, support.py)
3. Apply      tweaks the reviewer tried and liked, all-or-nothing, in review rank
              order (recipe_modifier.py)
4. Explain    JSON and Markdown with a verified diff, attribution and every outcome
              (enhanced_recipe_generator.py, report.py)

Every run ends in an explicit RecipeRun: enhanced, unchanged (report still written),
skipped (no reviews) or failed (with the reason). Nothing fails silently.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger

from .enhanced_recipe_generator import EnhancedRecipeGenerator
from .loader import RecipeDataError, load_recipe_file, parse_recipe, parse_reviews
from .models import EnhancedRecipe, Recipe, RecipeRun, Review, Tweak, TweakOutcome
from .recipe_modifier import RecipeModifier, change_records
from .tweak_extractor import DEFAULT_SDK_RETRIES, DEFAULT_TIMEOUT_SECONDS, TweakExtractor


def skip_reason(tweak: Tweak) -> Optional[str]:
    """Why the agreed policy does not apply a tweak, or None if it does."""
    if tweak.status != "tried":
        return "not_tried"
    if tweak.reviewer_outcome != "positive":
        return "outcome_not_positive"
    if not tweak.edits:
        return "no_edits"
    return None


def is_applicable(tweak: Tweak) -> bool:
    """The agreed policy: apply only tweaks the reviewer made and reported positively."""
    return skip_reason(tweak) is None


class LLMAnalysisPipeline:
    """Complete pipeline for analyzing recipes and generating enhanced versions."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        output_dir: str = "data/enhanced",
        pipeline_version: str = "2.0.0",
        llm_client=None,
        model: str = "gpt-3.5-turbo",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_SDK_RETRIES,
        max_reviews: Optional[int] = None,
    ):
        """
        Initialize the complete LLM Analysis Pipeline.

        Args:
            openai_api_key: OpenAI API key (loads from env if not provided)
            output_dir: Directory to save enhanced recipes
            pipeline_version: Version identifier for tracking
            llm_client: OpenAI-compatible client to use instead of building one
            model: Model name for extraction
            timeout: Seconds before a single LLM request is abandoned
            max_retries: SDK retries per LLM request for transient errors
            max_reviews: Analyse at most this many reviews per recipe (best-ranked first)

        Raises:
            LLMConfigurationError: If no client is given and no API key is available
        """
        # Load environment variables
        load_dotenv()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_reviews = max_reviews

        # Initialize pipeline components
        self.tweak_extractor = TweakExtractor(
            api_key=openai_api_key, model=model, client=llm_client,
            timeout=timeout, max_retries=max_retries,
        )
        self.recipe_modifier = RecipeModifier()
        self.enhanced_generator = EnhancedRecipeGenerator(
            pipeline_version=pipeline_version
        )

        logger.info(f"Initialized LLM Analysis Pipeline v{pipeline_version}")
        logger.info(f"Output directory: {self.output_dir}")

    # --- Input --------------------------------------------------------------------

    def load_recipe_data(self, file_path: str) -> Dict[str, Any]:
        """Load raw recipe data from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def parse_recipe_data(self, recipe_data: Dict[str, Any]) -> Recipe:
        """Parse raw recipe data into a normalized Recipe with stable line IDs."""
        return parse_recipe(recipe_data)

    def parse_reviews_data(self, recipe_data: Dict[str, Any]) -> List[Review]:
        """Parse reviews and featured tweaks into unique Reviews, best-ranked first."""
        return parse_reviews(recipe_data)

    # --- Steps ---------------------------------------------------------------------

    def select_reviews(self, reviews: List[Review]) -> Tuple[List[Review], List[Review]]:
        """Step 1: reviews to analyse in rank order, and those left out by max_reviews."""
        ranked = sorted(reviews, key=lambda r: r.rank or 0)
        if self.max_reviews is None:
            return ranked, []
        return ranked[: self.max_reviews], ranked[self.max_reviews:]

    @staticmethod
    def plan_application(tweaks: List[Tweak]) -> Tuple[List[Tweak], List[TweakOutcome]]:
        """Split tweaks into those to apply and skip outcomes (with the policy reason)."""
        applicable = [t for t in tweaks if is_applicable(t)]
        skipped = [TweakOutcome(tweak_id=t.id, status="skipped", reason=skip_reason(t))
                   for t in tweaks if not is_applicable(t)]
        return applicable, skipped

    def run_recipe(self, recipe_file: str, save_output: bool = True) -> RecipeRun:
        """
        Run the complete pipeline on one recipe file.

        Returns:
            RecipeRun whose status says what happened; never raises for bad input
            or model failures
        """
        logger.info(f"Processing recipe file: {recipe_file}")
        try:
            # Step 0: Load, validate and normalize
            try:
                loaded = load_recipe_file(recipe_file)
            except RecipeDataError as e:
                logger.error(f"Invalid recipe file {recipe_file}: {e}")
                return RecipeRun(recipe_file=recipe_file, status="failed", reason="invalid_input", detail=str(e))
            recipe = loaded.recipe
            run = dict(recipe_file=recipe_file, recipe_id=recipe.recipe_id)
            logger.info(f"Loaded recipe: {recipe.title} ({len(loaded.reviews)} unique reviews)")

            # Step 1: Select
            selected, left_out = self.select_reviews(loaded.reviews)
            if not selected:
                logger.warning("No reviews found")
                return RecipeRun(**run, status="skipped", reason="no_reviews")
            if left_out:
                logger.info(f"Analysing the top {len(selected)} reviews; {len(left_out)} left out")

            # Step 2: Extract and validate
            logger.info(f"Step 2: Extracting tweaks from {len(selected)} reviews...")
            results = [result for _, result in self.tweak_extractor.extract_all(selected, recipe)]
            if all(r.status == "failed" for r in results):
                errors = sorted({e for r in results for e in r.errors} |
                                {p for r in results for rej in r.rejected for p in rej.problems})
                logger.error(f"Extraction failed for every review: {errors}")
                return RecipeRun(**run, status="failed", reason="extraction_failed", detail="; ".join(errors))

            # Step 3: Apply in review rank order; higher-ranked reviews win conflicts
            tweaks = [t for r in results for t in r.tweaks]
            applicable, skipped = self.plan_application(tweaks)
            logger.info(f"Step 3: Applying {len(applicable)} of {len(tweaks)} extracted tweaks...")
            modified, outcomes = self.recipe_modifier.apply_tweaks(
                recipe, applicable, {r.id: r.rank for r in selected}
            )
            by_id = {t.id: t for t in tweaks}
            review_of = {r.id: r for r in selected}
            applied = [(by_id[o.tweak_id], review_of[by_id[o.tweak_id].review_id], change_records(o))
                       for o in outcomes if o.status == "applied"]

            # Step 4: Explain
            enhanced = self.enhanced_generator.generate_from_tweaks(
                recipe, modified, applied, results, outcomes + skipped, reviews=loaded.reviews
            )
            enhanced.extraction_source = self.tweak_extractor.source
            enhanced.model = self.tweak_extractor.model
            enhanced.prompt_version = self.tweak_extractor.prompt_version
            outputs = []
            if save_output:
                slug = recipe.title.lower().replace(" ", "-")[:30]
                outputs = self.enhanced_generator.save_report(
                    enhanced, str(self.output_dir / f"enhanced_{recipe.recipe_id}_{slug}.json")
                )

            if applied:
                status, reason = "enhanced", None
            else:
                status = "unchanged"
                reason = "nothing_applicable" if not applicable else "nothing_applied"
            logger.info(f"Recipe {recipe.recipe_id}: {status}"
                        f"{f' ({reason})' if reason else ''}, {len(applied)} tweaks applied")
            return RecipeRun(**run, status=status, reason=reason, enhanced=enhanced, outputs=outputs)

        except Exception as e:
            logger.exception(f"Failed to process recipe {recipe_file}")
            return RecipeRun(recipe_file=recipe_file, status="failed", reason="error",
                             detail=f"{type(e).__name__}: {e}")

    def process_single_recipe(
        self, recipe_file: str, save_output: bool = True
    ) -> Optional[EnhancedRecipe]:
        """
        Process a single recipe through the complete pipeline.

        Returns:
            The EnhancedRecipe (also when nothing could be applied), or None if the
            recipe was skipped or failed; use run_recipe() for the reason
        """
        return self.run_recipe(recipe_file, save_output).enhanced

    def run_directory(self, data_dir: str = "data") -> List[RecipeRun]:
        """Run every recipe_*.json file in a directory, in file-name order."""
        recipe_files = sorted(Path(data_dir).glob("recipe_*.json"))
        logger.info(f"Found {len(recipe_files)} recipe files to process")
        runs = [self.run_recipe(str(f)) for f in recipe_files]
        for run in runs:
            logger.info(f"{Path(run.recipe_file).name}: {run.status}"
                        f"{f' ({run.reason})' if run.reason else ''}")
        return runs

    def process_recipe_directory(self, data_dir: str = "data") -> List[EnhancedRecipe]:
        """Process all recipe files; returns the recipes that produced a report."""
        return [run.enhanced for run in self.run_directory(data_dir) if run.enhanced]

    # --- Summary -------------------------------------------------------------------

    def generate_run_summary(self, runs: List[RecipeRun]) -> Dict[str, Any]:
        """Summary of a directory run, including recipes that were skipped or failed."""
        def count(status, recipe):
            return sum(o.status == status for o in recipe.tweak_outcomes) if recipe else 0

        return {
            "pipeline_summary": {
                "recipes": len(runs),
                "by_status": {s: sum(r.status == s for r in runs)
                              for s in ("enhanced", "unchanged", "skipped", "failed")},
                "tweaks_applied": sum(count("applied", r.enhanced) for r in runs),
                "tweaks_refused": sum(count("refused", r.enhanced) for r in runs),
                "tweaks_skipped": sum(count("skipped", r.enhanced) for r in runs),
                "diffs_verified": sum(bool(r.enhanced and r.enhanced.diff_verified) for r in runs),
            },
            "recipes": [
                {
                    "recipe_file": Path(r.recipe_file).name,
                    "recipe_id": r.recipe_id,
                    "status": r.status,
                    "reason": r.reason,
                    "detail": r.detail,
                    "tweaks_applied": count("applied", r.enhanced),
                    "tweaks_refused": count("refused", r.enhanced),
                    "tweaks_skipped": count("skipped", r.enhanced),
                    "line_changes": len(r.enhanced.recipe_diff) if r.enhanced else 0,
                    "diff_verified": r.enhanced.diff_verified if r.enhanced else None,
                    "outputs": [Path(o).name for o in r.outputs],
                }
                for r in runs
            ],
        }

    def save_run_summary(self, runs: List[RecipeRun], output_path: Optional[str] = None) -> str:
        """Save the directory-run summary to JSON."""
        output_path = output_path or str(self.output_dir / "pipeline_summary_report.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.generate_run_summary(runs), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved pipeline summary report to: {output_path}")
        return output_path

    def generate_summary_report(self, enhanced_recipes: List[EnhancedRecipe]) -> Dict[str, Any]:
        """Summary of enhanced recipes (kept for compatibility; prefer generate_run_summary)."""
        if not enhanced_recipes:
            return {"status": "no_recipes_processed"}
        change_type_counts: Dict[str, int] = {}
        for recipe in enhanced_recipes:
            for change_type in recipe.enhancement_summary.change_types:
                change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
        return {
            "pipeline_summary": {
                "recipes_processed": len(enhanced_recipes),
                "total_modifications_applied": sum(len(r.modifications_applied) for r in enhanced_recipes),
                "total_changes_made": sum(r.enhancement_summary.total_changes for r in enhanced_recipes),
                "change_type_distribution": change_type_counts,
            },
            "enhanced_recipes": [
                {
                    "recipe_id": r.recipe_id,
                    "title": r.title,
                    "modifications_count": len(r.modifications_applied),
                    "changes_count": r.enhancement_summary.total_changes,
                    "change_types": r.enhancement_summary.change_types,
                }
                for r in enhanced_recipes
            ],
        }

    def save_summary_report(
        self, enhanced_recipes: List[EnhancedRecipe], output_path: Optional[str] = None
    ) -> str:
        """Save a summary of enhanced recipes (kept for compatibility)."""
        output_path = output_path or str(self.output_dir / "pipeline_summary_report.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.generate_summary_report(enhanced_recipes), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved pipeline summary report to: {output_path}")
        return output_path
