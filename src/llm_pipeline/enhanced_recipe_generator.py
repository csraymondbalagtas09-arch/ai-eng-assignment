"""
Step 3: Enhanced Recipe Generation with Attribution

This module generates enhanced recipes with full citation tracking.
It combines modified recipes with attribution information to create
comprehensive enhanced recipe objects.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .models import (
    ChangeRecord,
    EnhancedRecipe,
    EnhancementSummary,
    ExtractionResult,
    ModificationApplied,
    ModificationObject,
    Recipe,
    Review,
    SourceReview,
    Tweak,
    TweakOutcome,
)
from .report import attribute_diff, compute_diff, render_markdown


class EnhancedRecipeGenerator:
    """Generates enhanced recipes with full citation tracking and attribution."""

    def __init__(self, pipeline_version: str = "1.0.0"):
        """
        Initialize the Enhanced Recipe Generator.

        Args:
            pipeline_version: Version identifier for the pipeline
        """
        self.pipeline_version = pipeline_version
        logger.info(f"Initialized EnhancedRecipeGenerator v{pipeline_version}")

    def create_source_review(self, review: Review) -> SourceReview:
        """
        Convert a Review object to a SourceReview for attribution.

        Args:
            review: Original review object

        Returns:
            SourceReview with attribution information
        """
        return SourceReview(
            text=review.text,
            reviewer=review.username,
            rating=review.rating,
            review_id=review.id,
            sources=list(review.sources),
            featured=review.featured,
            rank=review.rank,
        )

    def create_modification_applied(
        self,
        modification: ModificationObject,
        source_review: Review,
        change_records: List[ChangeRecord],
        tweak_id: Optional[str] = None,
    ) -> ModificationApplied:
        """
        Create a ModificationApplied record for attribution.

        Args:
            modification: Original modification object
            source_review: Review that suggested this modification
            change_records: List of changes that were actually made
            tweak_id: Stable ID of the tweak this modification was recorded as

        Returns:
            ModificationApplied with full attribution
        """
        return ModificationApplied(
            source_review=self.create_source_review(source_review),
            modification_type=modification.modification_type,
            reasoning=modification.reasoning,
            changes_made=change_records,
            tweak_id=tweak_id,
        )

    def calculate_enhancement_summary(
        self, modifications_applied: List[ModificationApplied]
    ) -> EnhancementSummary:
        """
        Calculate summary statistics for all applied modifications.

        Args:
            modifications_applied: List of all modifications applied

        Returns:
            EnhancementSummary with aggregate statistics
        """
        total_changes = sum(len(mod.changes_made) for mod in modifications_applied)
        change_types = sorted(set(mod.modification_type for mod in modifications_applied))

        # Generate expected impact summary
        impact_descriptions = []
        for mod in modifications_applied:
            if mod.reasoning:
                impact_descriptions.append(mod.reasoning)

        expected_impact = "; ".join(impact_descriptions[:3])  # Limit to top 3
        if len(impact_descriptions) > 3:
            expected_impact += (
                f" (and {len(impact_descriptions) - 3} more improvements)"
            )

        return EnhancementSummary(
            total_changes=total_changes,
            change_types=change_types,
            expected_impact=expected_impact
            or "Community-validated recipe improvements",
        )

    def generate_enhanced_recipe(
        self,
        original_recipe: Recipe,
        modified_recipe: Recipe,
        modification: ModificationObject,
        source_review: Review,
        change_records: List[ChangeRecord],
        tweak_id: Optional[str] = None,
    ) -> EnhancedRecipe:
        """
        Generate a complete enhanced recipe with attribution.

        Args:
            original_recipe: Original unmodified recipe
            modified_recipe: Recipe with modifications applied
            modification: Single modification that was applied
            source_review: Review that suggested the modification
            change_records: Changes made for the modification
            tweak_id: Stable ID of the tweak this modification was recorded as

        Returns:
            Complete EnhancedRecipe with attribution
        """
        # Create modification applied record
        modification_applied = self.create_modification_applied(
            modification, source_review, change_records, tweak_id=tweak_id
        )
        return self._assemble(original_recipe, modified_recipe, [modification_applied])

    def generate_from_tweaks(
        self,
        original_recipe: Recipe,
        modified_recipe: Recipe,
        applied: List[Tuple[Tweak, Review, List[ChangeRecord]]],
        extraction_results: List[ExtractionResult],
        tweak_outcomes: List[TweakOutcome],
        reviews: Optional[List[Review]] = None,
    ) -> EnhancedRecipe:
        """
        Generate an enhanced recipe from applied tweaks, recording every extraction.

        Args:
            original_recipe: Original unmodified recipe
            modified_recipe: Recipe with the tweaks applied (the original if none applied)
            applied: Each applied tweak with its review and the changes it made
            extraction_results: Extraction results for every review analyzed
            tweak_outcomes: Outcome of every extracted tweak (applied, refused, skipped)
            reviews: Every review considered, in rank order

        Returns:
            Complete EnhancedRecipe with attribution and a verified diff
        """
        modifications_applied = []
        for tweak, review, records in applied:
            record = self.create_modification_applied(
                tweak.to_modification(), review, records, tweak_id=tweak.id
            )
            record.evidence_quote = tweak.evidence_quote
            modifications_applied.append(record)
        enhanced = self._assemble(
            original_recipe, modified_recipe, modifications_applied, extraction_results,
            tweak_outcomes,
        )
        enhanced.reviews = [self.create_source_review(r) for r in reviews or []]
        diff, problems = attribute_diff(compute_diff(original_recipe, modified_recipe), tweak_outcomes)
        enhanced.recipe_diff, enhanced.diff_problems = diff, problems
        enhanced.diff_verified = not problems
        if problems:
            logger.error(f"Diff does not reconcile with the applied edits: {problems}")
        return enhanced

    def _assemble(
        self,
        original_recipe: Recipe,
        modified_recipe: Recipe,
        modifications_applied: List[ModificationApplied],
        extraction_results: Optional[List[ExtractionResult]] = None,
        tweak_outcomes: Optional[List[TweakOutcome]] = None,
    ) -> EnhancedRecipe:
        logger.info(f"Generating enhanced recipe for: {original_recipe.title}")

        # Calculate enhancement summary
        enhancement_summary = self.calculate_enhancement_summary(modifications_applied)

        # Generate enhanced recipe ID and title; only claim enhancement if text changed
        changed = enhancement_summary.total_changes > 0
        enhanced_recipe_id = f"{original_recipe.recipe_id}_enhanced"
        enhanced_title = original_recipe.title + (" (Community Enhanced)" if changed else "")

        # Create the enhanced recipe
        enhanced_recipe = EnhancedRecipe(
            recipe_id=enhanced_recipe_id,
            original_recipe_id=original_recipe.recipe_id,
            title=enhanced_title,
            ingredients=modified_recipe.ingredients,
            instructions=modified_recipe.instructions,
            modifications_applied=modifications_applied,
            enhancement_summary=enhancement_summary,
            description=original_recipe.description,
            servings=original_recipe.servings,
            prep_time=getattr(original_recipe, "prep_time", None),
            cook_time=getattr(original_recipe, "cook_time", None),
            total_time=getattr(original_recipe, "total_time", None),
            created_at=datetime.now().isoformat(),
            pipeline_version=self.pipeline_version,
            source_recipe_hash=original_recipe.content_hash,
            extraction_results=extraction_results or [],
            tweak_outcomes=tweak_outcomes or [],
            status="enhanced" if changed else "unchanged",
        )

        logger.info(
            f"Generated enhanced recipe with {enhancement_summary.total_changes} changes "
            f"from {len(modifications_applied)} modifications"
        )

        return enhanced_recipe

    def generate_comparison_data(
        self, original_recipe: Recipe, enhanced_recipe: EnhancedRecipe
    ) -> Dict[str, Any]:
        """
        Generate side-by-side comparison data for UI display.

        Args:
            original_recipe: Original recipe
            enhanced_recipe: Enhanced recipe

        Returns:
            Dictionary with comparison data
        """
        comparison = {
            "original": {
                "title": original_recipe.title,
                "ingredients": original_recipe.ingredients,
                "instructions": original_recipe.instructions,
                "servings": original_recipe.servings,
            },
            "enhanced": {
                "title": enhanced_recipe.title,
                "ingredients": enhanced_recipe.ingredients,
                "instructions": enhanced_recipe.instructions,
                "servings": enhanced_recipe.servings,
            },
            "changes": {
                "total_modifications": len(enhanced_recipe.modifications_applied),
                "total_changes": enhanced_recipe.enhancement_summary.total_changes,
                "change_types": enhanced_recipe.enhancement_summary.change_types,
                "expected_impact": enhanced_recipe.enhancement_summary.expected_impact,
            },
            "citations": [
                {
                    "reviewer": mod.source_review.reviewer,
                    "rating": mod.source_review.rating,
                    "modification_type": mod.modification_type,
                    "reasoning": mod.reasoning,
                    "changes": [
                        {
                            "type": change.type,
                            "from": change.from_text,
                            "to": change.to_text,
                            "operation": change.operation,
                        }
                        for change in mod.changes_made
                    ],
                }
                for mod in enhanced_recipe.modifications_applied
            ],
        }

        return comparison

    def save_enhanced_recipe(
        self, enhanced_recipe: EnhancedRecipe, output_path: str
    ) -> str:
        """
        Save enhanced recipe to JSON file.

        Args:
            enhanced_recipe: Enhanced recipe to save
            output_path: Path to save the file

        Returns:
            Path to the saved file
        """
        import json
        import os

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to dict and save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_recipe.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved enhanced recipe to: {output_path}")
        return output_path

    def save_report(self, enhanced_recipe: EnhancedRecipe, json_path: str) -> List[str]:
        """
        Save the enhanced recipe as JSON and its explanation as Markdown alongside it.

        Returns:
            Paths of the two files written
        """
        md_path = os.path.splitext(json_path)[0] + ".md"
        self.save_enhanced_recipe(enhanced_recipe, json_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_markdown(enhanced_recipe))
        logger.info(f"Saved explanation to: {md_path}")
        return [json_path, md_path]
