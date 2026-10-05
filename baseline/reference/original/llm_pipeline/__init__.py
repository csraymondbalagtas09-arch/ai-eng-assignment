"""
LLM Analysis Pipeline for Recipe Enhancement

This package provides a multi-step pipeline for analyzing community recipe modifications
and generating enhanced recipes with full citation tracking.

Pipeline Steps:
1. Tweak Extraction: Parse review text into structured modification objects
2. Recipe Modification: Apply modifications using search-and-replace
3. Enhanced Recipe Generation: Create enhanced recipes with attribution
"""

from .models import (
    ModificationEdit,
    ModificationObject,
    EnhancedRecipe,
    ModificationApplied,
    EnhancementSummary,
)
from .tweak_extractor import TweakExtractor
from .recipe_modifier import RecipeModifier
from .enhanced_recipe_generator import EnhancedRecipeGenerator
from .pipeline import LLMAnalysisPipeline

__all__ = [
    "ModificationEdit",
    "ModificationObject",
    "EnhancedRecipe",
    "ModificationApplied",
    "EnhancementSummary",
    "TweakExtractor",
    "RecipeModifier",
    "EnhancedRecipeGenerator",
    "LLMAnalysisPipeline",
]