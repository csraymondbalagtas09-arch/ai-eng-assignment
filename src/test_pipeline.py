#!/usr/bin/env python3
"""
LLM Analysis Pipeline Test Script

Runs the complete pipeline on one recipe or on every recipe in the data directory,
writing JSON and Markdown reports.

Usage (from src/):
    python test_pipeline.py single                 # chocolate chip cookies, live model
    python test_pipeline.py all                    # every recipe, live model
    python test_pipeline.py single --replay ../baseline/recordings/recipe_10813.json
    python test_pipeline.py all --replay ../baseline/recordings/*.json

--replay answers from recorded responses instead of calling a model (no API key
needed); a review without a recording is reported as a failed extraction.
"""

import argparse
import sys

from dotenv import load_dotenv
from loguru import logger

from llm_pipeline.pipeline import LLMAnalysisPipeline
from llm_pipeline.replay import ReplayClient
from llm_pipeline.tweak_extractor import LLMConfigurationError

# Load environment variables from .env file
load_dotenv()

DEFAULT_RECIPE = "../data/recipe_10813_best-chocolate-chip-cookies.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the recipe enhancement pipeline")
    parser.add_argument("mode", choices=["single", "all"])
    parser.add_argument("--recipe", default=DEFAULT_RECIPE, help="recipe file for 'single'")
    parser.add_argument("--data-dir", default="../data", help="recipe directory for 'all'")
    parser.add_argument("--out", default="data/enhanced", help="output directory")
    parser.add_argument("--replay", nargs="+", metavar="FILE", help="recorded responses instead of a live model")
    parser.add_argument("--max-reviews", type=int, help="analyse at most this many reviews per recipe")
    args = parser.parse_args()

    try:
        pipeline = LLMAnalysisPipeline(
            output_dir=args.out,
            llm_client=ReplayClient(*args.replay) if args.replay else None,
            max_reviews=args.max_reviews,
        )
    except LLMConfigurationError as e:
        logger.error(f"{e}. Use --replay to run from recorded responses.")
        return 2

    runs = [pipeline.run_recipe(args.recipe)] if args.mode == "single" else pipeline.run_directory(args.data_dir)
    if args.mode == "all":
        logger.info(f"Summary report: {pipeline.save_run_summary(runs)}")

    logger.info("=" * 60)
    for run in runs:
        line = f"{run.recipe_file}: {run.status}" + (f" ({run.reason})" if run.reason else "")
        if run.enhanced:
            counts = {s: sum(o.status == s for o in run.enhanced.tweak_outcomes) for s in ("applied", "refused", "skipped")}
            line += (f" | tweaks applied {counts['applied']}, refused {counts['refused']}, skipped {counts['skipped']}"
                     f" | {len(run.enhanced.recipe_diff)} line changes, diff verified: {run.enhanced.diff_verified}")
        if run.detail:
            line += f" | {run.detail}"
        (logger.error if run.status == "failed" else logger.info)(line)
        for path in run.outputs:
            logger.info(f"  wrote {path}")
    return 1 if any(r.status == "failed" for r in runs) else 0


if __name__ == "__main__":
    sys.exit(main())
