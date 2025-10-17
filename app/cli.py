"""Command-line entrypoint for matching a CV against job descriptions."""

import argparse
from pathlib import Path
from typing import Optional

from .assessment import assess_matches
from .jobs import load_job_descriptions, read_cv_text
from .matcher import init_matcher, match_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match a CV against job descriptions.")
    parser.add_argument(
        "--cv",
        type=str,
        required=True,
        help="Path to the CV file (text or Markdown).",
    )
    return parser.parse_args()


def run_cli(cv_path: Path, *, top_k: int = 5, model: Optional[str] = "gemini-2.0-flash-exp") -> None:
    jobs = load_job_descriptions()
    if not jobs:
        print("No job descriptions found in the jd directory. Please add Markdown job files and try again.")
        return

    cv_text = read_cv_text(cv_path)

    print("=" * 60)
    print("LangChain CV to Job Matcher")
    print("=" * 60)
    print()

    try:
        matcher = init_matcher(jobs)
    except ValueError as err:
        print(str(err))
        return

    if matcher.vectorstore is None:
        print("Matcher vector store is unavailable. Initialization failed.")
        return

    print("Vector store ready; using existing data.\n")

    print("Matching CV to jobs...")
    matches = match_jobs(cv_text, matcher=matcher, top_k=top_k)

    print("\n" + "=" * 60)
    print("TOP MATCHING JOBS")
    print("=" * 60)

    unique_matches = []
    seen_ids = set()
    for match in matches:
        if match["job_id"] in seen_ids:
            continue
        seen_ids.add(match["job_id"])
        unique_matches.append(match)

    for i, match in enumerate(unique_matches, 1):
        print(f"\n{i}. {match['title']} at {match['company']}")
        print(f"   Similarity Score: {match['similarity_score']:.2%}")
        print(f"   Job ID: {match['job_id']}")

    print("\n" + "=" * 60)
    print("ASK LLM FOR ASSESSMENT WITH PROVIDED INSTRUCTION")
    print("=" * 60)

    if unique_matches:
        assessment = assess_matches(cv_text, unique_matches, model=model)
        if assessment:
            print("\n" + "=" * 60)
            print("LLM ASSESSMENT")
            print("=" * 60 + "\n")
            print(assessment)
        else:
            print("\nSkipping LLM assessment (no response returned).")


def main() -> None:
    args = parse_args()
    run_cli(Path(args.cv))
