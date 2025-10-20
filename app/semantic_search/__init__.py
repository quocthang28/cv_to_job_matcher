"""Semantic search utilities for matching CVs to job descriptions."""

from .matcher import (
    CVJobMatcher,
    ingest_job_content,
    init_matcher,
    jobs_indexed,
    matcher_ready,
    match_jobs,
)

__all__ = [
    "CVJobMatcher",
    "init_matcher",
    "match_jobs",
    "matcher_ready",
    "jobs_indexed",
    "ingest_job_content",
]
