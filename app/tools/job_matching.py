"""Tools for matching CVs to job descriptions."""

from typing import Any, Dict, List

from langchain_core.tools import tool

from ..semantic_search import match_jobs

__all__ = ["find_relevant_jobs_tool"]


@tool("find_relevant_jobs_tool")
def find_relevant_jobs_tool(cv_text: str) -> Dict[str, Any]:
    """Use ONLY when the user provides a CV text or asks for job matches."""
    try:
        matches = match_jobs(cv_text)
    except Exception as err:  # Broad catch to ensure the agent sees the failure.
        return {
            "jobs": [],
            "error": f"Error while matching CV to jobs: {err}",
        }

    if not matches:
        return {"jobs": []}

    jobs_payload: List[Dict[str, Any]] = []
    for match in matches:
        job_id = match.get("job_id") or "N/A"
        title = match.get("title") or "Unknown Title"
        company = match.get("company") or "Unknown Company"
        location = match.get("location") or "Unknown Location"
        short_desc = f"{title} at {company} ({location})"
        jobs_payload.append(
            {
                "job_id": job_id,
                "short_desc": short_desc,
                "metadata": {
                    "title": title,
                    "company": company,
                    "location": location,
                    "salary": match.get("salary") or "Not specified",
                },
            }
        )
    return {"jobs": jobs_payload}
