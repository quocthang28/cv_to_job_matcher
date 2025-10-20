"""Tools for matching CVs to job descriptions."""

import json

from langchain_core.tools import tool

from ..semantic_search import match_jobs

__all__ = ["find_relevant_jobs_tool"]


@tool("find_relevant_jobs_tool")
def find_relevant_jobs_tool(cv_text: str) -> str:
    """Use ONLY when the user provides a CV text or asks for job matches."""
    try:
        matches = match_jobs(cv_text)
    except Exception as err:  # Broad catch to ensure the agent sees the failure.
        payload = {
            "jobs": [],
            "error": f"Error while matching CV to jobs: {err}",
        }
        return json.dumps(payload)
    if not matches:
        return json.dumps({"jobs": []})

    jobs_payload = []
    for match in matches:
        job_id = match.get("job_id") or "N/A"
        title = match.get("title") or "Unknown Title"
        company = match.get("company") or "Unknown Company"
        location = match.get("location") or "Unknown Location"
        similarity = match.get("similarity_score")
        similarity_display = ""
        if isinstance(similarity, (int, float)):
            similarity_display = f" | Similarity: {similarity:.2%}"

        short_desc = f"{title} at {company} ({location}){similarity_display}"
        jobs_payload.append(
            {
                "job_id": job_id,
                "short_desc": short_desc,
                "metadata": {
                    "title": title,
                    "company": company,
                    "location": location,
                    "salary": match.get("salary") or "Not specified",
                    "similarity_score": similarity if isinstance(similarity, (int, float)) else None,
                    "source_excerpt": match.get("content"),
                },
            }
        )
    return json.dumps({"jobs": jobs_payload})
