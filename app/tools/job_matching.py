"""Tools for matching CVs to job descriptions."""

from langchain_core.tools import tool

from ..semantic_search import match_jobs

__all__ = ["find_relevant_jobs_tool"]


@tool("find_relevant_jobs_tool")
def find_relevant_jobs_tool(cv_text: str) -> str:
    """Use ONLY when the user provides a CV text or asks for job matches."""
    try:
        matches = match_jobs(cv_text)
    except Exception as err:  # Broad catch to ensure the agent sees the failure.
        return f"Error while matching CV to jobs: {err}"
    if not matches:
        return "No matching jobs found for the provided CV."

    lines = []
    for idx, match in enumerate(matches, 1):
        similarity = match.get("similarity_score")
        if isinstance(similarity, (int, float)):
            similarity_display = f"{similarity:.2%}"
        else:
            similarity_display = "N/A"
        title = match.get("title", "Unknown Title")
        company = match.get("company", "Unknown Company")
        location = match.get("location", "Unknown Location")
        job_id = match.get("job_id", "N/A")
        lines.append(
            f"{idx}. {title} at {company} (Similarity: {similarity_display})\n"
            f"   Location: {location} | Job ID: {job_id}"
        )
    return "\n".join(lines)
