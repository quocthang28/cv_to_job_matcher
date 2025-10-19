"""Shared LangChain tool definitions for the chatbot."""

from typing import Any, Dict, List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .matcher import match_jobs

__all__ = [
    "find_relevant_jobs_tool",
    "review_cv_tool",
    "classify_tool_intent_tool",
]

_intent_classifier_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_cv_reviewer_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)


@tool("find_relevant_jobs_tool")
def find_relevant_jobs_tool(cv_text: str) -> str:
    """Use ONLY when the user provides a CV text or asks: 
'find matching jobs', 'recommend jobs', or 'compare my resume'. 
Do NOT use this tool for general job discussions or follow-ups."""
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


@tool("review_cv_tool")
def review_cv_tool(cv_text: str) -> str:
    """Provide professional resume feedback from an HR perspective."""
    if not cv_text or not cv_text.strip():
        return "Unable to review an empty CV. Please provide the resume text."

    system_message = (
        "You are an experienced HR professional. Review the provided CV text and "
        "give concise, actionable feedback. Highlight strengths, point out areas to "
        "improve (structure, clarity, impact, keywords), and suggest any tailoring "
        "ideas for job applications. Keep tone supportive and professional."
    )
    result = _cv_reviewer_model.invoke(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": cv_text},
        ]
    )
    if hasattr(result, "content"):
        return str(result.content or "").strip()
    return str(result).strip()


@tool("classify_tool_intent")
def classify_tool_intent_tool(query: str) -> str:
    """Inspect the latest user text and return the tool name to execute next.

    Return 'find_relevant_jobs_tool' when the user is sharing a CV/resume or
    asking for job matching. Return 'review_cv_tool' when the user wants resume
    feedback. Return an empty string when no tool is needed.
    """
    if not query:
        return ""

    system_message = (
        "You are an intent classifier for a chatbot assistant. Respond with exactly one of:\n"
        "1. find_relevant_jobs_tool\n"
        "2. review_cv_tool\n"
        "3. '' (empty string)\n\n"
        "Return 'find_relevant_jobs_tool' only when the user is sharing a CV/resume, "
        "asking for job matching, or seeking job recommendations. Return 'review_cv_tool' "
        "when the user is asking for CV/resume feedback, critique, or improvements. "
        "Return '' for general chit-chat, greetings, or anything unrelated to job matching "
        "or resume reviews."
    )
    result = _intent_classifier_model.invoke(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
    )
    intent = ""
    if hasattr(result, "content"):
        intent = str(result.content or "").strip()
    else:
        intent = str(result).strip()
    if intent not in ("find_relevant_jobs_tool", "review_cv_tool", ""):
        return ""
    return intent
