"""Tool for providing feedback on CV content."""

from langchain_core.tools import tool

from ..config import get_cv_reviewer_model

__all__ = ["review_cv_tool"]


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
        "Finally, suggest user if they want you to find some jobs suitable for their profile."
    )
    model = get_cv_reviewer_model()
    result = model.invoke(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": cv_text},
        ]
    )
    if hasattr(result, "content"):
        return str(result.content or "").strip()
    return str(result).strip()
