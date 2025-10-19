"""Tool for retrieving CVs from the backend and summarizing them."""

from typing import Dict

import httpx
from langchain_core.tools import tool

from ..config import MOCK_BACKEND_URL, get_chat_model

__all__ = ["upload_cv_tool"]


def _summarize_cv_text(cv_text: str) -> str:
    """Generate a concise summary of the retrieved CV content."""
    summary_prompt = (
        "You are assisting a recruiter. Summarize the candidate's CV in three concise bullet "
        "points covering their primary role, years of experience, and standout skills. "
        "Keep the tone neutral and professional."
    )
    model = get_chat_model()
    try:
        response = model.invoke(
            [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": cv_text},
            ]
        )
    except Exception:
        snippet = cv_text.strip()[:400]
        return (
            "I retrieved your CV but couldn't generate an automatic summary right now. "
            "Here is an excerpt:\n"
            f"{snippet}..."
        )[:1000]

    if hasattr(response, "content"):
        summary_text = str(response.content or "").strip()
    else:
        summary_text = str(response).strip()

    if not summary_text:
        snippet = cv_text.strip()[:400]
        return (
            "I retrieved your CV but didn't receive a summary from the model. "
            "Here is an excerpt:\n"
            f"{snippet}..."
        )[:1000]
    return summary_text


@tool("upload_cv_tool")
def upload_cv_tool(user_id: str) -> Dict[str, str]:
    """Fetch a user's CV from the backend, summarize it, and suggest next steps."""
    if not user_id or not user_id.strip():
        return {
            "message": "Please provide your user id so I can retrieve your resume.",
            "user_id": "",
            "cv_text": "",
        }

    backend_base = MOCK_BACKEND_URL.rstrip("/")
    try:
        response = httpx.get(f"{backend_base}/users/{user_id}/cv", timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as err:
        return {
            "message": f"Backend returned an error while fetching the CV: {err}.",
            "user_id": user_id,
            "cv_text": "",
        }
    except httpx.HTTPError as err:
        return {
            "message": f"Unable to reach the backend service: {err}.",
            "user_id": user_id,
            "cv_text": "",
        }

    try:
        payload = response.json()
    except ValueError:
        return {
            "message": "Received an invalid response from the backend while retrieving the CV.",
            "user_id": user_id,
            "cv_text": "",
        }

    cv_text = payload.get("cv_text")
    if not isinstance(cv_text, str) or not cv_text.strip():
        return {
            "message": "The backend did not return any CV content. Please try uploading again.",
            "user_id": user_id,
            "cv_text": "",
        }

    summary = _summarize_cv_text(cv_text)
    message = (
        f"I've retrieved your CV from the backend for user '{user_id}'.\n\n"
        f"{summary}\n\nWould you like a CV review or should I look for relevant job matches next?"
    )
    return {
        "message": message.strip(),
        "user_id": user_id,
        "cv_text": cv_text,
    }
