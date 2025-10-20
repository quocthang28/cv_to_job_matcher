"""Tool for retrieving CVs from the mock backend store and summarizing them."""

from typing import Dict

from langchain_core.tools import tool

from ..config import get_chat_model
from ..mock_backend import fetch_mock_user_cv

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

    try:
        payload = fetch_mock_user_cv(user_id)
    except ValueError:
        return {
            "message": "Please provide your user id so I can retrieve your resume.",
            "user_id": "",
            "cv_text": "",
        }
    except KeyError:
        return {
            "message": f"I couldn't find a saved CV for user '{user_id}'.",
            "user_id": user_id,
            "cv_text": "",
        }
    except FileNotFoundError as err:
        return {
            "message": f"The configured CV file is missing: {err}.",
            "user_id": user_id,
            "cv_text": "",
        }
    except RuntimeError as err:
        return {
            "message": f"Unable to read the configured CV: {err}.",
            "user_id": user_id,
            "cv_text": "",
        }

    cv_text = payload.get("cv_text") or ""
    if not cv_text.strip():
        return {
            "message": "The stored CV did not contain any text. Please upload a fresh copy.",
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
