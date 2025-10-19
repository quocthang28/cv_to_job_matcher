"""Tool for routing chat requests to the correct capability."""

from langchain_core.tools import tool

from ..config import get_intent_classifier_model

__all__ = ["classify_tool_intent_tool"]


@tool("classify_tool_intent")
def classify_tool_intent_tool(query: str) -> str:
    """Inspect the latest user text and return the tool name to execute next."""
    if not query:
        return ""

    system_message = (
        "You are an intent classifier for a chatbot assistant. Respond with exactly one of:\n"
        "1. upload_cv_tool\n"
        "2. find_relevant_jobs_tool\n"
        "3. review_cv_tool\n"
        "4. '' (empty string)\n\n"
        "Return 'upload_cv_tool' when the user asks to load their uploaded CV, references their account/user id, "
        "or requests that the assistant remember their resume. "
        "Return 'find_relevant_jobs_tool' only when the user is sharing CV/resume text, "
        "asking for job matching, or seeking job recommendations. Return 'review_cv_tool' "
        "when the user is asking for CV/resume feedback, critique, or improvements. "
        "Return '' for general chit-chat, greetings, or anything unrelated to job matching "
        "or resume reviews."
    )
    model = get_intent_classifier_model()
    result = model.invoke(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]
    )
    if hasattr(result, "content"):
        intent = str(result.content or "").strip()
    else:
        intent = str(result).strip()
    if intent not in ("upload_cv_tool", "find_relevant_jobs_tool", "review_cv_tool", ""):
        return ""
    return intent
