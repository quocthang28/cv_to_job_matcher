"""LangChain chatbot orchestration backed by the CV matcher tool."""

from types import SimpleNamespace
from typing import Any, Dict, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from .config import CHAT_SYSTEM_PROMPT, DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE
from .tools import (
    classify_tool_intent_tool,
    find_relevant_jobs_tool,
    review_cv_tool,
    upload_cv_tool,
)
from .mock_backend import get_mock_user_cv_text

_chat_histories: Dict[str, InMemoryChatMessageHistory] = {}
_user_cv_store: Dict[str, Optional[str]] = {}


def _get_chat_history(user_id: str) -> InMemoryChatMessageHistory:
    """Return per-user memory storage for the chatbot."""
    if user_id not in _chat_histories:
        _chat_histories[user_id] = InMemoryChatMessageHistory()
    return _chat_histories[user_id]


def _store_user_cv(user_id: str, cv_text: Optional[str]) -> None:
    """Persist the most recent CV associated with a user."""
    if not cv_text:
        return
    _user_cv_store[user_id] = cv_text


def _get_user_cv_text(user_id: str) -> Optional[str]:
    """Retrieve previously stored CV text for the user."""
    return _user_cv_store.get(user_id)


def _stringify_content(content: Any) -> str:
    """Normalize message content (string or content blocks) to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _classify_intent(user_input: str, *, previous_bot_message: Optional[str] = None) -> str:
    """Use the lightweight classifier tool to decide which branch to follow."""
    if not user_input:
        return ""
    query_text = user_input
    if previous_bot_message:
        query_text = f"Assistant: {previous_bot_message}\nUser: {user_input}"
    classification_result = classify_tool_intent_tool.invoke({"query": query_text})
    return str(classification_result or "").strip()


def _ensure_cv_text(source: Optional[str]) -> Optional[str]:
    """Return a sanitized CV text string or None when unusable."""
    if not source:
        return None
    text = source.strip()
    return text or None


def _fetch_backend_cv_text(user_id: str) -> Optional[str]:
    """Retrieve CV content for a user directly from the mock backend."""
    if not user_id:
        return None

    cv_text = get_mock_user_cv_text(user_id)
    return _ensure_cv_text(cv_text)

def _handle_job_search(
    *,
    history: InMemoryChatMessageHistory,
    user_id: str,
    cv_text: Optional[str],
    user_input: str,
) -> Dict[str, Any]:
    """Invoke the matching tool when a CV is present."""
    tool_source = _ensure_cv_text(cv_text)
    used_stored_cv = False
    used_backend_cv = False
    if tool_source is None:
        stored_cv = _ensure_cv_text(_get_user_cv_text(user_id))
        if stored_cv is not None:
            tool_source = stored_cv
            used_stored_cv = True
    if tool_source is None:
        backend_cv = _fetch_backend_cv_text(user_id)
        if backend_cv is not None:
            tool_source = backend_cv
            used_backend_cv = True
            _store_user_cv(user_id, backend_cv)
    if tool_source is None:
        tool_source = _ensure_cv_text(user_input)
    if tool_source is None:
        reply_text = (
            "I need a CV or resume text to recommend relevant jobs. "
            "Please upload your CV so I can help."
        )
        history.add_ai_message(reply_text)
        return {
            "output": reply_text,
            "intermediate_steps": [],
        }

    tool_result = find_relevant_jobs_tool.invoke({"cv_text": tool_source})
    reply_text = _stringify_content(tool_result)
    history.add_ai_message(reply_text)
    if used_backend_cv:
        tool_input_label = "backend_cv"
    elif used_stored_cv or _ensure_cv_text(cv_text):
        tool_input_label = "stored_user_cv_text"
    else:
        tool_input_label = "user_message_cv"
    action = SimpleNamespace(tool="find_relevant_jobs_tool", tool_input=tool_input_label)
    return {
        "output": reply_text,
        "intermediate_steps": [(action, reply_text)],
    }


def _handle_cv_review(
    *,
    history: InMemoryChatMessageHistory,
    user_id: str,
    cv_text: Optional[str],
    user_input: str,
) -> Dict[str, Any]:
    """Invoke the review tool and capture the response."""
    review_source = _ensure_cv_text(cv_text)
    used_stored_cv = False
    used_backend_cv = False
    if review_source is None:
        stored_cv = _ensure_cv_text(_get_user_cv_text(user_id))
        if stored_cv is not None:
            review_source = stored_cv
            used_stored_cv = True
    if review_source is None:
        backend_cv = _fetch_backend_cv_text(user_id)
        if backend_cv is not None:
            review_source = backend_cv
            used_backend_cv = True
            _store_user_cv(user_id, backend_cv)
    if review_source is None:
        review_source = _ensure_cv_text(user_input)
    if review_source is None:
        reply_text = (
            "I'm unable to locate a CV to review. Please upload your CV or paste the content so I can help."
        )
        history.add_ai_message(reply_text)
        return {
            "output": reply_text,
            "intermediate_steps": [],
        }
    review_output = review_cv_tool.invoke({"cv_text": review_source})
    review_text = str(review_output or "").strip()
    history.add_ai_message(review_text)
    review_input_label = "user_message_cv"
    if used_backend_cv:
        review_input_label = "backend_cv"
    elif used_stored_cv or _ensure_cv_text(cv_text):
        review_input_label = "stored_user_cv_text"
    action = SimpleNamespace(tool="review_cv_tool", tool_input=review_input_label)
    return {
        "output": review_text,
        "intermediate_steps": [(action, review_text)],
    }


def _handle_free_chat(
    *,
    history: InMemoryChatMessageHistory,
    model_name: str,
    temperature: float,
) -> Dict[str, Any]:
    """Fallback branch that routes to the general chat model."""
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )
    prompt_messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT), *history.messages]
    response = model.invoke(prompt_messages)
    if hasattr(response, "content"):
        reply_text = str(response.content or "").strip()
    else:
        reply_text = str(response).strip()
    history.add_ai_message(reply_text)
    return {
        "output": reply_text,
        "intermediate_steps": [],
    }


def _handle_cv_upload(
    *,
    history: InMemoryChatMessageHistory,
    user_id: str,
    fallback_cv_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch and store CV content via the upload tool, returning the summary response."""
    tool_result = upload_cv_tool.invoke({"user_id": user_id})

    message_text: str
    cv_text: Optional[str] = None

    if isinstance(tool_result, dict):
        message_text = _stringify_content(tool_result.get("message"))
        cv_text = _ensure_cv_text(tool_result.get("cv_text"))
    else:
        message_text = _stringify_content(tool_result)

    if not cv_text and fallback_cv_text:
        cv_text = _ensure_cv_text(fallback_cv_text)

    if cv_text:
        _store_user_cv(user_id, cv_text)

    reply_text = message_text or (
        "I attempted to load your CV but did not receive a response from the backend."
    )
    history.add_ai_message(reply_text)
    action = SimpleNamespace(tool="upload_cv_tool", tool_input=user_id)
    return {
        "output": reply_text,
        "intermediate_steps": [(action, reply_text)],
    }


def chat_turn(
    user_input: str,
    *,
    user_id: str,
    model_name: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    cv_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a conversational turn with memory support.

    Args:
        user_input: Latest user message.
        user_id: Identifier used for user-scoped memory storage.
        model_name: Chat model to use for this turn.
        temperature: Sampling temperature.
        cv_text: Optional CV text provided directly with the request.

    Returns:
        Agent executor output dictionary (contains `output` and any tool traces).
    """
    if not user_id:
        raise ValueError("A user_id is required for chat turns.")

    history = _get_chat_history(user_id)
    previous_message_text: Optional[str] = None
    if history.messages:
        last_history_message = history.messages[-1]
        if getattr(last_history_message, "type", "") == "ai":
            previous_message_text = _stringify_content(last_history_message.content)
    history.add_user_message(user_input)

    provided_cv_text = _ensure_cv_text(cv_text)
    if provided_cv_text:
        _store_user_cv(user_id, provided_cv_text)

    stored_cv_text = _get_user_cv_text(user_id)

    intended_tool = _classify_intent(
        user_input,
        previous_bot_message=previous_message_text,
    )
    if intended_tool == "upload_cv_tool":
        return _handle_cv_upload(
            history=history,
            user_id=user_id,
            fallback_cv_text=stored_cv_text,
        )
    if intended_tool == "find_relevant_jobs_tool":
        return _handle_job_search(
            history=history,
            user_id=user_id,
            cv_text=stored_cv_text,
            user_input=user_input,
        )
    if intended_tool == "review_cv_tool":
        return _handle_cv_review(
            history=history,
            user_id=user_id,
            cv_text=stored_cv_text,
            user_input=user_input,
        )

    return _handle_free_chat(
        history=history,
        model_name=model_name,
        temperature=temperature,
    )
