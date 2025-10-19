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
)

_chat_histories: Dict[str, InMemoryChatMessageHistory] = {}


def _get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return per-session memory storage for the chatbot."""
    if session_id not in _chat_histories:
        _chat_histories[session_id] = InMemoryChatMessageHistory()
    return _chat_histories[session_id]


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


def _classify_intent(user_input: str) -> str:
    """Use the lightweight classifier tool to decide which branch to follow."""
    if not user_input:
        return ""
    classification_result = classify_tool_intent_tool.invoke({"query": user_input})
    return str(classification_result or "").strip()


def _ensure_cv_text(source: Optional[str]) -> Optional[str]:
    """Return a sanitized CV text string or None when unusable."""
    if not source:
        return None
    text = source.strip()
    return text or None


def _handle_job_search(
    *,
    history: InMemoryChatMessageHistory,
    cv_text: Optional[str],
    user_input: str,
) -> Dict[str, Any]:
    """Invoke the matching tool when a CV is present."""
    tool_source = _ensure_cv_text(cv_text) or _ensure_cv_text(user_input)
    if tool_source is None:
        reply_text = (
            "I need a CV or resume text to recommend relevant jobs. "
            "Please upload your CV or paste the text so I can help."
        )
        history.add_ai_message(reply_text)
        return {
            "output": reply_text,
            "intermediate_steps": [],
        }

    tool_result = find_relevant_jobs_tool.invoke({"cv_text": tool_source})
    reply_text = _stringify_content(tool_result)
    history.add_ai_message(reply_text)
    tool_input_label = "uploaded_cv_text" if cv_text else "user_message_cv"
    action = SimpleNamespace(tool="find_relevant_jobs_tool", tool_input=tool_input_label)
    return {
        "output": reply_text,
        "intermediate_steps": [(action, reply_text)],
    }


def _handle_cv_review(
    *,
    history: InMemoryChatMessageHistory,
    cv_text: Optional[str],
    user_input: str,
) -> Dict[str, Any]:
    """Invoke the review tool and capture the response."""
    review_source = cv_text if _ensure_cv_text(cv_text) else user_input
    review_output = review_cv_tool.invoke({"cv_text": review_source})
    review_text = str(review_output or "").strip()
    history.add_ai_message(review_text)
    review_input_label = "uploaded_cv_text" if cv_text else "user_message_cv"
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


def chat_turn(
    user_input: str,
    *,
    session_id: str,
    model_name: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    cv_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a conversational turn with memory support.

    Args:
        user_input: Latest user message.
        session_id: Conversation identifier used for memory storage.
        model_name: Chat model to use for this turn.
        temperature: Sampling temperature.

    Returns:
        Agent executor output dictionary (contains `output` and any tool traces).
    """
    if not session_id:
        raise ValueError("A session_id is required for chat turns.")

    history = _get_chat_history(session_id)
    history.add_user_message(user_input)

    intended_tool = _classify_intent(user_input)
    if intended_tool == "find_relevant_jobs_tool":
        return _handle_job_search(history=history, cv_text=cv_text, user_input=user_input)
    if intended_tool == "review_cv_tool":
        return _handle_cv_review(history=history, cv_text=cv_text, user_input=user_input)

    return _handle_free_chat(
        history=history,
        model_name=model_name,
        temperature=temperature,
    )
