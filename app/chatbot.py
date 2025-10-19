"""LangChain chatbot orchestration backed by the CV matcher tool."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from langchain.agents import create_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from .config import CHAT_SYSTEM_PROMPT, DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE
from .tools import (
    classify_tool_intent_tool,
    find_relevant_jobs_tool,
    review_cv_tool,
)

_chat_histories: Dict[str, InMemoryChatMessageHistory] = {}
_chat_agent: Optional[Any] = None
_chat_config: Dict[str, Any] = {}


def _get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return per-session memory storage for the chatbot."""
    if session_id not in _chat_histories:
        _chat_histories[session_id] = InMemoryChatMessageHistory()
    return _chat_histories[session_id]


def get_chat_runnable(
    *,
    model_name: str = DEFAULT_CHAT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Any:
    """
    Build (or reuse) a LangChain agent graph with tool-calling support.

    Args:
        model_name: OpenAI chat model identifier.
        temperature: Sampling temperature for the chat model.

    Returns:
        Compiled agent graph configured for session-based conversations.
    """
    global _chat_agent, _chat_config

    cached = (
        _chat_agent is not None
        and _chat_config.get("model_name") == model_name
        and _chat_config.get("temperature") == temperature
    )
    if cached:
        return _chat_agent

    tools = [find_relevant_jobs_tool]

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=CHAT_SYSTEM_PROMPT,
    )

    _chat_agent = agent
    _chat_config = {"model_name": model_name, "temperature": temperature}

    return agent


def _stringify_content(content: Any) -> str:
    """Normalize message content (string or content blocks) to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
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


def _build_intermediate_steps(
    new_messages: Sequence[BaseMessage],
) -> List[Any]:
    """Create intermediate step tuples from newly generated tool messages."""
    steps: List[Any] = []
    tool_call_details: Dict[str, Dict[str, Any]] = {}

    for message in new_messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if not call:
                    continue
                call_id = call.get("id") or ""
                tool_call_details[call_id] = {
                    "name": call.get("name") or "unknown_tool",
                    "args": call.get("args"),
                }
        elif isinstance(message, ToolMessage):
            tool_info = tool_call_details.get(message.tool_call_id, {})
            tool_name = tool_info.get("name") or message.name or "unknown_tool"
            raw_args = tool_info.get("args")
            if isinstance(raw_args, (str, int, float, bool)) or raw_args is None:
                tool_input = "" if raw_args is None else str(raw_args)
            else:
                tool_input = json.dumps(raw_args)
            steps.append(
                (
                    SimpleNamespace(tool=tool_name, tool_input=tool_input),
                    _stringify_content(message.content),
                )
            )

    return steps


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

    classification_result = classify_tool_intent_tool.invoke({"query": user_input})
    intended_tool = str(classification_result or "").strip()

    message = user_input
    if cv_text:
        message = f"{user_input}\n\nCandidate CV:\n{cv_text}"

    history = _get_chat_history(session_id)

    if intended_tool == "find_relevant_jobs_tool":
        agent = get_chat_runnable(model_name=model_name, temperature=temperature)
        prior_messages = list(history.messages)
        user_message = HumanMessage(content=message)
        conversation: List[BaseMessage] = prior_messages + [user_message]

        try:
            agent_result = agent.invoke({"messages": conversation})
        except Exception:
            # Preserve history on failure before re-raising.
            history.messages = prior_messages
            raise

        messages: List[BaseMessage]
        if isinstance(agent_result, dict):
            state_messages = agent_result.get("messages", [])
            if not isinstance(state_messages, Sequence):
                raise ValueError("Agent response missing message history.")
            messages = list(state_messages)
        elif isinstance(agent_result, Sequence):
            messages = list(agent_result)
        else:
            raise ValueError("Agent response missing message history.")

        if not messages:
            messages = conversation

        new_messages = messages[len(conversation) :]
        intermediate_steps = _build_intermediate_steps(new_messages)

        history.messages = messages

        reply_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                reply_text = _stringify_content(msg.content)
                break

        return {
            "output": reply_text,
            "intermediate_steps": intermediate_steps,
        }

    history.add_user_message(message)

    if intended_tool == "review_cv_tool":
        review_source = cv_text if cv_text else user_input
        review_output = review_cv_tool.invoke({"cv_text": review_source})
        review_text = str(review_output or "").strip()
        history.add_ai_message(review_text)
        action = SimpleNamespace(tool="review_cv_tool", tool_input=review_source)
        return {
            "output": review_text,
            "intermediate_steps": [(action, review_text)],
        }

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
