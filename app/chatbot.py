"""LangChain chatbot orchestration backed by the CV matcher tool."""

from types import SimpleNamespace
from typing import Any, Dict, Optional, cast

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from .config import CHAT_SYSTEM_PROMPT, DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE
from .tools import (
    classify_tool_intent_tool,
    find_relevant_jobs_tool,
    review_cv_tool,
)

_chat_histories: Dict[str, InMemoryChatMessageHistory] = {}
_chat_runnable: Optional[RunnableWithMessageHistory] = None
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
) -> RunnableWithMessageHistory:
    """
    Build (or reuse) a LangChain runnable with tool-calling and memory support.

    Args:
        model_name: OpenAI chat model identifier.
        temperature: Sampling temperature for the chat model.

    Returns:
        RunnableWithMessageHistory configured for session-based conversations.
    """
    global _chat_runnable, _chat_config

    cached = (
        _chat_runnable is not None
        and _chat_config.get("model_name") == model_name
        and _chat_config.get("temperature") == temperature
    )
    if cached:
        return cast(RunnableWithMessageHistory, _chat_runnable)

    tools = [find_relevant_jobs_tool]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CHAT_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )
    agent = create_tool_calling_agent(model, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,
    )

    _chat_runnable = RunnableWithMessageHistory(
        executor,
        lambda session_id: _get_chat_history(session_id),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )
    _chat_config = {"model_name": model_name, "temperature": temperature}

    return cast(RunnableWithMessageHistory, _chat_runnable)


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

    if intended_tool == "find_relevant_jobs_tool":
        runnable = get_chat_runnable(model_name=model_name, temperature=temperature)
        return runnable.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}},
        )

    history = _get_chat_history(session_id)
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
