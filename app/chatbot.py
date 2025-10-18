"""LangChain chatbot orchestration backed by the CV matcher tool."""

from typing import Any, Dict, List, Optional, cast

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from .config import CHAT_SYSTEM_PROMPT, DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE
from .matcher import match_jobs

_chat_histories: Dict[str, InMemoryChatMessageHistory] = {}
_chat_runnable: Optional[RunnableWithMessageHistory] = None
_chat_config: Dict[str, Any] = {}


def _format_matches_for_tool(matches: List[Dict[str, Any]]) -> str:
    """Format matcher output so the LLM can surface it cleanly to users."""
    if not matches:
        return "No matching jobs found for the provided CV."

    lines = []
    for idx, match in enumerate(matches, 1):
        similarity = match.get("similarity_score")
        similarity_display = f"{similarity:.2%}" if isinstance(similarity, (int, float)) else "N/A"
        lines.append(
            (
                f"{idx}. {match.get('title', 'Unknown Title')} at "
                f"{match.get('company', 'Unknown Company')} "
                f"(Similarity: {similarity_display})\n"
                f"   Location: {match.get('location', 'Unknown Location')} | "
                f"Job ID: {match.get('job_id', 'N/A')}"
            )
        )
    return "\n".join(lines)


@tool("find_relevant_jobs")
def find_relevant_jobs_tool(cv_text: str) -> str:
    """Use ONLY when the user provides a CV text or asks: 
'find matching jobs', 'recommend jobs', or 'compare my resume'. 
Do NOT use this tool for general job discussions or follow-ups."""
    try:
        matches = match_jobs(cv_text)
    except Exception as err:  # Broad catch to ensure the agent sees the failure.
        return f"Error while matching CV to jobs: {err}"
    return _format_matches_for_tool(matches)


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

    runnable = get_chat_runnable(model_name=model_name, temperature=temperature)
    message = user_input
    if cv_text:
        message = f"{user_input}\n\nCandidate CV:\n{cv_text}"

    return runnable.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )
