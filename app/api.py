"""FastAPI surface for the CV matcher chatbot."""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .chatbot import chat_turn
from .config import DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE
from .jobs import load_job_descriptions
from .semantic_search import init_matcher, jobs_indexed, matcher_ready

app = FastAPI(title="CV Matcher Chatbot")


class InitRequest(BaseModel):
    force_reinitialize: bool = False


class InitResponse(BaseModel):
    jobs_indexed: int
    vector_store_path: str


class ChatRequest(BaseModel):
    user_id: str
    message: str
    model_name: Optional[str] = None
    temperature: Optional[float] = None


class ChatResponse(BaseModel):
    user_id: str
    reply: str
    intermediate_steps: Optional[List[Dict[str, Any]]] = None


@app.on_event("startup")
async def _startup_init() -> None:
    """Attempt to warm the matcher on service startup."""
    try:
        init_matcher()
    except ValueError:
        # Jobs may not be present yet; defer initialization until first request.
        pass


@app.get("/health")
def healthcheck() -> Dict[str, Any]:
    """Report service status and matcher readiness."""
    return {
        "status": "ok",
        "vectorstore_ready": matcher_ready(),
        "jobs_indexed": jobs_indexed(),
    }


@app.post("/init", response_model=InitResponse)
def init_endpoint(request: InitRequest) -> InitResponse:
    """Initialize or rebuild the matcher vector store."""
    jobs = load_job_descriptions()
    if not jobs:
        raise HTTPException(
            status_code=400,
            detail="No job descriptions found. Populate the 'jd' directory and try again.",
        )

    matcher = init_matcher(jobs, force_reinitialize=request.force_reinitialize)
    if matcher.vectorstore is None:
        raise HTTPException(
            status_code=500,
            detail="Matcher initialization failed; vector store is unavailable.",
        )

    return InitResponse(
        jobs_indexed=len(jobs),
        vector_store_path=str(matcher.persist_directory.resolve()),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Handle a chat turn for a given user."""

    try:
        result = chat_turn(
            request.message,
            user_id=request.user_id,
            model_name=request.model_name or DEFAULT_CHAT_MODEL,
            temperature=DEFAULT_TEMPERATURE if request.temperature is None else request.temperature,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err

    intermediate_steps: List[Dict[str, Any]] = []
    for action, output in result.get("intermediate_steps", []):
        intermediate_steps.append(
            {
                "tool": getattr(action, "tool", "unknown"),
                "input": getattr(action, "tool_input", ""),
                "output": output,
            }
        )

    reply = str(result.get("output", "")).strip()
    return ChatResponse(
        user_id=request.user_id,
        reply=reply,
        intermediate_steps=intermediate_steps or None,
    )
