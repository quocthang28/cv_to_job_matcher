"""FastAPI surface for the CV matcher chatbot."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .chatbot import chat_turn
from .config import DEFAULT_CHAT_MODEL, DEFAULT_TEMPERATURE
from .jobs import extract_uploaded_cv_text, load_job_descriptions
from .matcher import init_matcher, jobs_indexed, matcher_ready

app = FastAPI(title="CV Matcher Chatbot")
_cv_store: Dict[str, str] = {}


class InitRequest(BaseModel):
    force_reinitialize: bool = False


class InitResponse(BaseModel):
    jobs_indexed: int
    vector_store_path: str


class UploadResponse(BaseModel):
    cv_id: str
    filename: Optional[str]
    char_length: int


class ChatRequest(BaseModel):
    session_id: str
    message: str
    cv_id: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None


class ChatResponse(BaseModel):
    session_id: str
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


@app.post("/upload", response_model=UploadResponse)
async def upload_cv(file: UploadFile = File(...)) -> UploadResponse:
    """Persist uploaded CV text in memory and return a reference id."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = extract_uploaded_cv_text(content, file.filename)
    except (ValueError, ImportError) as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    cv_id = str(uuid4())
    _cv_store[cv_id] = text
    return UploadResponse(
        cv_id=cv_id,
        filename=file.filename,
        char_length=len(text),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Handle a chat turn, optionally enriching the input with an uploaded CV."""
    cv_text = None
    if request.cv_id:
        cv_text = _cv_store.get(request.cv_id)
        if cv_text is None:
            raise HTTPException(status_code=404, detail="CV id not found.")

    try:
        result = chat_turn(
            request.message,
            session_id=request.session_id,
            model_name=request.model_name or DEFAULT_CHAT_MODEL,
            temperature=DEFAULT_TEMPERATURE if request.temperature is None else request.temperature,
            cv_text=cv_text,
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
        session_id=request.session_id,
        reply=reply,
        intermediate_steps=intermediate_steps or None,
    )
