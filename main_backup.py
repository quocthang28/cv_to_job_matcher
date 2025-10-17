import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

PERSIST_DIR = Path("job_vectorstore")
REVIEW_INSTRUCTIONS_PATH = Path("review_instructions.md")
GEMINI_MODEL_CANDIDATES = [
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-pro",
]

_matcher_instance: Optional["CVJobMatcher"] = None
_matcher_jobs: List[Dict[str, Optional[str]]] = []
CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant that matches candidates to job descriptions. "
    "Whenever the user shares a CV or asks for job suggestions, call the available "
    "tool to retrieve relevant job matches and present concise summaries with job title, "
    "company, location, similarity score, and job ID."
)
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
_chat_histories: Dict[str, InMemoryChatMessageHistory] = {}
_chat_runnable: Optional[RunnableWithMessageHistory] = None
_chat_config: Dict[str, Any] = {}
_cv_store: Dict[str, str] = {}
app = FastAPI(title="CV Matcher Chatbot")

load_dotenv()


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

def load_job_descriptions(base_dir: Path = Path("jd")) -> List[Dict[str, Optional[str]]]:
    """Load job descriptions from Markdown files (structured or free-form)."""
    jobs: List[Dict[str, Optional[str]]] = []
    if not base_dir.exists():
        return jobs

    field_aliases = {
        "title": ["title", "job title", "role", "position"],
        "company": ["company", "company name", "employer", "organization"],
        "location": ["location", "based in", "work location"],
        "category": ["category", "department", "team"],
        "salary": ["salary", "salary range", "compensation", "pay", "rate"],
    }

    def normalize_key(key: str) -> str:
        return re.sub(r"[\s_-]+", "", key.lower())

    def parse_front_matter(raw_text: str) -> Tuple[Dict[str, str], str]:
        lines = raw_text.splitlines()
        if lines and lines[0].strip() == "---":
            metadata: Dict[str, str] = {}
            idx = 1
            while idx < len(lines):
                line = lines[idx].strip()
                if line == "---":
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
                idx += 1
            body = "\n".join(lines[idx + 1:]) if idx < len(lines) else ""
            return metadata, body
        return {}, raw_text

    def value_from_metadata(metadata: Dict[str, str], key: str) -> Optional[str]:
        aliases = {normalize_key(alias) for alias in field_aliases.get(key, [])}
        for raw_key, value in metadata.items():
            if normalize_key(raw_key) in aliases:
                return value
        return None

    for path in sorted(base_dir.glob("*.md")):
        raw_content = path.read_text(encoding="utf-8").strip()
        if not raw_content:
            continue

        front_matter, body = parse_front_matter(raw_content)
        content = body.strip() or raw_content
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        def extract_field(key: str) -> Optional[str]:
            front_matter_value = value_from_metadata(front_matter, key)
            if front_matter_value:
                return front_matter_value

            aliases = field_aliases.get(key, [])
            for alias in aliases:
                pattern = re.compile(
                    rf"^\s*(?:[-*•]|\d+\.)?\s*{re.escape(alias)}\s*[:|=-]\s*(.+)$",
                    re.IGNORECASE,
                )
                for line in lines:
                    match = pattern.match(line.lstrip("* ").replace("**", ""))
                    if match:
                        candidate = match.group(1).strip()
                        if candidate:
                            return candidate
            return None

        title = (
            extract_field("title")
            or next(
                (
                    line.lstrip("#").strip()
                    for line in lines
                    if line.lstrip().startswith("#")
                ),
                None,
            )
            or path.stem.replace("_", " ").title()
        )

        company = extract_field("company") or "Unknown Company"
        location = extract_field("location") or "Unknown Location"
        category = extract_field("category") or "Unknown Category"
        salary = extract_field("salary")

        jobs.append(
            {
                "id": path.stem,
                "title": title,
                "company": company,
                "location": location,
                "category": category,
                "salary": salary,
                "description": content,
                "path": str(path),
            }
        )

    return jobs

class CVJobMatcher:
    def __init__(self, persist_directory: Path = PERSIST_DIR):
        """Initialize the CV Job Matcher with embeddings and vector store"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_directory = Path(persist_directory)
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        if self.persist_directory.exists():
            try:
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    collection_name="job_descriptions",
                    persist_directory=str(self.persist_directory)
                )
                print(f"Loaded existing vector store from {self.persist_directory}")
            except Exception:
                # Existing directory but no usable data; will rebuild on init
                self.vectorstore = None
        
    def initialize_job_database(self, jobs):
        """
        Initialize vector database with job descriptions
        
        Args:
            jobs: List of job dictionaries with description
        """
        documents = []
        for job in jobs:
            # Create a comprehensive text for each job
            job_text = f"""
            Job Title: {job['title']}
            Company: {job['company']}
            Location: {job['location']}
            Category: {job['category']}
            Salary: {job.get('salary') or 'Not specified'}
            
            {job['description']}
            """
            
            metadata = {
                "job_id": job['id'],
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "category": job['category'],
                "salary": job.get('salary'),
                "source": job['path']
            }
            metadata = {key: value for key, value in metadata.items() if value is not None}

            doc = Document(
                page_content=job_text,
                metadata=metadata
            )
            documents.append(doc)
        
        # Split documents and create vector store
        splits = self.text_splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="job_descriptions",
            persist_directory=str(self.persist_directory)
        )
        
        print(f"✓ Initialized job database with {len(jobs)} jobs")
        
    def match_cv_to_jobs(self, cv_text, top_k=3):
        """
        Match CV to job descriptions using semantic similarity
        
        Args:
            cv_text: The user's CV as text
            top_k: Number of top matches to return
            
        Returns:
            List of matched jobs with similarity scores
        """
        if not self.vectorstore:
            raise ValueError("Job database not initialized. Call initialize_job_database first.")
        
        # Search for similar jobs using vector similarity
        results = self.vectorstore.similarity_search_with_score(
            cv_text,
            k=top_k
        )
        
        matches = []
        for doc, score in results:
            matches.append({
                "job_id": doc.metadata['job_id'],
                "title": doc.metadata['title'],
                "company": doc.metadata['company'],
                "salary": doc.metadata.get('salary') or "Not specified",
                "location": doc.metadata.get('location', "Unknown location"),
                "similarity_score": round(1 - score, 4),  # Convert distance to similarity
                "content": doc.page_content
            })

        return matches


def init_matcher(
    jobs: Optional[List[Dict[str, Optional[str]]]] = None,
    *,
    force_reinitialize: bool = False,
) -> CVJobMatcher:
    """
    Initialize (or reuse) the global matcher instance and vector store.

    Args:
        jobs: Optional preloaded job list to avoid re-reading from disk.
        force_reinitialize: Rebuild the vector store even if one exists.

    Returns:
        A ready-to-use CVJobMatcher instance.
    """
    global _matcher_instance, _matcher_jobs

    if jobs is None:
        jobs = load_job_descriptions()

    if not jobs:
        raise ValueError(
            "No job descriptions found. Ensure the 'jd' directory has Markdown files."
        )

    if (
        _matcher_instance is not None
        and _matcher_instance.vectorstore is not None
        and not force_reinitialize
    ):
        return _matcher_instance

    matcher = CVJobMatcher()
    if matcher.vectorstore is None or force_reinitialize:
        matcher.initialize_job_database(jobs)

    _matcher_instance = matcher
    _matcher_jobs = jobs
    return matcher


def match_jobs(
    cv_text: str,
    *,
    matcher: Optional[CVJobMatcher] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve top job matches for the provided CV text.

    Args:
        cv_text: The CV content as a string.
        matcher: Optional matcher instance; falls back to global instance.
        top_k: Number of matches to return.

    Returns:
        List of job metadata dictionaries sorted by similarity.
    """
    if not cv_text.strip():
        raise ValueError("CV text must not be empty.")

    if matcher is None:
        matcher = init_matcher(_matcher_jobs or None)

    return matcher.match_cv_to_jobs(cv_text, top_k=top_k)


def _format_matches_for_tool(matches: List[Dict[str, Any]]) -> str:
    """Format matcher output so the LLM can surface it cleanly to users."""
    if not matches:
        return "No matching jobs found for the provided CV."

    lines = []
    for idx, match in enumerate(matches, 1):
        similarity = match.get("similarity_score")
        if isinstance(similarity, (int, float)):
            similarity_display = f"{similarity:.2%}"
        else:
            similarity_display = "N/A"
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
    """LangChain tool that runs the matcher and returns human-readable summaries."""
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
    temperature: float = 0.2,
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
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

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
    temperature: float = 0.2,
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

    # Ensure matcher is ready before the agent attempts tool usage.
    if _matcher_instance is None or _matcher_instance.vectorstore is None:
        init_matcher(_matcher_jobs or None)

    runnable = get_chat_runnable(model_name=model_name, temperature=temperature)
    message = user_input
    if cv_text:
        message = f"{user_input}\n\nCandidate CV:\n{cv_text}"

    return runnable.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )


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
    ready = _matcher_instance is not None and _matcher_instance.vectorstore is not None
    return {
        "status": "ok",
        "vectorstore_ready": ready,
        "jobs_indexed": len(_matcher_jobs),
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
        vector_store_path=str(PERSIST_DIR.resolve()),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_cv(file: UploadFile = File(...)) -> UploadResponse:
    """Persist uploaded CV text in memory and return a reference id."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as err:
        raise HTTPException(
            status_code=400,
            detail="CV file must be UTF-8 encoded text or Markdown.",
        ) from err

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
            temperature=0.2 if request.temperature is None else request.temperature,
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
    
def read_cv_text(cv_path: Path) -> str:
    """Read CV text from the provided path."""
    if not cv_path.exists():
        raise FileNotFoundError(f"CV file not found: {cv_path}")
    return cv_path.read_text(encoding="utf-8")

def load_review_instructions(path: Path = REVIEW_INSTRUCTIONS_PATH) -> str:
    """Load review prompt instructions from markdown file."""
    if not path.exists():
        raise FileNotFoundError(f"Review instructions file not found: {path}")
    return path.read_text(encoding="utf-8")

def assess_matches(
    cv_text: str,
    matches: List[Dict[str, Optional[str]]],
    model: Optional[str] = "gemini-2.0-flash-exp",
) -> Optional[str]:
    """
    Ask Gemini for a qualitative assessment of how the CV aligns to the matched jobs.
    """
    load_dotenv()
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
        or os.getenv("API_KEY")
    )
    if not api_key:
        print("Gemini API key not found in environment. Skipping LLM assessment.")
        return None

    match_sections: List[str] = []
    for idx, match in enumerate(matches, 1):
        description = (match.get("content") or "").strip()
        if len(description) > 1200:
            description = f"{description[:1200].rstrip()}..."
        similarity = match.get("similarity_score")
        similarity_display = (
            f"{similarity:.2%}" if isinstance(similarity, (int, float)) else "N/A"
        )
        match_sections.append(
            "\n".join(
                [
                    f"{idx}. Title: {match.get('title', 'Unknown Title')} | "
                    f"Company: {match.get('company', 'Unknown Company')} | "
                    f"Similarity: {similarity_display}",
                    f"Location: {match.get('location', 'Unknown Location')}",
                    f"Job Context:\n{description or 'No additional job context provided.'}",
                ]
            )
        )

    matches_block = "\n\n".join(match_sections) if match_sections else "No matches found."

    try:
        instructions_template = load_review_instructions()
    except FileNotFoundError as err:
        print(f"{err}. Skipping LLM assessment.")
        return None

    prompt = (
        instructions_template.replace("{cv}", cv_text.strip())
        .replace("{matches}", matches_block)
        .strip()
    )

    candidate_models: List[str] = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(m for m in GEMINI_MODEL_CANDIDATES if m not in candidate_models)

    attempts: List[str] = []

    def _parse_response(response: Any) -> Optional[str]:
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    text_fragment = getattr(item, "text", None)
                    if isinstance(text_fragment, str):
                        parts.append(text_fragment)
                        continue
                    if isinstance(item, dict):
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                combined = "".join(parts).strip()
                return combined or None
        if isinstance(response, str):
            return response.strip()
        return str(response)

    for candidate in candidate_models:
        try:
            client = ChatGoogleGenerativeAI(
                model=candidate,
                temperature=0.2,
                google_api_key=api_key,
            )
            response: Any = client.invoke(prompt)
            parsed = _parse_response(response)
            if parsed:
                print(f"\nUsing Gemini model '{candidate}'.")
                return parsed
            attempts.append(f"{candidate}: Empty response returned.")
        except Exception as err:
            attempts.append(f"{candidate}: {err}")

    print("Unable to obtain Gemini assessment. Attempts:")
    for line in attempts:
        print(f" - {line}")
    return None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match a CV against job descriptions.")
    parser.add_argument(
        "--cv",
        type=str,
        required=True,
        help="Path to the CV file (text or Markdown).",
    )
    return parser.parse_args()

def main():
    """Main function to demonstrate the CV matcher"""
    load_dotenv()
    args = parse_args()

    jobs = load_job_descriptions()
    if not jobs:
        print("No job descriptions found in the jd directory. Please add Markdown job files and try again.")
        return

    cv_text = read_cv_text(Path(args.cv))

    print("=" * 60)
    print("LangChain CV to Job Matcher Demo")
    print("=" * 60)
    print()

    try:
        matcher = init_matcher(jobs)
    except ValueError as err:
        print(str(err))
        return

    if matcher.vectorstore is None:
        print("Matcher vector store is unavailable. Initialization failed.")
        return

    print("Vector store ready; using existing data.\n")

    # Match CV to jobs
    print("Matching CV to jobs...")
    matches = match_jobs(cv_text, matcher=matcher, top_k=5)
    
    print("\n" + "=" * 60)
    print("TOP MATCHING JOBS")
    print("=" * 60)
    
    unique_matches = []
    seen_ids = set()
    for match in matches:
        if match["job_id"] in seen_ids:
            continue
        seen_ids.add(match["job_id"])
        unique_matches.append(match)

    for i, match in enumerate(unique_matches, 1):
        print(f"\n{i}. {match['title']} at {match['company']}")
        print(f"   Similarity Score: {match['similarity_score']:.2%}")
        print(f"   Job ID: {match['job_id']}")

    print("\n" + "=" * 60)
    print("ASK LLM FOR ASSESSMENT WITH PROVIDED INSTRUCTION")
    print("=" * 60)

    if unique_matches:
        assessment = assess_matches(cv_text, unique_matches)
        if assessment:
            print("\n" + "=" * 60)
            print("GEMINI ASSESSMENT")
            print("=" * 60 + "\n")
            print(assessment)
        else:
            print("\nSkipping Gemini assessment (no response returned).")

if __name__ == "__main__":
    main()
