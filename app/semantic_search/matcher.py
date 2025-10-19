"""Vector-store matcher utilities for CV to job recommendation."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import DEFAULT_TOP_K, PERSIST_DIR
from ..jobs import load_job_descriptions

_matcher_instance: Optional["CVJobMatcher"] = None
_matcher_jobs: List[Dict[str, Optional[str]]] = []

class CVJobMatcher:
    """Encapsulates vector-store creation and similarity search against job data."""

    def __init__(self, persist_directory: Path = PERSIST_DIR):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_directory = Path(persist_directory)
        self.vectorstore: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        if self.persist_directory.exists():
            try:
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    collection_name="job_descriptions",
                    persist_directory=str(self.persist_directory),
                )
                print(f"Loaded existing vector store from {self.persist_directory}")
            except Exception:
                # Existing directory but no usable data; will rebuild on init.
                self.vectorstore = None

    def initialize_job_database(self, jobs: List[Dict[str, Optional[str]]]) -> None:
        """Create (or rebuild) the persistent vector store with job descriptions."""
        documents: List[Document] = []
        for job in jobs:
            job_text = f"""
            Job Title: {job['title']}
            Company: {job['company']}
            Location: {job['location']}
            Category: {job['category']}
            Salary: {job.get('salary') or 'Not specified'}

            {job['description']}
            """

            metadata = {
                "job_id": job["id"],
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "category": job["category"],
                "salary": job.get("salary"),
                "source": job["path"],
            }
            metadata = {key: value for key, value in metadata.items() if value is not None}

            documents.append(
                Document(
                    page_content=job_text,
                    metadata=metadata,
                )
            )

        splits = self.text_splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="job_descriptions",
            persist_directory=str(self.persist_directory),
        )
        print(f"✓ Initialized job database with {len(jobs)} jobs")

    def match_cv_to_jobs(self, cv_text: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Return top-K matching jobs for the provided CV text."""
        if not self.vectorstore:
            raise ValueError("Job database not initialized. Call initialize_job_database first.")

        results = self.vectorstore.similarity_search_with_score(cv_text, k=top_k)

        matches: List[Dict[str, Any]] = []
        for doc, score in results:
            matches.append(
                {
                    "job_id": doc.metadata["job_id"],
                    "title": doc.metadata["title"],
                    "company": doc.metadata["company"],
                    "salary": doc.metadata.get("salary") or "Not specified",
                    "location": doc.metadata.get("location", "Unknown location"),
                    "similarity_score": round(1 - score, 4),
                    "content": doc.page_content,
                }
            )
        return matches


def init_matcher(
    jobs: Optional[List[Dict[str, Optional[str]]]] = None,
    *,
    force_reinitialize: bool = False,
) -> CVJobMatcher:
    """Initialize (or reuse) the global matcher instance and vector store."""
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
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Retrieve top job matches for the provided CV text."""
    if not cv_text.strip():
        raise ValueError("CV text must not be empty.")

    if matcher is None:
        matcher = init_matcher(_matcher_jobs or None)

    return matcher.match_cv_to_jobs(cv_text, top_k=top_k)


def matcher_ready() -> bool:
    """Return True if the matcher has a loaded vector store."""
    return _matcher_instance is not None and _matcher_instance.vectorstore is not None


def jobs_indexed() -> int:
    """Return the number of jobs currently cached from the last initialization."""
    return len(_matcher_jobs)
