"""Vector-store matcher utilities for CV to job recommendation."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

from ..config import DEFAULT_TOP_K, PERSIST_DIR
from ..jobs import load_job_descriptions

_matcher_instance: Optional["CVJobMatcher"] = None
_matcher_jobs: List[Dict[str, Any]] = []
_matcher_jobs_signature: Optional[str] = None


def _compute_jobs_signature(jobs: List[Dict[str, Any]]) -> str:
    digest = hashlib.sha1()
    for job in sorted(jobs, key=lambda item: str(item.get("id", ""))):
        digest.update(str(job.get("id", "")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(job.get("path", "")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(job.get("title", "")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(job.get("company", "")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(job.get("description", "")).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class SentenceTransformerEmbeddings(Embeddings):
    """Minimal adapter exposing encode methods expected by Chroma."""

    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vectors = self._model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors[0].tolist()


class CVJobMatcher:
    """Encapsulates vector-store creation and similarity search against job data."""

    def __init__(self, persist_directory: Path = PERSIST_DIR):
        self.embeddings = SentenceTransformerEmbeddings(
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
        else:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            # Vector store will be created lazily on first write.

    def initialize_job_database(self, jobs: List[Dict[str, Any]]) -> None:
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

    def _ensure_vectorstore(self) -> Chroma:
        """Return a ready Chroma instance, creating one when necessary."""
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name="job_descriptions",
                persist_directory=str(self.persist_directory),
            )
        return self.vectorstore

    def ingest_job(self, job_id: str, raw_content: str) -> Dict[str, Any]:
        """Add or update a single job posting inside the vector store."""
        job_id = job_id.strip()
        if not job_id:
            raise ValueError("job_id must not be empty.")

        content = raw_content.strip()
        if not content:
            raise ValueError("content must not be empty.")

        title = next(
            (
                line.lstrip("#").strip()
                for line in content.splitlines()
                if line.strip()
            ),
            None,
        )
        job_text = f"Job ID: {job_id}\n\n{content}"
        metadata = {
            "job_id": job_id,
            "title": title or job_id,
            "company": None,
            "location": None,
            "category": None,
            "salary": None,
            "source": "ingest_api",
        }

        document = Document(page_content=job_text, metadata=metadata)
        splits = self.text_splitter.split_documents([document])
        store = self._ensure_vectorstore()
        # Remove existing documents for this job to prevent duplicates.
        store.delete(where={"job_id": job_id})
        store.add_documents(splits)
        store.persist()

        return {
            "job_id": job_id,
            "title": metadata["title"],
            "description": content,
            "chunks_added": len(splits),
        }

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
    jobs: Optional[List[Dict[str, Any]]] = None,
    *,
    force_reinitialize: bool = False,
) -> CVJobMatcher:
    """Initialize (or reuse) the global matcher instance and vector store."""
    global _matcher_instance, _matcher_jobs, _matcher_jobs_signature

    if jobs is None:
        jobs = load_job_descriptions()

    if not jobs:
        raise ValueError(
            "No job descriptions found. Ensure the 'jd' directory has Markdown files."
        )

    current_signature = _compute_jobs_signature(jobs)

    if (
        _matcher_instance is not None
        and _matcher_instance.vectorstore is not None
        and not force_reinitialize
        and _matcher_jobs_signature == current_signature
    ):
        return _matcher_instance

    matcher = CVJobMatcher()
    if matcher.vectorstore is None or force_reinitialize:
        matcher.initialize_job_database(jobs)

    _matcher_instance = matcher
    _matcher_jobs = jobs
    _matcher_jobs_signature = current_signature
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


def ingest_job_content(
    job_id: str,
    content: str,
    *,
    matcher: Optional[CVJobMatcher] = None,
) -> Dict[str, Any]:
    """Ingest a single job description into the persisted vector store."""
    global _matcher_instance, _matcher_jobs

    if matcher is None:
        if _matcher_instance is None:
            matcher = CVJobMatcher()
            _matcher_instance = matcher
        else:
            matcher = _matcher_instance

    result = matcher.ingest_job(job_id, content)

    job_entry = {
        "id": result["job_id"],
        "title": result.get("title"),
        "company": None,
        "location": None,
        "category": None,
        "salary": None,
        "description": result.get("description"),
        "path": "ingest_api",
    }
    _matcher_jobs = [job for job in _matcher_jobs if job.get("id") != job_entry["id"]]
    _matcher_jobs.append(job_entry)

    return result
