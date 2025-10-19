"""Utilities for loading job descriptions, CV text, and review instructions."""

import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .config import REVIEW_INSTRUCTIONS_PATH

try:  # PyPDFLoader ships with langchain-community; guard for safety.
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:  # pragma: no cover - fallback handled at runtime.
    PyPDFLoader = None  # type: ignore[assignment]


_FIELD_ALIASES: Dict[str, List[str]] = {
    "title": ["title", "job title", "role", "position"],
    "company": ["company", "company name", "employer", "organization"],
    "location": ["location", "based in", "work location"],
    "category": ["category", "department", "team"],
    "salary": ["salary", "salary range", "compensation", "pay", "rate"],
}
_SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class JobDocument:
    """Structured representation of a job posting pulled from disk."""

    identifier: str
    title: str
    company: str
    location: str
    category: str
    salary: Optional[str]
    description: str
    path: Path

    def to_mapping(self) -> Dict[str, Optional[str]]:
        return {
            "id": self.identifier,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "category": self.category,
            "salary": self.salary,
            "description": self.description,
            "path": str(self.path),
        }


def _load_text_content(path: Path) -> str:
    """Extract text from Markdown or PDF documents."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if PyPDFLoader is None:
            raise ImportError(
                "PyPDFLoader is required to process PDF files but is not installed."
            )
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        return "\n\n".join(
            doc.page_content.strip() for doc in documents if doc.page_content.strip()
        )

    # Default to UTF-8 text read (Markdown and other plain-text content).
    return path.read_text(encoding="utf-8")


def _normalize_key(key: str) -> str:
    return re.sub(r"[\s_-]+", "", key.lower())


def _parse_front_matter(raw_text: str) -> Tuple[Dict[str, str], str]:
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


def _value_from_metadata(metadata: Dict[str, str], key: str) -> Optional[str]:
    aliases = {_normalize_key(alias) for alias in _FIELD_ALIASES.get(key, [])}
    for raw_key, value in metadata.items():
        if _normalize_key(raw_key) in aliases:
            return value
    return None


def _extract_field_from_lines(lines: Iterable[str], alias: str) -> Optional[str]:
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


def _extract_structured_field(
    *,
    key: str,
    front_matter: Dict[str, str],
    lines: List[str],
) -> Optional[str]:
    front_matter_value = _value_from_metadata(front_matter, key)
    if front_matter_value:
        return front_matter_value

    for alias in _FIELD_ALIASES.get(key, []):
        value = _extract_field_from_lines(lines, alias)
        if value:
            return value
    return None


def _iter_job_files(base_dir: Path) -> Iterator[Path]:
    for path in sorted(base_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES:
            yield path


def _parse_job_document(path: Path) -> Optional[JobDocument]:
    raw_content = _load_text_content(path).strip()
    if not raw_content:
        return None

    front_matter, body = _parse_front_matter(raw_content)
    content = body.strip() or raw_content
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    title = (
        _extract_structured_field(key="title", front_matter=front_matter, lines=lines)
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

    company = _extract_structured_field(
        key="company", front_matter=front_matter, lines=lines
    ) or "Unknown Company"
    location = _extract_structured_field(
        key="location", front_matter=front_matter, lines=lines
    ) or "Unknown Location"
    category = _extract_structured_field(
        key="category", front_matter=front_matter, lines=lines
    ) or "Unknown Category"
    salary = _extract_structured_field(
        key="salary", front_matter=front_matter, lines=lines
    )

    return JobDocument(
        identifier=path.stem,
        title=title,
        company=company,
        location=location,
        category=category,
        salary=salary,
        description=content,
        path=path,
    )


def load_job_descriptions(base_dir: Path = Path("jd")) -> List[Dict[str, Optional[str]]]:
    """Load job descriptions from Markdown files (structured or free-form)."""
    jobs: List[Dict[str, Optional[str]]] = []
    if not base_dir.exists():
        return jobs

    for job_file in _iter_job_files(base_dir):
        document = _parse_job_document(job_file)
        if document:
            jobs.append(document.to_mapping())

    return jobs


def read_cv_text(cv_path: Path) -> str:
    """Read CV text from the provided path."""
    if not cv_path.exists():
        raise FileNotFoundError(f"CV file not found: {cv_path}")
    return _load_text_content(cv_path).strip()


def extract_uploaded_cv_text(content: bytes, filename: Optional[str]) -> str:
    """
    Extract CV text from uploaded file bytes.

    Supports Markdown/UTF-8 text files and PDF resumes using PyPDFLoader.
    """
    suffix = Path(filename or "").suffix.lower()

    if suffix == ".pdf":
        if PyPDFLoader is None:
            raise ImportError(
                "PyPDFLoader is required to process PDF files but is not installed."
            )
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        try:
            return _load_text_content(tmp_path).strip()
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    try:
        return content.decode("utf-8").strip()
    except UnicodeDecodeError as err:
        raise ValueError(
            "CV file must be UTF-8 encoded text/Markdown or a PDF document."
        ) from err


def load_review_instructions(path: Path = REVIEW_INSTRUCTIONS_PATH) -> str:
    """Load review prompt instructions from markdown file."""
    if not path.exists():
        raise FileNotFoundError(f"Review instructions file not found: {path}")
    return path.read_text(encoding="utf-8")
