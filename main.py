import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

PERSIST_DIR = Path("job_vectorstore")
REVIEW_INSTRUCTIONS_PATH = Path("review_instructions.md")
GEMINI_MODEL_CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-pro",
]

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

def assess_matches_with_gemini(
    cv_text: str,
    matches: List[Dict[str, Optional[str]]],
    model: Optional[str] = None,
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

    def _parse_response(response: object) -> Optional[str]:
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
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
            response = client.invoke(prompt)
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
    
    # Initialize matcher
    matcher = CVJobMatcher()
    
    # Initialize job database
    if matcher.vectorstore is None:
        matcher.initialize_job_database(jobs)
        print()
    else:
        print("Vector store already initialized; using existing data.\n")
    
    # Match CV to jobs
    print("Matching CV to jobs...")
    matches = matcher.match_cv_to_jobs(cv_text, top_k=5)
    
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
    print("TOP MATCHING JOBS")
    print("=" * 60)

    if unique_matches:
        assessment = assess_matches_with_gemini(cv_text, unique_matches)
        if assessment:
            print("\n" + "=" * 60)
            print("GEMINI ASSESSMENT")
            print("=" * 60 + "\n")
            print(assessment)
        else:
            print("\nSkipping Gemini assessment (no response returned).")

if __name__ == "__main__":
    main()
