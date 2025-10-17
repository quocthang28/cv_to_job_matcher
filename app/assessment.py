"""Gemini-based qualitative assessment of CV/job matches."""

from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import GEMINI_MODEL_CANDIDATES
from .jobs import load_review_instructions


def assess_matches(
    cv_text: str,
    matches: List[Dict[str, Optional[str]]],
    model: Optional[str] = "gemini-2.0-flash-exp",
) -> Optional[str]:
    """Use Gemini to generate a qualitative match assessment."""
    from os import getenv

    api_key = getenv("GOOGLE_API_KEY") or getenv("GOOGLE_GENAI_API_KEY") or getenv("API_KEY")
    if not api_key:
        print("Gemini API key not found in environment. Skipping LLM assessment.")
        return None

    match_sections: List[str] = []
    for idx, match in enumerate(matches, 1):
        description = (match.get("content") or "").strip()
        if len(description) > 1200:
            description = f"{description[:1200].rstrip()}..."
        similarity = match.get("similarity_score")
        similarity_display = f"{similarity:.2%}" if isinstance(similarity, (int, float)) else "N/A"
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
