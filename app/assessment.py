"""OpenAI-based qualitative assessment of CV/job matches."""

from typing import Any, Dict, List, Optional

from .config import get_chat_model
from .jobs import load_review_instructions


def assess_matches(
    cv_text: str,
    matches: List[Dict[str, Optional[str]]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Optional[str]:
    """Use OpenAI to generate a qualitative match assessment."""
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

    try:
        client = get_chat_model(model_name=model, temperature=temperature)
        response: Any = client.invoke(prompt)

        # Parse the response
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content.strip()

        if isinstance(response, str):
            return response.strip()

        return str(response)

    except Exception as err:
        print(f"Unable to obtain OpenAI assessment: {err}")
        return None
