"""Utilities that provide mock CV content for local development."""

from pathlib import Path
from typing import Dict, Optional

try:  # Reuse helper when executed as module or script.
    from .jobs import read_cv_text
except ImportError:  # pragma: no cover - fallback when running as script.
    from jobs import read_cv_text  # type: ignore

__all__ = ["MOCK_USER_CV_MAP", "fetch_mock_user_cv", "get_mock_user_cv_text"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CV_DIRECTORY = _PROJECT_ROOT / "cv"

MOCK_USER_CV_MAP: Dict[str, Path] = {
    "user-1": _CV_DIRECTORY / "cv.pdf",
    "user-2": _CV_DIRECTORY / "cv1.md",
    "user-3": _CV_DIRECTORY / "cv2.md",
    "user-4": _CV_DIRECTORY / "cv3.md",
}


def fetch_mock_user_cv(user_id: str) -> Dict[str, str]:
    """
    Return the mock CV payload for a given user id.

    Raises:
        ValueError: if user_id is blank.
        KeyError: if no mock CV is configured for the user.
        FileNotFoundError: if the configured file is missing.
        RuntimeError: if reading the CV content fails.
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id must not be empty.")

    normalized_user_id = user_id.strip()
    cv_path = MOCK_USER_CV_MAP.get(normalized_user_id)
    if cv_path is None:
        raise KeyError(f"No CV configured for user '{normalized_user_id}'.")

    if not cv_path.exists() or not cv_path.is_file():
        raise FileNotFoundError(f"Configured CV missing: {cv_path}")

    try:
        cv_text = read_cv_text(cv_path)
    except Exception as err:  # pragma: no cover - defensive read path.
        raise RuntimeError(f"Failed to read CV '{cv_path}': {err}") from err

    return {
        "user_id": normalized_user_id,
        "cv_path": str(cv_path),
        "cv_text": cv_text,
    }


def get_mock_user_cv_text(user_id: str) -> Optional[str]:
    """Convenience wrapper returning just the CV text or None on failure."""
    try:
        payload = fetch_mock_user_cv(user_id)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError):
        return None
    return payload.get("cv_text")
