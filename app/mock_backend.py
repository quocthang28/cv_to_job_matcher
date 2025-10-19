"""Minimal HTTP mock backend that mimics an external CV service."""

from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException

try:  # When executed as `python -m app.mock_backend`
    from .jobs import read_cv_text
except ImportError:  # pragma: no cover
    from jobs import read_cv_text  # type: ignore

app = FastAPI(title="Mock CV Backend", version="0.1.0")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CV_DIRECTORY = _PROJECT_ROOT / "cv"

MOCK_USER_CV_MAP: Dict[str, Path] = {
    "user-1": _CV_DIRECTORY / "cv.pdf",
    "user-2": _CV_DIRECTORY / "cv1.md",
    "user-3": _CV_DIRECTORY / "cv2.md",
    "user-4": _CV_DIRECTORY / "cv3.md",
}


@app.get("/health", tags=["meta"])
def healthcheck() -> Dict[str, str]:
    """Report service readiness just like a real backend would."""
    return {"status": "ok"}


@app.get("/users/{user_id}/cv", tags=["cv"])
def get_user_cv_by_id(user_id: str) -> Dict[str, str]:
    """Retrieve CV content based on the mocked user-to-file mapping."""
    cv_path = MOCK_USER_CV_MAP.get(user_id)
    if cv_path is None:
        raise HTTPException(status_code=404, detail="CV not found for this user.")

    if not cv_path.exists() or not cv_path.is_file():
        raise HTTPException(status_code=500, detail=f"Configured CV missing: {cv_path}")

    try:
        cv_text = read_cv_text(cv_path)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to read CV: {err}") from err

    return {
        "user_id": user_id,
        "cv_path": str(cv_path),
        "cv_text": cv_text,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.mock_backend:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )
