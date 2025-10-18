"""Entry point exposing the FastAPI app."""

from app import app


def main() -> None:
    """Run the FastAPI application with uvicorn."""
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
