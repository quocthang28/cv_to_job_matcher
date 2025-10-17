"""Entry point exposing the FastAPI app and CLI runner."""

from app import app


def main() -> None:
    from app.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
