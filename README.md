## Prerequisites
- Python 3.10 or newer

## Setup
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Copy `.env.example` to `.env` (or edit `.env`) and set your Google Gemini API key (`GOOGLE_API_KEY`, `GOOGLE_GENAI_API_KEY`, or `API_KEY`).

## Usage
- Start the FastAPI service:
  ```bash
  uvicorn main:app --reload
  ```
- Interact with the API (examples):
  - Check health: `GET http://127.0.0.1:8000/health`
  - Initialize matcher: `POST http://127.0.0.1:8000/init`
  - Upload CV: `POST http://127.0.0.1:8000/upload`
  - Chat: `POST http://127.0.0.1:8000/chat`
