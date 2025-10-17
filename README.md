## Prerequisites
- Python 3.10 or newer

## Setup
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Copy `.env.example` to `.env` (or edit `.env`) and set your Google Gemini API key (`GOOGLE_API_KEY`, `GOOGLE_GENAI_API_KEY`, or `API_KEY`).

## Usage
- Run the matcher:
  ```bash
  python3 main.py --cv cv/cv{1/2/3}.md
  ```
