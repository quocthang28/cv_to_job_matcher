"""Shared configuration and constants for the CV matcher service."""

from pathlib import Path

from dotenv import load_dotenv

# Ensure environment variables from `.env` are available across modules.
load_dotenv()

PERSIST_DIR = Path("job_vectorstore")
REVIEW_INSTRUCTIONS_PATH = Path("review_instructions.md")

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_K = 5

CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant that supports potential candidates in finding suitable jobs. "
    "Whenever the user shares a CV or asks for job suggestions, call the available tool to "
    "retrieve relevant job matches and present concise summaries with job title, company, "
    "location, similarity score, and job ID. If the user request is unrelated to job matching "
    "or the provided CV, respond directly without invoking the tool."
)
