"""Shared configuration and constants for the CV matcher service."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Ensure environment variables from `.env` are available across modules.
load_dotenv()

PERSIST_DIR = Path("job_vectorstore")
REVIEW_INSTRUCTIONS_PATH = Path("review_instructions.md")

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_K = 5

INTENT_CLASSIFIER_TEMPERATURE = 0.0
CV_REVIEW_TEMPERATURE = 0.4

CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant that supports potential candidates in finding suitable jobs. "
    "Whenever the user shares a CV or asks for job suggestions, call the available tool to "
    "retrieve relevant job matches and present concise summaries with job title, company, "
    "location, similarity score, and job ID. If the user request is unrelated to job matching "
    "or the provided CV, respond with your intentions as an HR assistant without invoking the tool."
)

@lru_cache(maxsize=None)
def _build_chat_model(
    model_name: str,
    temperature: float,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

def get_chat_model(
    *,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatOpenAI:
    """Return a cached ChatOpenAI client with the desired configuration."""
    resolved_model = model_name or DEFAULT_CHAT_MODEL
    resolved_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    return _build_chat_model(resolved_model, resolved_temperature)


def get_intent_classifier_model() -> ChatOpenAI:
    """Return the shared client used for intent classification."""
    return get_chat_model(temperature=INTENT_CLASSIFIER_TEMPERATURE)


def get_cv_reviewer_model() -> ChatOpenAI:
    """Return the shared client used for CV review feedback."""
    return get_chat_model(temperature=CV_REVIEW_TEMPERATURE)
