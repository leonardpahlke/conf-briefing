"""LLM client abstraction using Ollama."""

import json
import re
import time

import ollama
from pydantic import BaseModel

from conf_briefing.config import Config
from conf_briefing.console import console, tag

_RETRY_DELAYS = (2, 5, 10)
_RETRYABLE_ERRORS = (ollama.ResponseError, ConnectionError, TimeoutError, OSError)
_client_cache: dict[str, ollama.Client] = {}


def is_llm_available(config: Config) -> bool:
    """Check if the Ollama server is reachable and the model is available."""
    try:
        client = ollama.Client(host=config.llm.ollama_base_url)
        client.show(config.llm.model)
        return True
    except Exception:
        return False


def query_llm(
    config: Config,
    system: str,
    prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0,
    response_format: dict | None = None,
) -> str:
    """Send a prompt to the LLM via Ollama and return the text response.

    Retries up to 3 times on transient errors with exponential backoff.
    """
    base_url = config.llm.ollama_base_url
    if base_url not in _client_cache:
        _client_cache[base_url] = ollama.Client(host=base_url)
    client = _client_cache[base_url]
    last_exc: Exception | None = None

    kwargs: dict = {
        "model": config.llm.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }
    if response_format is not None:
        kwargs["format"] = response_format

    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            response = client.chat(**kwargs)
            text = response.message.content
            if not text:
                raise ValueError("LLM returned empty response")
            return text
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            if attempt < len(_RETRY_DELAYS):
                delay = _RETRY_DELAYS[attempt]
                console.print(
                    f"  {tag('ollama')} Retry {attempt + 1}/{len(_RETRY_DELAYS)} "
                    f"after {type(exc).__name__}, waiting {delay}s..."
                )
                time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def _extract_json_from_text(text: str) -> str:
    """Extract JSON from LLM response text.

    Handles: markdown fenced blocks, raw JSON objects/arrays.
    """
    # Try fenced code block first (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find a raw JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        end = text.rfind(end_char)
        if end > start:
            return text[start : end + 1]

    return text.strip()


def query_llm_json(
    config: Config,
    system: str,
    prompt: str,
    *,
    max_tokens: int = 4096,
    schema: type[BaseModel] | None = None,
    temperature: float = 0,
) -> dict | list:
    """Send a prompt to the LLM and parse the JSON response.

    When a Pydantic schema is provided, uses Ollama's structured output via the
    `format` parameter — this guarantees valid JSON matching the schema, so no
    manual extraction or retry is needed.

    Falls back to extraction + retry when no schema is given.
    """
    if schema is not None:
        # Structured output: Ollama guarantees valid JSON matching the schema
        response = query_llm(
            config,
            system,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=schema.model_json_schema(),
        )
        return json.loads(response)

    # Fallback: extract JSON from free-form LLM text
    for attempt in range(2):
        response = query_llm(
            config, system, prompt, max_tokens=max_tokens, temperature=temperature
        )
        extracted = _extract_json_from_text(response)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            if attempt == 0:
                console.print(
                    f"  {tag('ollama')} JSON parse failed, retrying query..."
                )
                continue
            raise ValueError(
                f"LLM returned invalid JSON after retry.\n"
                f"Response (first 500 chars): {response[:500]}"
            ) from None

    raise ValueError("LLM JSON query failed")  # unreachable, satisfies type checker
