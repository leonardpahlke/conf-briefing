"""LLM client abstraction using Ollama."""

import json

import ollama

from conf_briefing.config import Config


def is_llm_available(config: Config) -> bool:
    """Check if the Ollama server is reachable and the model is available."""
    try:
        client = ollama.Client(host=config.llm.ollama_base_url)
        client.show(config.llm.model)
        return True
    except Exception:
        return False


def query_llm(config: Config, system: str, prompt: str, max_tokens: int = 4096) -> str:
    """Send a prompt to the LLM via Ollama and return the text response."""
    client = ollama.Client(host=config.llm.ollama_base_url)
    response = client.chat(
        model=config.llm.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"num_predict": max_tokens},
    )
    text = response.message.content
    if not text:
        raise ValueError("LLM returned empty response")
    return text


def query_llm_json(config: Config, system: str, prompt: str, max_tokens: int = 4096) -> dict | list:
    """Send a prompt to the LLM and parse the JSON response."""
    response = query_llm(config, system, prompt, max_tokens)
    # Extract JSON from response (handle markdown code blocks)
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # remove closing fence
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON: {e}\nResponse (first 500 chars): {text[:500]}"
        ) from e
