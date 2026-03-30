"""LLM client abstraction."""

import json

import anthropic

from conf_briefing.config import Config


def get_client() -> anthropic.Anthropic:
    """Create an Anthropic client (reads ANTHROPIC_API_KEY env var)."""
    return anthropic.Anthropic()


def query_llm(config: Config, system: str, prompt: str, max_tokens: int = 4096) -> str:
    """Send a prompt to the LLM and return the text response."""
    client = get_client()
    message = client.messages.create(
        model=config.llm.model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def query_llm_json(config: Config, system: str, prompt: str, max_tokens: int = 4096) -> dict:
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
