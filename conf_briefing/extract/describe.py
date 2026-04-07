"""VLM slide description: describe visual content of extracted slides."""

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama

from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file

_VLM_PROMPT = (
    "Describe the technical content of this conference presentation slide. "
    "Focus on diagrams, architecture, data flow, and key visual elements. "
    "Note any visible URLs, GitHub links, or QR codes. Be concise."
)

_RETRY_DELAYS = (2, 5, 10)
_RETRYABLE_ERRORS = (ollama.ResponseError, ConnectionError, TimeoutError, OSError)


def describe_slide(client: ollama.Client, model: str, image_path: Path) -> str:
    """Send a single slide image to a VLM and return the description."""
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": _VLM_PROMPT,
                        "images": [image_b64],
                    },
                ],
                options={"num_predict": 512},
            )
            text = (response.message.content or "").strip()
            if not text:
                return "[no content]"  # Non-slide frame; sentinel prevents retry
            return text
        except _RETRYABLE_ERRORS as exc:
            last_exc = exc
            if attempt < len(_RETRY_DELAYS):
                delay = _RETRY_DELAYS[attempt]
                console.print(
                    f"  {tag('vlm')} Retry {attempt + 1}/{len(_RETRY_DELAYS)} "
                    f"after {type(exc).__name__}, waiting {delay}s..."
                )
                time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def describe_all_slides(config: Config) -> list[Path]:
    """Add VLM descriptions to all slide JSONs that lack them.

    Iterates slide JSON files, skips slides that already have a non-empty
    description, calls the VLM for the rest, and writes back the updated JSON.
    Returns list of updated slide JSON paths.
    """
    vlm_model = config.extract.vlm_model
    if not vlm_model:
        return []

    data_dir = config.data_dir
    slides_dir = data_dir / "slides"

    if not slides_dir.exists():
        console.print(f"{tag('vlm')} No slides directory found, skipping VLM descriptions.")
        return []

    slide_jsons = sorted(slides_dir.glob("*.json"))
    if not slide_jsons:
        console.print(f"{tag('vlm')} No slide JSON files found.")
        return []

    client = ollama.Client(host=config.llm.ollama_base_url)
    num_parallel = config.llm.num_parallel

    def _describe_one(idx_slide: tuple[int, dict]) -> tuple[int, str | None, str]:
        idx, slide = idx_slide
        image_file = slide.get("image_file", "")
        image_path = data_dir / image_file
        if not image_path.exists():
            return idx, None, f"image not found ({image_file})"
        t0 = time.monotonic()
        try:
            desc = describe_slide(client, vlm_model, image_path)
        except Exception as exc:
            return idx, None, str(exc)
        return idx, desc, f"{time.monotonic() - t0:.1f}s"

    updated: list[Path] = []
    total_files = len(slide_jsons)

    for fi, json_path in enumerate(slide_jsons, 1):
        data = load_json_file(json_path)
        slides = data.get("slides", [])

        # Count slides needing descriptions
        to_describe = [
            (i, s) for i, s in enumerate(slides) if not s.get("description")
        ]

        if not to_describe:
            continue

        console.print(
            f"{tag('vlm')} [{fi}/{total_files}] {json_path.stem}: "
            f"describing {len(to_describe)} slide(s) "
            f"({num_parallel} parallel)..."
        )

        described = 0

        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            futures = {
                executor.submit(_describe_one, item): item for item in to_describe
            }
            for future in as_completed(futures):
                idx, desc, info = future.result()
                described += 1
                if desc:
                    slides[idx]["description"] = desc
                    console.print(
                        f"  {tag('vlm')} [{described}/{len(to_describe)}] "
                        f"slide {idx} ({info})"
                    )
                else:
                    console.print(
                        f"  {tag('vlm')} [{described}/{len(to_describe)}] "
                        f"slide {idx} [red]failed[/red]: {info}"
                    )

        # Write back
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        updated.append(json_path)

    if updated:
        console.print(
            f"{tag('vlm')} Described slides in {len(updated)} file(s)."
        )
    else:
        console.print(f"{tag('vlm')} All slides already have descriptions.")

    return updated
