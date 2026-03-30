"""Video extraction: transcription and slide OCR."""

from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.extract.preflight import ExtractContext, check_extract_ready
from conf_briefing.extract.slides import extract_all_slides
from conf_briefing.extract.transcribe import transcribe_all
from conf_briefing.extract.transcribe_whisper_cpp import transcribe_all_wcpp

__all__ = [
    "ExtractContext",
    "check_extract_ready",
    "extract_all_slides",
    "run_extract",
    "transcribe_all",
    "transcribe_all_wcpp",
]


def run_extract(config: Config) -> None:
    """Run video extraction: transcription + slide OCR + optional VLM."""
    console.rule("[bold cyan]Extract[/bold cyan]")
    ctx = check_extract_ready(config)

    # Transcription — dispatch to best available backend
    if ctx.transcribe_backend == "whisper-cpp":
        transcribe_all_wcpp(
            config,
            model_path=ctx.wcpp_model_path,
            wcpp_binary=ctx.wcpp_binary,
            initial_prompt=ctx.initial_prompt,
        )
    else:
        transcribe_all(
            config,
            device=ctx.device,
            compute_type=ctx.compute_type,
            initial_prompt=ctx.initial_prompt,
        )

    # Slide extraction (scene detection + Tesseract OCR)
    extract_all_slides(config)

    console.print(f"{tag('extract')} Done.")
