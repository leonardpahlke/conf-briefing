"""Video extraction: transcription and slide OCR."""

from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.extract.preflight import check_extract_ready
from conf_briefing.extract.slides import extract_all_slides
from conf_briefing.extract.transcribe import transcribe_all

__all__ = ["check_extract_ready", "extract_all_slides", "run_extract", "transcribe_all"]


def run_extract(config: Config) -> None:
    """Run video extraction: transcription + slide OCR."""
    console.rule("[bold cyan]Extract[/bold cyan]")
    device, compute_type = check_extract_ready(config)
    transcribe_all(config, device=device, compute_type=compute_type)
    extract_all_slides(config)
    console.print(f"{tag('extract')} Done.")
