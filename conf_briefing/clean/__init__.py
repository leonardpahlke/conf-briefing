"""Data cleaning and normalization."""

from pathlib import Path

from conf_briefing.clean.normalize import match_transcripts, normalize_schedule
from conf_briefing.config import Config
from conf_briefing.console import console, tag

__all__ = ["match_transcripts", "normalize_schedule", "run_clean"]


def _newest_mtime(paths: list[Path]) -> float:
    """Return the newest mtime among *paths*, or 0 if none exist."""
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes, default=0)


def _is_clean_fresh(config: Config) -> bool:
    """Return True when matched.json is newer than every input file."""
    data_dir = config.data_dir
    out = data_dir / "matched.json"
    if not out.exists():
        return False
    out_mtime = out.stat().st_mtime

    # Input files: schedule.json + all transcript/slide JSONs
    inputs: list[Path] = []
    schedule = data_dir / "schedule.json"
    if schedule.exists():
        inputs.append(schedule)
    for subdir in ("transcripts", "slides"):
        d = data_dir / subdir
        if d.exists():
            inputs.extend(d.glob("*.json"))

    if not inputs:
        return True  # nothing to process

    return _newest_mtime(inputs) < out_mtime


def run_clean(config: Config) -> None:
    """Run the full cleaning pipeline."""
    console.rule("[bold green]Clean[/bold green]")
    if _is_clean_fresh(config):
        console.print(f"{tag('clean')} Up-to-date, skipping.")
        return
    console.print(f"{tag('clean')} Cleaning data for: {config.conference.name}")
    normalize_schedule(config)
    match_transcripts(config)
    console.print(f"{tag('clean')} Done.")
