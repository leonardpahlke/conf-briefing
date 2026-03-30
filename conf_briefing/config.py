"""Configuration loading and validation."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RecordingsConfig:
    youtube_playlist: str = ""
    video_ids: list[str] = field(default_factory=list)
    strategy: str = "api"  # "api" (youtube subtitles) or "local" (yt-dlp + whisper)
    whisper_model: str = "base"  # tiny, base, small, medium, large-v3


@dataclass
class ConferenceConfig:
    name: str = "Conference"
    schedule: str = ""
    schedule_url: str = ""
    sched_api_key: str = ""
    recordings: RecordingsConfig = field(default_factory=RecordingsConfig)


@dataclass
class LLMConfig:
    model: str = "claude-sonnet-4-20250514"


@dataclass
class QueryConfig:
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    top_k: int = 15


@dataclass
class Config:
    conference: ConferenceConfig = field(default_factory=ConferenceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    data_dir: str = "data"


def load_config(path: str | Path) -> Config:
    """Load configuration from a TOML file.

    The data directory is derived from the config file path by stripping .toml:
      events/kubecon-eu-2026.toml → events/kubecon-eu-2026/
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Derive data_dir from config file path (strip .toml suffix)
    data_dir = str(path.with_suffix(""))
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    recordings_raw = raw.get("conference", {}).get("recordings", {})
    recordings = RecordingsConfig(
        youtube_playlist=recordings_raw.get("youtube_playlist", ""),
        video_ids=recordings_raw.get("video_ids", []),
        strategy=recordings_raw.get("strategy", "api"),
        whisper_model=recordings_raw.get("whisper_model", "base"),
    )

    conf_raw = raw.get("conference", {})
    conference = ConferenceConfig(
        name=conf_raw.get("name", "Conference"),
        schedule=conf_raw.get("schedule", ""),
        schedule_url=conf_raw.get("schedule_url", ""),
        sched_api_key=conf_raw.get("sched_api_key", "") or os.environ.get("SCHED_API_KEY", ""),
        recordings=recordings,
    )

    llm_raw = raw.get("llm", {})
    llm = LLMConfig(
        model=llm_raw.get("model", "claude-sonnet-4-20250514"),
    )

    query_raw = raw.get("query", {})
    query = QueryConfig(
        embedding_model=query_raw.get("embedding_model", "nomic-embed-text"),
        ollama_base_url=query_raw.get("ollama_base_url", "http://localhost:11434"),
        top_k=query_raw.get("top_k", 15),
    )

    return Config(
        conference=conference,
        llm=llm,
        query=query,
        data_dir=data_dir,
    )
