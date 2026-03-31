"""Configuration loading and validation."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from conf_briefing.console import console, tag


@dataclass
class RecordingsConfig:
    source_url: str = ""  # playlist/channel URL (provider auto-detected from domain)
    video_ids: list[str] = field(default_factory=list)
    video_format: str = "mp4"


@dataclass
class ConferenceConfig:
    name: str = "Conference"
    schedule: str = ""
    schedule_url: str = ""
    sched_api_key: str = ""
    recordings: RecordingsConfig = field(default_factory=RecordingsConfig)


@dataclass
class LLMConfig:
    model: str = "qwen3:14b"
    ollama_base_url: str = "http://localhost:11434"


_DEFAULT_INITIAL_PROMPT = (
    "KubeCon, Kubernetes, Istio, eBPF, Cilium, Prometheus, gRPC, Envoy, "
    "ArgoCD, Helm, containerd, CRI-O, Knative, Backstage, OpenTelemetry, Crossplane"
)


@dataclass
class ExtractConfig:
    whisper_model: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    device: str = "auto"  # "auto", "cuda", "cpu"
    compute_type: str = "auto"  # "auto", "float16", "int8"
    scene_threshold: float = 27.0  # PySceneDetect ContentDetector threshold
    initial_prompt: str = _DEFAULT_INITIAL_PROMPT  # domain-specific terminology for transcription
    vlm_model: str = ""  # empty = skip VLM; e.g. "gemma3:12b"


@dataclass
class QueryConfig:
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    top_k: int = 15


@dataclass
class Config:
    conference: ConferenceConfig = field(default_factory=ConferenceConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    data_dir: Path = Path("data")


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
    data_dir = path.with_suffix("")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Warn about unknown top-level keys
    known_sections = {"conference", "extract", "llm", "query"}
    for key in raw:
        if key not in known_sections:
            console.print(
                f"{tag('config')} [yellow]Warning: unknown config section "
                f"\\[{key}] ignored[/yellow]"
            )

    conf_raw = raw.get("conference", {})
    _warn_unknown(
        conf_raw,
        {"name", "schedule", "schedule_url", "sched_api_key", "recordings"},
        "conference",
    )

    recordings_raw = conf_raw.get("recordings", {})
    _warn_unknown(
        recordings_raw,
        {"source_url", "video_ids", "video_format"},
        "conference.recordings",
    )
    recordings = RecordingsConfig(
        source_url=recordings_raw.get("source_url", ""),
        video_ids=recordings_raw.get("video_ids", []),
        video_format=recordings_raw.get("video_format", "mp4"),
    )

    conference = ConferenceConfig(
        name=conf_raw.get("name", "Conference"),
        schedule=conf_raw.get("schedule", ""),
        schedule_url=conf_raw.get("schedule_url", ""),
        sched_api_key=conf_raw.get("sched_api_key", "") or os.environ.get("SCHED_API_KEY", ""),
        recordings=recordings,
    )

    extract_raw = raw.get("extract", {})
    _warn_unknown(
        extract_raw,
        {"whisper_model", "device", "compute_type", "scene_threshold", "initial_prompt", "vlm_model"},
        "extract",
    )
    extract = ExtractConfig(
        whisper_model=extract_raw.get("whisper_model", "deepdml/faster-whisper-large-v3-turbo-ct2"),
        device=extract_raw.get("device", "auto"),
        compute_type=extract_raw.get("compute_type", "auto"),
        scene_threshold=extract_raw.get("scene_threshold", 27.0),
        initial_prompt=extract_raw.get("initial_prompt", _DEFAULT_INITIAL_PROMPT),
        vlm_model=extract_raw.get("vlm_model", ""),
    )

    llm_raw = raw.get("llm", {})
    _warn_unknown(llm_raw, {"model", "ollama_base_url"}, "llm")
    llm = LLMConfig(
        model=llm_raw.get("model", "qwen3:14b"),
        ollama_base_url=llm_raw.get("ollama_base_url", "http://localhost:11434"),
    )

    query_raw = raw.get("query", {})
    _warn_unknown(query_raw, {"embedding_model", "ollama_base_url", "top_k"}, "query")
    query = QueryConfig(
        embedding_model=query_raw.get("embedding_model", "nomic-embed-text"),
        ollama_base_url=query_raw.get("ollama_base_url", "http://localhost:11434"),
        top_k=query_raw.get("top_k", 15),
    )

    return Config(
        conference=conference,
        extract=extract,
        llm=llm,
        query=query,
        data_dir=data_dir,
    )


def _warn_unknown(raw: dict, known: set[str], section: str) -> None:
    """Warn about unknown keys in a config section."""
    unknown = set(raw.keys()) - known
    if unknown:
        console.print(
            f"{tag('config')} [yellow]Warning: unknown \\[{section}] keys ignored: "
            f"{', '.join(sorted(unknown))}[/yellow]"
        )
