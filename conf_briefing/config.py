"""Configuration loading and validation."""

import os
import tomllib
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path

from conf_briefing.console import console, tag

# Minimum video duration (seconds) to consider for analysis.
# Videos shorter than this are assumed to be highlight reels or teasers.
MIN_VIDEO_DURATION_SEC = 120


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
    num_parallel: int = 2  # concurrent Ollama request slots (set OLLAMA_NUM_PARALLEL to match)


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
    diarize: bool = True  # enable speaker diarization (requires whisperx + HF_TOKEN)
    language: str = "en"  # language code for transcription


@dataclass
class QueryConfig:
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    top_k: int = 15


_DEFAULT_EVAL_TOPICS = [
    "general industry trends",
    "emerging technology",
    "practical applicability",
]


@dataclass
class AnalyzeConfig:
    eval_topics: list[str] = field(default_factory=lambda: list(_DEFAULT_EVAL_TOPICS))


@dataclass
class Config:
    conference: ConferenceConfig = field(default_factory=ConferenceConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    analyze: AnalyzeConfig = field(default_factory=AnalyzeConfig)
    data_dir: Path = Path("data")


def _field_names(cls) -> set[str]:
    """Return the set of field names for a dataclass."""
    return {f.name for f in dataclass_fields(cls)}


def _build_dataclass(cls, raw: dict):
    """Build a dataclass instance, passing only keys that match field names.

    Missing keys use the dataclass field defaults.
    """
    valid_keys = _field_names(cls)
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return cls(**filtered)


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
    known_sections = _field_names(Config) - {"data_dir"}
    for key in raw:
        if key not in known_sections:
            console.print(
                f"{tag('config')} [yellow]Warning: unknown config section "
                f"\\[{key}] ignored[/yellow]"
            )

    conf_raw = raw.get("conference", {})
    _warn_unknown(
        conf_raw,
        _field_names(ConferenceConfig) | {"recordings"},
        "conference",
    )

    recordings_raw = conf_raw.get("recordings", {})
    _warn_unknown(
        recordings_raw,
        _field_names(RecordingsConfig),
        "conference.recordings",
    )
    recordings = _build_dataclass(RecordingsConfig, recordings_raw)

    conf_flat = {k: v for k, v in conf_raw.items() if k != "recordings"}
    conference = _build_dataclass(ConferenceConfig, conf_flat)
    conference.recordings = recordings
    if not conference.sched_api_key:
        conference.sched_api_key = os.environ.get("SCHED_API_KEY", "")

    extract_raw = raw.get("extract", {})
    _warn_unknown(extract_raw, _field_names(ExtractConfig), "extract")
    extract = _build_dataclass(ExtractConfig, extract_raw)

    llm_raw = raw.get("llm", {})
    _warn_unknown(llm_raw, _field_names(LLMConfig), "llm")
    llm = _build_dataclass(LLMConfig, llm_raw)

    query_raw = raw.get("query", {})
    _warn_unknown(query_raw, _field_names(QueryConfig), "query")
    query = _build_dataclass(QueryConfig, query_raw)

    analyze_raw = raw.get("analyze", {})
    _warn_unknown(analyze_raw, _field_names(AnalyzeConfig), "analyze")
    analyze = _build_dataclass(AnalyzeConfig, analyze_raw)

    return Config(
        conference=conference,
        extract=extract,
        llm=llm,
        query=query,
        analyze=analyze,
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
