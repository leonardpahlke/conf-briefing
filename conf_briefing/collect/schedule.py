"""Fetch conference schedule from various providers or a local file."""

import json
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import requests

from conf_briefing.config import Config
from conf_briefing.console import console, tag

# Map URL domain patterns to scraper modules.
# Each scraper must expose: scrape_schedule(url: str) -> list[dict]
PROVIDER_PATTERNS: list[tuple[str, str]] = [
    ("sched.com", "conf_briefing.collect.sched_scraper"),
    # ("meetup.com", "conf_briefing.collect.meetup_scraper"),
]


def _resolve_provider(url: str) -> tuple[str, str] | None:
    """Match a URL to a schedule provider.

    Returns (provider_name, module_path) or None.
    """
    hostname = urlparse(url).hostname or ""
    for domain, module in PROVIDER_PATTERNS:
        if hostname.endswith(domain):
            return domain, module
    return None


def _scrape_from_provider(url: str, cache_dir: Path | None = None) -> list[dict]:
    """Dispatch to the right scraper based on URL domain."""
    match = _resolve_provider(url)
    if match is None:
        raise ValueError(
            f"No schedule provider found for URL: {url}\n"
            f"Supported: {', '.join(d for d, _ in PROVIDER_PATTERNS)}"
        )

    import importlib

    provider_name, module_path = match
    mod = importlib.import_module(module_path)
    console.print(f"{tag('collect')} Using {provider_name} provider for schedule")
    return mod.scrape_schedule(url, cache_dir=cache_dir)


def fetch_from_sched_api(url: str, api_key: str) -> list[dict]:
    """Fetch sessions from the Sched API (requires paid plan key)."""
    endpoint = f"{url.rstrip('/')}/api/session/list"
    params = {"api_key": api_key, "format": "json", "fields": "name,description,speakers,venue"}
    resp = requests.get(endpoint, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def load_from_file(path: str | Path) -> list[dict]:
    """Load schedule from a local TOML or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Schedule file not found: {path}")

    if path.suffix == ".toml":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return data.get("sessions", [])
    return json.loads(path.read_text())


def coerce_session(raw: dict) -> dict:
    """Coerce a raw session from various provider formats into a consistent schema."""
    speakers = raw.get("speakers", [])
    if isinstance(speakers, str):
        speakers = [{"name": s.strip(), "company": ""} for s in speakers.split(",")]
    elif isinstance(speakers, list):
        speakers = [s if isinstance(s, dict) else {"name": str(s), "company": ""} for s in speakers]

    return {
        "title": raw.get("title") or raw.get("name", ""),
        "abstract": raw.get("abstract") or raw.get("description", ""),
        "speakers": speakers,
        "track": raw.get("track") or raw.get("venue", ""),
        "format": raw.get("format", ""),
        "time": raw.get("time") or raw.get("event_start", ""),
        "tags": raw.get("tags", []),
    }


def fetch_schedule(config: Config) -> Path:
    """Fetch or load schedule and save as normalized JSON.

    Resolution order:
    1. schedule_url + sched_api_key → Sched API (fast, requires paid key)
    2. schedule_url alone → auto-detect provider and scrape public page
    3. schedule file path → load from local JSON/TOML
    """
    data_dir = config.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "schedule.json"

    if config.conference.schedule_url and config.conference.sched_api_key:
        console.print(f"{tag('collect')} Fetching schedule from Sched API...")
        raw_sessions = fetch_from_sched_api(
            config.conference.schedule_url, config.conference.sched_api_key
        )
    elif config.conference.schedule_url:
        talks_dir = data_dir / "talks"
        raw_sessions = _scrape_from_provider(config.conference.schedule_url, cache_dir=talks_dir)
    elif config.conference.schedule:
        console.print(f"{tag('collect')} Loading schedule from file: {config.conference.schedule}")
        raw_sessions = load_from_file(config.conference.schedule)
    else:
        console.print(f"{tag('collect')} No schedule source configured, skipping.")
        return out_path

    sessions = [coerce_session(s) for s in raw_sessions]
    out_path.write_text(json.dumps(sessions, indent=2, ensure_ascii=False))
    console.print(f"{tag('collect')} Saved {len(sessions)} sessions to {out_path}")
    return out_path
