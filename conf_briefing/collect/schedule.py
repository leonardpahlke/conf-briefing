"""Fetch conference schedule from Sched API or load from local file."""

import json
import tomllib
from pathlib import Path

import requests

from conf_briefing.config import Config


def fetch_from_sched(url: str, api_key: str) -> list[dict]:
    """Fetch sessions from the Sched API."""
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


def normalize_session(raw: dict) -> dict:
    """Normalize a raw session into a consistent schema."""
    speakers = raw.get("speakers", [])
    if isinstance(speakers, str):
        speakers = [{"name": s.strip(), "company": ""} for s in speakers.split(",")]
    elif isinstance(speakers, list):
        speakers = [
            s if isinstance(s, dict) else {"name": str(s), "company": ""}
            for s in speakers
        ]

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
    """Fetch or load schedule and save as normalized JSON."""
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "schedule.json"

    if config.conference.sched_url and config.conference.sched_api_key:
        print("[collect] Fetching schedule from Sched API...")
        raw_sessions = fetch_from_sched(
            config.conference.sched_url, config.conference.sched_api_key
        )
    elif config.conference.sched_url:
        print(f"[collect] sched_url is set ({config.conference.sched_url})")
        print(f"[collect] Get your API key at: {config.conference.sched_url.rstrip('/')}/editor/export/api")
        api_key = input("[collect] Paste your Sched API key (or press Enter to skip): ").strip()
        if not api_key:
            print("[collect] No API key provided, skipping schedule fetch.")
            return out_path
        print("[collect] Fetching schedule from Sched API...")
        raw_sessions = fetch_from_sched(config.conference.sched_url, api_key)
    elif config.conference.schedule:
        print(f"[collect] Loading schedule from file: {config.conference.schedule}")
        raw_sessions = load_from_file(config.conference.schedule)
    else:
        print("[collect] No schedule source configured, skipping.")
        return out_path

    sessions = [normalize_session(s) for s in raw_sessions]
    out_path.write_text(json.dumps(sessions, indent=2, ensure_ascii=False))
    print(f"[collect] Saved {len(sessions)} sessions to {out_path}")
    return out_path
