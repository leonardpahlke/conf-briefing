# Configuration

Each conference event is defined by a TOML file under `events/`. The data directory is derived automatically from the config file path:

```
events/kubecon-eu-2026.toml   → config file (tracked in git)
events/kubecon-eu-2026/       → data directory (gitignored)
```

## Example

```toml
[conference]
name = "KubeCon EU 2026"
schedule_url = "https://kccnceu2026.sched.com"

[conference.recordings]
source_url = "https://www.youtube.com/playlist?list=PLj6h78..."

[llm]
model = "qwen3:14b"
ollama_base_url = "http://localhost:11434"

[query]
embedding_model = "nomic-embed-text"
top_k = 15
```

## Conference

| Field          | Description                                                       |
|----------------|-------------------------------------------------------------------|
| `name`         | Display name used in report headers.                              |
| `schedule`     | Path to local schedule data file (JSON/TOML).                     |
| `schedule_url` | URL to a schedule provider (e.g. sched.com). Auto-detected.      |
| `sched_api_key`| API key for Sched (optional, falls back to scraping).             |

### Recordings

| Field          | Description                                                    | Default |
|----------------|----------------------------------------------------------------|---------|
| `source_url`   | Playlist/channel URL for talk recordings (YouTube).            |         |
| `video_ids`    | List of individual video IDs.                                  |         |
| `video_format` | Video format to download.                                      | `mp4`   |

## LLM

Analysis uses [Ollama](https://ollama.com/) running locally.

| Field             | Description                        | Default                  |
|-------------------|------------------------------------|--------------------------|
| `model`           | Ollama model for analysis.         | `qwen3:14b`              |
| `ollama_base_url` | Base URL of the Ollama server.     | `http://localhost:11434` |

## Query

Vector-based retrieval for RAG queries. Also uses Ollama for embeddings.

| Field             | Description                                     | Default                  |
|-------------------|-------------------------------------------------|--------------------------|
| `embedding_model` | Ollama model for embeddings.                    | `nomic-embed-text`       |
| `ollama_base_url` | Base URL of the Ollama server.                  | `http://localhost:11434` |
| `top_k`           | Default number of chunks to retrieve per query. | `15`                     |
