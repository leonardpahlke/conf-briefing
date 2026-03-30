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
# sched_api_key = "${SCHED_API_KEY}"

[conference.recordings]
youtube_playlist = "https://www.youtube.com/playlist?list=PLj6h78..."

[llm]
model = "claude-sonnet-4-20250514"

[query]
embedding_model = "nomic-embed-text"
ollama_base_url = "http://localhost:11434"
top_k = 15
```

## Conference

| Field          | Description                                                       |
|----------------|-------------------------------------------------------------------|
| `name`         | Display name used in report headers.                              |
| `schedule`     | Path to schedule data file (titles, abstracts, speakers, tracks). |
| `schedule_url` | URL to a schedule provider (e.g. sched.com). Auto-detected.      |
| `sched_api_key`| API key for Sched (if fetching programmatically).                 |

### Recordings

| Field              | Description                                              | Default  |
|--------------------|----------------------------------------------------------|----------|
| `youtube_playlist` | YouTube playlist URL for talk recordings.                 |          |
| `video_ids`        | List of individual YouTube video IDs.                     |          |
| `strategy`         | `"api"` (YouTube subtitles) or `"local"` (yt-dlp + Whisper). | `"api"`  |
| `whisper_model`    | Whisper model size when using local strategy.             | `"base"` |

**Local strategy** downloads audio via yt-dlp and transcribes with OpenAI Whisper. Works on any language, no API limits. Requires the `local` extra:

```sh
uv sync --extra local
```

Whisper models from smallest/fastest to largest/most accurate: `tiny`, `base`, `small`, `medium`, `large-v3`. Works on NVIDIA (CUDA), AMD (ROCm), and CPU.

## LLM

| Field   | Description                       | Default                    |
|---------|-----------------------------------|----------------------------|
| `model` | Claude model to use for analysis. | `claude-sonnet-4-20250514` |

The Anthropic API key is read from the `ANTHROPIC_API_KEY` environment variable.

## RAG Query

Settings for the vector-based retrieval system. Requires [Ollama](https://ollama.com/) running locally.

| Field             | Description                                          | Default                    |
|-------------------|------------------------------------------------------|----------------------------|
| `embedding_model` | Ollama model used for embeddings.                    | `nomic-embed-text`         |
| `ollama_base_url` | Base URL of the Ollama server.                       | `http://localhost:11434`   |
| `top_k`           | Default number of chunks to retrieve per query.      | `15`                       |

The ChromaDB vector store is located at `{data_dir}/chroma/` (derived automatically from the config path).
