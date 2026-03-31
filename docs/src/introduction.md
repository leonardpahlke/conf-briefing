# Introduction

Conf Briefing is an AI-powered conference analysis tool. It helps you understand what a conference covers, which talks to attend, and what insights came out of the recorded sessions — without manually reviewing hundreds of abstracts and hours of video.

## Event-based Structure

Each conference is managed as an event with its own config file and data directory:

```
events/kubecon-eu-2026.toml   → configuration (tracked in git)
events/kubecon-eu-2026/       → generated data (gitignored)
```

## What It Does

- **Agenda analysis** — extracts topics, keywords, company presence, and clusters talks into themes from the published schedule.
- **Cluster ranking** — ranks clusters by relevance with talk shortlists so you can pick what to attend.
- **Recording analysis** — transcribes talks, extracts slides (with optional VLM descriptions of diagrams), and produces per-talk summaries, Q&A highlights, and emerging signals.
- **RAG query** — indexes all outputs into a local vector store for follow-up questions with source references.

## Workflow

1. **Configure** — create an event config under `events/` (see [Configuration](./configuration.md)).
2. **Pull models** — `just pull-models <event>` downloads required Ollama models (LLM, embeddings, VLM).
3. **Collect** — `just collect <event>` fetches schedule data and downloads recordings.
4. **Extract** — `just extract <event>` transcribes audio, extracts slides, describes visuals, and runs LLM analysis.
5. **Report** — `just report <event>` generates markdown reports and charts.
6. **Query** — `just query <event> "your question"` for follow-up questions over all analysis data.

See [Generation](./generation.md) for details on each step.
