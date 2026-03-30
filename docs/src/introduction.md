# Introduction

Conf Briefing is an AI-powered conference analysis tool. It helps you understand what a conference covers, which talks to attend, and what insights came out of the recorded sessions — without manually reviewing hundreds of abstracts and hours of video.

## Event-based Structure

Each conference is managed as an event with its own config file and data directory:

```
events/kubecon-eu-2026.toml   → configuration (tracked in git)
events/kubecon-eu-2026/       → generated data (gitignored)
```

## Agenda Analysis

Analyze all talks on the conference agenda. Starting from the published abstracts, the tool extracts topics, keywords, company presence, and clusters talks into coherent themes. The output is a landscape overview of the conference.

## Cluster Ranking

Based on the identified clusters, the tool ranks them and argues why each might be interesting. The reader picks the clusters they care about and gets a focused shortlist of talks.

## Recording Analysis

After the conference, when talk recordings are published on YouTube, the tool downloads transcripts and analyzes them — extracting key takeaways, audience Q&A, problems discussed, and emerging technology signals.

## RAG Query

With hundreds of talks and hours of transcripts, the full analysis data far exceeds what fits in a single LLM context window. The RAG query system indexes all analysis outputs into a local vector store, letting you ask follow-up questions about any part of the conference and get answers with source references.

## Workflow

1. **Configure** — create an event config under `events/` (see [Configuration](./configuration.md)).
2. **Generate** — run the analysis to produce reports (see [Generation](./generation.md)).
3. **Review** — read the generated markdown reports and share them with your team.
4. **Query** — index the analysis data and ask follow-up questions (see [Generation](./generation.md#rag-query)).
