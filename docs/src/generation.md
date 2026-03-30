# Generation

All pipeline commands require a config file via `-c`:

```sh
conf-briefing -c events/kubecon-eu-2026.toml <command>
```

Output files are written to the event's data directory (e.g. `events/kubecon-eu-2026/`). Reports go to `{data_dir}/reports/` and chart images to `{data_dir}/reports/images/`.

## Agenda Analysis

**Input:** Conference schedule data (titles, abstracts, speakers, companies, tracks).

Reads all talks and produces a conference landscape report:

- **Topic extraction** — main themes across all abstracts.
- **Keyword frequency and co-occurrence** — which terms appear most, which appear together.
- **Company and speaker presence** — who is presenting, how many talks per organization.
- **Talk clustering** — groups of related talks forming coherent themes.
- **Track and format breakdown** — distribution across tracks, keynotes vs. breakouts vs. lightning talks.

**Output:** Markdown report with conference statistics, identified clusters, keyword analysis, and company/speaker summary.

## Cluster Ranking

**Input:** Clusters from agenda analysis.

Takes the clusters and makes them actionable:

- **Cluster summaries** — what each cluster covers.
- **Relevance arguments** — why each cluster might be interesting (trends, practical applicability, novelty).
- **Ranking** — clusters ordered by relevance.
- **Talk shortlists** — top talks per cluster with brief descriptions.

The reader picks clusters they care about and gets a filtered shortlist instead of reviewing hundreds of abstracts.

## Recording Analysis

**Input:** YouTube talk transcripts.

After the conference, when recordings are published, the tool downloads transcripts and analyzes the actual talk content:

- **Key takeaways** — main points and conclusions per talk.
- **Q&A extraction** — audience questions and answers given.
- **Problem identification** — pain points and challenges discussed.
- **Emerging technology scan** — new tools, projects, and ideas mentioned.
- **Cross-talk synthesis** — comparing perspectives across talks on the same topic.

**Output:** Markdown report with per-talk summaries, aggregated Q&A highlights, emerging signals, and cross-talk comparisons.

## RAG Query

**Prerequisites:** [Ollama](https://ollama.com/) running locally.

```sh
ollama pull nomic-embed-text
ollama pull qwen3:14b
ollama serve
```

### Indexing

After running the pipeline, build a vector index over all analysis outputs:

```sh
conf-briefing -c events/kubecon-eu-2026.toml index
```

This loads all analysis data (transcripts, abstracts, talk summaries, takeaways, Q&A, signals, cluster rankings, and conference narratives), splits them into chunks with metadata, and indexes them into a local ChromaDB store at `{data_dir}/chroma/`. Each run does a full rebuild.

**Chunk types indexed:**

| Type                   | Source                 | Description                              |
|------------------------|------------------------|------------------------------------------|
| `transcript_segment`   | `transcripts/*.json`   | Sliding window ~500 tokens, 100 overlap  |
| `talk_abstract`        | `schedule_clean.json`  | One chunk per talk abstract              |
| `talk_summary`         | `analysis_talks.json`  | One chunk per talk summary               |
| `talk_takeaways`       | `analysis_talks.json`  | Key takeaways per talk                   |
| `talk_qa`              | `analysis_talks.json`  | Q&A highlights per talk                  |
| `talk_signals`         | `analysis_talks.json`  | Problems, tools, and signals combined    |
| `cluster_summary`      | `analysis_ranking.json`| One chunk per ranked cluster             |
| `conference_narrative` | `analysis_agenda.json` | Conference-level narratives and themes   |

### Asking Questions

Query the indexed data with natural language:

```sh
conf-briefing -c events/kubecon-eu-2026.toml ask "What are the main themes of the conference?"
```

The tool retrieves the most relevant chunks, sends them as context to the LLM, and returns an answer with source references.

**Options:**

| Flag                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `-k` / `--top-k`         | Number of chunks to retrieve (overrides config).  |
| `-t` / `--chunk-types`   | Comma-separated chunk types to filter by.         |
| `--track`                 | Filter by conference track.                       |
| `-v` / `--verbose`        | Show retrieved chunks before the answer.          |

**Examples:**

```sh
# Ask with verbose output
conf-briefing -c events/kubecon-eu-2026.toml ask -v "What talks cover observability?"

# Filter to talk summaries only
conf-briefing -c events/kubecon-eu-2026.toml ask -t talk_summary "What are the key takeaways about security?"

# Filter by track
conf-briefing -c events/kubecon-eu-2026.toml ask --track "Platform Engineering" "What tools were discussed?"
```
