# Generation

All pipeline commands require a config file via `-c`:

```sh
conf-briefing -c events/kubecon-eu-2026.toml <command>
```

Output files are written to the event's data directory (e.g. `events/kubecon-eu-2026/`). Reports go to `{data_dir}/reports/` and chart images to `{data_dir}/reports/images/`.

## Setup

Sync dependencies and pull all required Ollama models (LLM, embeddings, and VLM if configured):

```sh
just pull-models kubecon-eu-2026 cpu      # CPU-only (faster-whisper, no diarization)
just pull-models kubecon-eu-2026 amd      # AMD/ROCm (WhisperX + diarization)
just pull-models kubecon-eu-2026 nvidia   # NVIDIA/CUDA (WhisperX + diarization)
```

For diarization (speaker identification), set a Hugging Face token for pyannote model access:

```sh
export HF_TOKEN="hf_..."
```

Verify extract dependencies are available:

```sh
just extract-check kubecon-eu-2026
```

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

After the conference, when recordings are published, the `extract` command processes them in three steps:

```sh
just extract kubecon-eu-2026
```

1. **Transcription** — audio is transcribed using the best available backend (WhisperX > whisper.cpp > faster-whisper, auto-detected). An `initial_prompt` provides domain terminology to improve accuracy. When WhisperX is installed with `HF_TOKEN` set, transcripts include speaker diarization (`[Speaker 1]`, `[Speaker 2]`).
2. **Slide extraction** — scene detection finds slide transitions, perceptual hashing removes duplicates, and Tesseract OCR extracts text from each slide. URLs and GitHub links are extracted from slide text.
3. **VLM descriptions** (optional) — when `vlm_model` is configured, each slide image is sent to a vision-language model to describe diagrams, architecture charts, visual content, and any visible URLs or QR codes.

The extract step is followed by cleaning, matching, and LLM analysis:

- **Slide-transcript alignment** — slides are interleaved with transcript segments by timestamp, giving the LLM a chronological view of what was shown and said together.
- **Entity canonicalization** — CNCF/cloud-native aliases are normalized (e.g. "k8s" becomes "Kubernetes", "otel" becomes "OpenTelemetry") for consistent entity tracking across talks.
- **Structured output** — LLM responses use Pydantic schemas via Ollama's structured output, guaranteeing valid JSON without parse retries.
- **Grounding** — prompts instruct the LLM to only include information explicitly stated in the transcript, preferring omission over fabrication.

Per-talk analysis extracts:

- **Key takeaways** — main points and conclusions per talk.
- **Q&A extraction** — audience questions and answers given.
- **Problem identification** — pain points and challenges discussed.
- **Emerging technology scan** — new tools, projects, and ideas mentioned.
- **Maturity assessment** — per-talk technology maturity ratings (assess/trial/adopt/hold) with evidence.
- **Speaker perspective** — whether each speaker is a practitioner, vendor, maintainer, or academic.
- **Technology stance** — how speakers feel about each technology (enthusiastic/cautious/critical/neutral) with evidence.
- **Technology relationships** — explicit relationships between technologies (replaces, competes with, builds on, integrates with, extends).
- **Caveats and concerns** — limitations and warnings mentioned.
- **References** — URLs, GitHub repos, and citations from slides and transcript.

The synthesis step aggregates across all talks:

- **Cross-talk synthesis** — comparing perspectives across talks on the same topic.
- **Tensions** — contradictions and unresolved debates between talks.
- **Maturity landscape** — aggregated technology maturity ratings across all talks.
- **Stakeholder map** — company roles, agendas, and notable claims.
- **Technology relationships** — aggregated relationship graph with supporting talk lists.
- **Quiet signals** — single mentions worth watching.
- **Absent topics** — expected topics that nobody discussed.
- **Recommended actions** — concrete next steps categorized by urgency.

**Output:** Markdown reports including an 8-layer intelligence briefing, per-talk summaries, aggregated Q&A highlights, emerging signals, and cross-talk comparisons.

## Intelligence Briefing

The primary output is a unified intelligence briefing (`briefing_report.md`) structured around 8 information layers:

1. **Landscape** — territory, players, tracks, volume (from agenda analysis)
2. **Narrative** — big themes, dominant conversations (from recording synthesis)
3. **Movement** — what changed since last time (*deferred — needs multi-conference data*)
4. **Maturity** — hype vs production, aspirational vs concrete (from maturity assessments)
5. **Tension** — contradictions, unresolved debates (from cross-talk comparison)
6. **Stakeholders** — vendor vs practitioner, company agendas (from stakeholder map)
7. **Signals** — quiet mentions, unexpected topics, absent topics (from signal extraction)
8. **Actions** — evaluate, watch, follow up, reconsider (from recommended actions)

Every section renders meaningfully with agenda-only, recordings-only, or both data sources. The existing detail reports (agenda, ranking, recording) remain as supplementary views.

## RAG Query

**Prerequisites:** [Ollama](https://ollama.com/) running locally.

```sh
just pull-models kubecon-eu-2026
ollama serve
```

### Indexing

After running the pipeline, build a vector index over all analysis outputs:

```sh
conf-briefing -c events/kubecon-eu-2026.toml index
```

This loads all analysis data (transcripts, abstracts, talk summaries, takeaways, Q&A, signals, cluster rankings, and conference narratives), splits them into chunks with metadata, and indexes them into a local ChromaDB store at `{data_dir}/chroma/`. Each run does a full rebuild.

**Chunk types indexed:**

| Type                   | Source                      | Description                              |
|------------------------|-----------------------------|------------------------------------------|
| `transcript_segment`   | `transcripts/*.json`        | Sliding window ~500 tokens, 100 overlap  |
| `talk_abstract`        | `schedule_clean.json`       | One chunk per talk abstract              |
| `talk_summary`         | `analysis_talks.json`       | One chunk per talk summary               |
| `talk_takeaways`       | `analysis_talks.json`       | Key takeaways per talk                   |
| `talk_qa`              | `analysis_talks.json`       | Q&A highlights per talk                  |
| `talk_signals`         | `analysis_talks.json`       | Problems, tools, and signals combined    |
| `cluster_summary`      | `analysis_ranking.json`     | One chunk per ranked cluster             |
| `conference_narrative` | `analysis_agenda.json`      | Conference-level narratives and themes   |
| `maturity_assessment`  | `analysis_recordings.json`  | One chunk per technology maturity rating  |
| `tension`              | `analysis_recordings.json`  | One chunk per debate/contradiction        |
| `recommended_action`   | `analysis_recordings.json`  | One chunk per recommended action          |

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
