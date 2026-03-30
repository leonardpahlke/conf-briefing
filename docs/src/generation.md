# Generation

> **Note:** The technical implementation is not decided yet. This chapter describes what each analysis produces, not how it is run.

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

**Input:** Clusters from agenda analysis + configured focus topics.

Takes the clusters and makes them actionable:

- **Cluster summaries** — what each cluster covers.
- **Relevance arguments** — why each cluster might be interesting (trends, practical applicability, novelty).
- **Ranking** — clusters ordered by relevance to the configured focus topics.
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
