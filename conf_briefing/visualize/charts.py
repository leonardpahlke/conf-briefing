"""Plotly chart generation with SVG export."""

import json
from pathlib import Path

import plotly.graph_objects as go

from conf_briefing.config import Config
from conf_briefing.console import console, tag


def _ensure_images_dir(config: Config) -> Path:
    images_dir = config.data_dir / "reports" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def chart_topic_frequency(agenda: dict, images_dir: Path) -> Path | None:
    """Bar chart of top keywords/topics."""
    keywords = agenda.get("top_keywords", [])
    if not keywords:
        return None

    # Use keyword position as proxy for frequency (most important first)
    labels = keywords[:15]
    values = list(range(len(labels), 0, -1))

    fig = go.Figure(
        data=[go.Bar(x=values, y=labels, orientation="h", marker_color="#4C78A8")],
        layout=go.Layout(
            title="Top Conference Topics",
            xaxis_title="Relative Prominence",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200),
            height=500,
            template="plotly_white",
        ),
    )

    out = images_dir / "topic_frequency.svg"
    fig.write_image(str(out), format="svg")
    console.print(f"{tag('visualize')} Generated {out}")
    return out


def chart_company_presence(agenda: dict, images_dir: Path) -> Path | None:
    """Bar chart of company talk counts."""
    companies = agenda.get("company_presence", {})
    if not companies:
        return None

    # Sort by count, take top 15
    sorted_companies = sorted(companies.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [c[0] for c in sorted_companies]
    values = [c[1] for c in sorted_companies]

    fig = go.Figure(
        data=[go.Bar(x=values, y=labels, orientation="h", marker_color="#E45756")],
        layout=go.Layout(
            title="Company Presence (by Talk Count)",
            xaxis_title="Number of Talks",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200),
            height=500,
            template="plotly_white",
        ),
    )

    out = images_dir / "company_presence.svg"
    fig.write_image(str(out), format="svg")
    console.print(f"{tag('visualize')} Generated {out}")
    return out


def chart_track_distribution(agenda: dict, images_dir: Path) -> Path | None:
    """Pie chart of track distribution."""
    tracks = agenda.get("track_distribution", {})
    if not tracks:
        return None

    fig = go.Figure(
        data=[go.Pie(labels=list(tracks.keys()), values=list(tracks.values()))],
        layout=go.Layout(
            title="Track Distribution",
            template="plotly_white",
            height=500,
        ),
    )

    out = images_dir / "track_distribution.svg"
    fig.write_image(str(out), format="svg")
    console.print(f"{tag('visualize')} Generated {out}")
    return out


def chart_cluster_relevance(ranking: list[dict], images_dir: Path) -> Path | None:
    """Horizontal bar chart of cluster relevance scores."""
    if not ranking:
        return None

    labels = [c["name"] for c in ranking]
    scores = [c.get("relevance_score", 0) for c in ranking]

    fig = go.Figure(
        data=[go.Bar(x=scores, y=labels, orientation="h", marker_color="#72B7B2")],
        layout=go.Layout(
            title="Cluster Relevance",
            xaxis_title="Relevance Score",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200),
            height=max(400, len(labels) * 35),
            template="plotly_white",
        ),
    )

    out = images_dir / "cluster_relevance.svg"
    fig.write_image(str(out), format="svg")
    console.print(f"{tag('visualize')} Generated {out}")
    return out


def generate_charts(config: Config) -> list[Path]:
    """Generate all charts from analysis data."""
    data_dir = config.data_dir
    images_dir = _ensure_images_dir(config)
    generated = []

    # Agenda charts
    agenda_path = data_dir / "analysis_agenda.json"
    if agenda_path.exists():
        agenda = json.loads(agenda_path.read_text())
        for chart_fn in [chart_topic_frequency, chart_company_presence, chart_track_distribution]:
            result = chart_fn(agenda, images_dir)
            if result:
                generated.append(result)

    # Ranking chart
    ranking_path = data_dir / "analysis_ranking.json"
    if ranking_path.exists():
        ranking = json.loads(ranking_path.read_text())
        if isinstance(ranking, list):
            result = chart_cluster_relevance(ranking, images_dir)
            if result:
                generated.append(result)

    if not generated:
        console.print(f"{tag('visualize')} No analysis data found, no charts generated.")

    return generated
