"""Plotly chart generation with SVG export."""

from collections import Counter
from pathlib import Path

from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file


def _ensure_images_dir(config: Config) -> Path:
    images_dir = config.data_dir / "reports" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def _extract_company(role_string: str) -> str:
    """Extract company name from role strings like 'Software Engineer, Google'."""
    if not role_string:
        return role_string
    parts = role_string.rsplit(", ", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return role_string.strip()


def _resolve_talk(
    title: str,
    talks_by_title: dict[str, dict],
    talks_by_prefix: dict[str, dict],
) -> dict | None:
    """Resolve a cluster talk title to its analysis via fuzzy matching."""
    if title in talks_by_title:
        return talks_by_title[title]
    prefix = title.split(" - ")[0].strip()
    if prefix in talks_by_prefix:
        return talks_by_prefix[prefix]
    for full_title, talk in talks_by_title.items():
        if full_title.startswith(title):
            return talk
    return None


def chart_topic_frequency(agenda: dict, images_dir: Path) -> Path | None:
    """Bar chart of top keywords by weighted mention count."""
    import plotly.graph_objects as go

    # Count keywords: top_keywords get weight 3, cluster keywords get 1
    word_counts: dict[str, int] = {}
    for kw in agenda.get("top_keywords", []):
        word_counts[kw] = word_counts.get(kw, 0) + 3
    for cluster in agenda.get("clusters", []):
        for kw in cluster.get("keywords", []):
            word_counts[kw] = word_counts.get(kw, 0) + 1

    if not word_counts:
        return None

    sorted_kw = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = [k for k, _ in sorted_kw]
    values = [v for _, v in sorted_kw]

    fig = go.Figure(
        data=[go.Bar(x=values, y=labels, orientation="h", marker_color="#4C78A8")],
        layout=go.Layout(
            title="Top Conference Topics",
            xaxis_title="Mentions (weighted)",
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
    """Bar chart of company talk counts (cleaned from role strings)."""
    import plotly.graph_objects as go

    raw = agenda.get("company_presence", {})
    if not raw:
        return None

    # Clean role strings to company names and aggregate
    clean_counts: dict[str, int] = {}
    for role_str, count in raw.items():
        company = _extract_company(role_str)
        clean_counts[company] = clean_counts.get(company, 0) + count

    sorted_companies = sorted(clean_counts.items(), key=lambda x: x[1], reverse=True)[:15]
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
    import plotly.graph_objects as go

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
    """Horizontal bar chart of cluster relevance scores with threshold lines."""
    import plotly.graph_objects as go

    if not ranking:
        return None

    labels = [c.get("name", "") for c in ranking]
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

    # Add threshold lines for deep-dive (0.70) and brief (0.40) cutoffs
    fig.add_vline(x=0.70, line_dash="dash", line_color="#54A24B", line_width=1.5,
                  annotation_text="Deep Dive", annotation_position="top right")
    fig.add_vline(x=0.40, line_dash="dot", line_color="#BAB0AC", line_width=1.5,
                  annotation_text="Brief", annotation_position="top right")

    out = images_dir / "cluster_relevance.svg"
    fig.write_image(str(out), format="svg")
    console.print(f"{tag('visualize')} Generated {out}")
    return out


def chart_maturity_strip(config: Config, images_dir: Path) -> Path | None:
    """Dot-strip chart: technologies on Y, maturity rings on X, color = evidence quality.

    Uses cluster-level maturity data (richer than global synthesis).
    """
    import plotly.graph_objects as go

    clusters_path = config.data_dir / "analysis_clusters.json"
    recordings_path = config.data_dir / "analysis_recordings.json"

    # Aggregate from cluster-level data
    evidence_rank = {
        "production_proven": 4, "benchmarked": 3, "case_study": 2, "anecdotal": 1,
    }
    by_tech: dict[str, dict] = {}

    if clusters_path.exists():
        clusters = load_json_file(clusters_path)
        for cluster in clusters:
            for item in cluster.get("maturity_landscape", []):
                tech = item.get("technology", "")
                if not tech:
                    continue
                rank = evidence_rank.get(item.get("evidence_quality", ""), 0)
                if tech not in by_tech or rank > evidence_rank.get(
                    by_tech[tech].get("evidence_quality", ""), 0,
                ):
                    by_tech[tech] = item

    # Fall back to global data if clusters have nothing
    if not by_tech and recordings_path.exists():
        recordings = load_json_file(recordings_path)
        for item in recordings.get("maturity_landscape", []):
            tech = item.get("technology", "")
            if tech:
                by_tech[tech] = item

    if not by_tech:
        return None

    items = sorted(by_tech.values(), key=lambda x: x.get("technology", ""))

    ring_order = {"assess": 0, "trial": 1, "adopt": 2, "hold": 3}
    evidence_colors = {
        "anecdotal": "#BAB0AC",
        "case_study": "#4C78A8",
        "benchmarked": "#F58518",
        "production_proven": "#54A24B",
    }

    techs = [it.get("technology", "") for it in items]
    x_vals = [ring_order.get(it.get("ring", "assess"), 0) for it in items]
    colors = [
        evidence_colors.get(it.get("evidence_quality", "anecdotal"), "#BAB0AC")
        for it in items
    ]
    hover = [
        f"{it.get('technology', '')}<br>Ring: {it.get('ring', 'assess')}<br>"
        f"Evidence: {it.get('evidence_quality', 'anecdotal')}<br>"
        f"Rationale: {it.get('rationale', '')[:80]}"
        for it in items
    ]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_vals,
                y=techs,
                mode="markers",
                marker=dict(size=14, color=colors, line=dict(width=1, color="#333")),
                text=hover,
                hoverinfo="text",
            )
        ],
        layout=go.Layout(
            title="Technology Maturity Landscape",
            xaxis=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["Assess", "Trial", "Adopt", "Hold"],
                title="Maturity Ring",
            ),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=250),
            height=max(500, len(techs) * 28),
            template="plotly_white",
        ),
    )

    out = images_dir / "maturity_strip.svg"
    fig.write_image(str(out), format="svg")
    console.print(f"{tag('visualize')} Generated {out}")
    return out


def chart_evidence_quality(
    agenda: dict,
    talks: list[dict],
    ranking: list[dict],
    images_dir: Path,
) -> Path | None:
    """Stacked bar chart: evidence quality breakdown per cluster."""
    import plotly.graph_objects as go

    if not agenda.get("clusters") or not talks:
        return None

    talks_by_title = {t["title"]: t for t in talks if t.get("title")}
    talks_by_prefix: dict[str, dict] = {}
    for title, talk in talks_by_title.items():
        prefix = title.split(" - ")[0].strip()
        if prefix not in talks_by_prefix:
            talks_by_prefix[prefix] = talk

    # Sort clusters by relevance score (highest first)
    rank_scores = {r["name"]: r.get("relevance_score", 0) for r in ranking}
    clusters_sorted = sorted(
        agenda.get("clusters", []),
        key=lambda c: rank_scores.get(c.get("name", ""), 0),
        reverse=True,
    )

    cluster_names = []
    evidence_counts: dict[str, list[int]] = {
        "production": [],
        "proof_of_concept": [],
        "theoretical": [],
        "vendor_demo": [],
    }

    for cluster in clusters_sorted:
        cname = cluster.get("name", "")
        counts: Counter = Counter()
        for title in cluster.get("talks", []):
            talk = _resolve_talk(title, talks_by_title, talks_by_prefix)
            if talk:
                eq = talk.get("evidence_quality", "unknown")
                if eq in evidence_counts:
                    counts[eq] += 1

        if sum(counts.values()) == 0:
            continue

        cluster_names.append(cname)
        for eq in evidence_counts:
            evidence_counts[eq].append(counts.get(eq, 0))

    if not cluster_names:
        return None

    colors = {
        "production": "#54A24B",
        "proof_of_concept": "#4C78A8",
        "theoretical": "#F58518",
        "vendor_demo": "#BAB0AC",
    }
    labels = {
        "production": "Production",
        "proof_of_concept": "Proof of Concept",
        "theoretical": "Theoretical",
        "vendor_demo": "Vendor Demo",
    }

    fig = go.Figure()
    for eq, counts_list in evidence_counts.items():
        if any(c > 0 for c in counts_list):
            fig.add_trace(go.Bar(
                y=cluster_names,
                x=counts_list,
                name=labels[eq],
                orientation="h",
                marker_color=colors[eq],
            ))

    fig.update_layout(
        barmode="stack",
        title="Evidence Quality by Cluster",
        xaxis_title="Number of Talks",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=250),
        height=max(400, len(cluster_names) * 35),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out = images_dir / "evidence_quality.svg"
    fig.write_image(str(out), format="svg")
    console.print(f"{tag('visualize')} Generated {out}")
    return out


def chart_stakeholder_breakdown(recordings: dict, images_dir: Path) -> Path | None:
    """Horizontal bar chart: companies by talk count, color = role type."""
    import plotly.graph_objects as go

    items = recordings.get("stakeholder_map", [])
    if not items:
        return None

    role_colors = {
        "vendor": "#E45756",
        "end_user": "#4C78A8",
        "maintainer": "#54A24B",
        "cloud_provider": "#F58518",
    }

    sorted_items = sorted(
        items, key=lambda x: x.get("talk_count", 0), reverse=True,
    )[:15]
    labels = [it.get("company", "") for it in sorted_items]
    values = [it.get("talk_count", 0) for it in sorted_items]
    colors = [
        role_colors.get(it.get("role", "vendor"), "#BAB0AC") for it in sorted_items
    ]

    fig = go.Figure(
        data=[go.Bar(x=values, y=labels, orientation="h", marker_color=colors)],
        layout=go.Layout(
            title="Stakeholder Breakdown (by Talk Count)",
            xaxis_title="Number of Talks",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200),
            height=max(400, len(labels) * 35),
            template="plotly_white",
        ),
    )

    out = images_dir / "stakeholder_breakdown.svg"
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
    agenda = load_json_file(agenda_path) if agenda_path.exists() else {}
    if agenda:
        for chart_fn in [
            chart_topic_frequency, chart_company_presence, chart_track_distribution,
        ]:
            result = chart_fn(agenda, images_dir)
            if result:
                generated.append(result)

    # Ranking chart
    ranking_path = data_dir / "analysis_ranking.json"
    ranking = load_json_file(ranking_path) if ranking_path.exists() else []
    if isinstance(ranking, list) and ranking:
        result = chart_cluster_relevance(ranking, images_dir)
        if result:
            generated.append(result)

    # Talks data (for evidence quality chart)
    talks_path = data_dir / "analysis_talks.json"
    talks = load_json_file(talks_path) if talks_path.exists() else []

    # Evidence quality chart (needs agenda + talks + ranking)
    if agenda and talks:
        result = chart_evidence_quality(agenda, talks, ranking, images_dir)
        if result:
            generated.append(result)

    # Maturity strip (uses cluster-level data)
    result = chart_maturity_strip(config, images_dir)
    if result:
        generated.append(result)

    # Recording-based charts (stakeholders)
    recordings_path = data_dir / "analysis_recordings.json"
    if recordings_path.exists():
        recordings = load_json_file(recordings_path)
        result = chart_stakeholder_breakdown(recordings, images_dir)
        if result:
            generated.append(result)

    if not generated:
        console.print(
            f"{tag('visualize')} No analysis data found, no charts generated."
        )

    return generated
