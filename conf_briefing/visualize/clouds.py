"""Word cloud generation."""

from pathlib import Path

from conf_briefing.config import Config
from conf_briefing.console import console, tag
from conf_briefing.io import load_json_file


def generate_wordcloud(config: Config) -> Path | None:
    """Generate a keyword word cloud from agenda analysis."""
    data_dir = config.data_dir
    agenda_path = data_dir / "analysis_agenda.json"

    if not agenda_path.exists():
        console.print(f"{tag('visualize')} No agenda analysis found, skipping word cloud.")
        return None

    agenda = load_json_file(agenda_path)

    # Collect all keywords from clusters and top_keywords
    words: dict[str, int] = {}
    for kw in agenda.get("top_keywords", []):
        words[kw] = words.get(kw, 0) + 3

    for cluster in agenda.get("clusters", []):
        for kw in cluster.get("keywords", []):
            words[kw] = words.get(kw, 0) + 1

    if not words:
        console.print(f"{tag('visualize')} No keywords found, skipping word cloud.")
        return None

    from wordcloud import WordCloud

    images_dir = config.data_dir / "reports" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=80,
    ).generate_from_frequencies(words)

    out = images_dir / "keyword_cloud.png"
    wc.to_file(str(out))
    console.print(f"{tag('visualize')} Generated {out}")
    return out
