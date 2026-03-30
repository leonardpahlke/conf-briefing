"""Scrape public sched.com schedule pages without an API key.

This module is intentionally decoupled from the rest of the project.
It only depends on requests + beautifulsoup4 and returns plain dicts.

Usage:
    from conf_briefing.collect.sched_scraper import scrape_schedule
    sessions = scrape_schedule("https://kccnceu2025.sched.com")
"""

import re
import time

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Delay between detail page requests (seconds)
REQUEST_DELAY = 1.0


def scrape_schedule(
    sched_url: str,
    *,
    fetch_details: bool = True,
    delay: float = REQUEST_DELAY,
) -> list[dict]:
    """Scrape all sessions from a public sched.com event page.

    Args:
        sched_url: Base URL like "https://kccnceu2025.sched.com"
        fetch_details: If True, fetch each session's detail page for
                       abstracts, speakers, and track info.
        delay: Seconds to wait between detail page requests.

    Returns:
        List of session dicts with keys:
        title, abstract, speakers, track, format, time, tags, venue, sched_url
    """
    base = sched_url.rstrip("/")
    sessions = _scrape_listing(base)

    if fetch_details and sessions:
        total = len(sessions)
        print(f"[sched] Fetching details for {total} sessions...")
        for i, session in enumerate(sessions, 1):
            detail_url = session.pop("_detail_url", "")
            if not detail_url:
                continue
            try:
                detail = _scrape_detail(detail_url)
                session.update(detail)
                print(f"[sched]   ({i}/{total}) {session['title'][:60]}")
            except Exception as e:
                print(f"[sched]   ({i}/{total}) failed: {e}")
            if i < total:
                time.sleep(delay)

    return sessions


def _scrape_listing(base_url: str) -> list[dict]:
    """Scrape the main schedule listing page for session links and titles."""
    resp = requests.get(base_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    sessions = []
    seen_urls = set()

    for span in soup.select("span.event"):
        link = span.select_one("a.name")
        if not link:
            continue

        href = link.get("href", "")
        if not href.startswith("event/"):
            continue

        detail_url = f"{base_url}/{href}"
        if detail_url in seen_urls:
            continue
        seen_urls.add(detail_url)

        # Extract title (text without venue span)
        venue_span = link.select_one("span.vs")
        venue = venue_span.get_text(strip=True) if venue_span else ""
        title = link.get_text(strip=True)
        if venue and title.endswith(venue):
            title = title[: -len(venue)].strip()

        sessions.append(
            {
                "title": title,
                "abstract": "",
                "speakers": [],
                "track": "",
                "format": "",
                "time": "",
                "tags": [],
                "venue": venue,
                "sched_url": detail_url,
                "_detail_url": detail_url,
            }
        )

    return sessions


def _scrape_detail(url: str) -> dict:
    """Scrape a single session detail page for abstract, speakers, track, time."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    result: dict = {}

    # Abstract from tip-description
    desc_div = soup.select_one("div.tip-description")
    if desc_div:
        # Remove the <strong> label if present
        strong = desc_div.select_one("strong")
        if strong:
            strong.decompose()
        result["abstract"] = desc_div.get_text(separator="\n", strip=True)

    # Speakers from tip-roles
    speakers = []
    roles_div = soup.select_one("div.tip-roles")
    if roles_div:
        for person in roles_div.select("div.sched-person-session"):
            name_tag = person.select_one("h2 a")
            company_tag = person.select_one("div.sched-event-details-role-company")
            if name_tag:
                speakers.append(
                    {
                        "name": name_tag.get_text(strip=True),
                        "company": company_tag.get_text(strip=True) if company_tag else "",
                    }
                )
    if speakers:
        result["speakers"] = speakers

    # Track from sched-event-type
    type_div = soup.select_one("div.sched-event-type")
    if type_div:
        type_link = type_div.select_one("a")
        if type_link:
            result["track"] = type_link.get_text(strip=True)

    # Time from sched-event-details-timeandplace
    time_div = soup.select_one("div.sched-event-details-timeandplace")
    if time_div:
        time_text = time_div.get_text(separator=" ", strip=True)
        # Clean up: take just the date/time line
        time_text = re.split(r"\s{2,}", time_text)[0]
        result["time"] = time_text

    # Custom fields as tags
    tags = []
    for li in soup.select("ul.tip-custom-fields li"):
        tags.append(li.get_text(strip=True))
    if tags:
        result["tags"] = tags

    return result
