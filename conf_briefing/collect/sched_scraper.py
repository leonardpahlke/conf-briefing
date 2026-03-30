"""Scrape public sched.com schedule pages without an API key.

This module is intentionally decoupled from the rest of the project.
It only depends on requests + beautifulsoup4 and returns plain dicts.
When Cloudflare blocks requests, it falls back to Patchright (optional)
to solve the JS challenge in a real browser.

Usage:
    from conf_briefing.collect.sched_scraper import scrape_schedule
    sessions = scrape_schedule("https://kccnceu2025.sched.com", cache_dir="talks/")
"""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from conf_briefing.console import console, progress_bar, tag

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Max concurrent detail page requests
MAX_WORKERS = 3

# Retry config for 429 / transient errors
MAX_RETRIES = 6
BACKOFF_SCHEDULE = [2, 4, 8, 10, 10, 10]  # seconds

# Minimum delay between starting new requests (across all workers)
REQUEST_SPACING = 0.4  # seconds

# Persistent browser profile for Cloudflare cookie reuse across runs
_BROWSER_PROFILE_DIR = Path.home() / ".cache" / "conf-briefing" / "browser-profile"


def _poll_for_cf_clearance(context, timeout: int = 120, interval: float = 1.0) -> str | None:
    """Poll browser cookies until cf_clearance appears or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for cookie in context.cookies():
            if cookie["name"] == "cf_clearance":
                return cookie["value"]
        time.sleep(interval)
    return None


def _get_session(base_url: str) -> requests.Session:
    """Create a requests session, using Patchright to solve CF challenges if needed.

    Tries a plain request first. If Cloudflare blocks it (403 with
    cf-mitigated header), launches a real Chrome browser via Patchright
    so the user can solve the challenge. Polls for the cf_clearance cookie,
    then transfers cookies + user-agent to a requests session.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    resp = session.get(base_url, timeout=30)
    if resp.status_code != 403:
        return session

    # Check if this is a Cloudflare challenge
    if "cf-mitigated" not in resp.headers:
        resp.raise_for_status()

    console.print(
        f"{tag('sched')} [yellow]Cloudflare JS challenge detected, launching browser...[/yellow]"
    )
    try:
        from patchright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "sched.com is behind Cloudflare protection and requires a browser.\n"
            "Install Patchright:  uv sync --extra scrape && patchright install chrome\n"
            "Or provide a local schedule file via [conference] schedule = 'path/to/file.json'"
        ) from e

    _BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    console.print(
        f"{tag('sched')} [yellow]Solve the Cloudflare challenge in the browser window...[/yellow]"
    )

    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(_BROWSER_PROFILE_DIR),
            channel="chrome",
            headless=False,
            no_viewport=True,
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(base_url, timeout=120_000, wait_until="domcontentloaded")

        cf_clearance = _poll_for_cf_clearance(context, timeout=120)
        if not cf_clearance:
            context.close()
            raise RuntimeError(
                "Cloudflare challenge was not solved within 120 seconds.\n"
                "Or provide a local schedule file via [conference] schedule = 'path/to/file.json'"
            )

        cookies = context.cookies()
        user_agent = page.evaluate("navigator.userAgent")
        context.close()

    # Apply cookies and the browser's real user-agent to the requests session
    session.headers["User-Agent"] = user_agent
    for cookie in cookies:
        session.cookies.set(cookie["name"], cookie["value"], domain=cookie["domain"])

    # Verify cookies work
    resp = session.get(base_url, timeout=30)
    if resp.status_code == 403:
        raise RuntimeError(
            "Could not bypass Cloudflare protection even with browser.\n"
            "Try running: patchright install chrome\n"
            "Or provide a local schedule file via [conference] schedule = 'path/to/file.json'"
        )

    console.print(f"{tag('sched')} Cloudflare challenge solved, cookies acquired")
    return session


def scrape_schedule(
    sched_url: str,
    *,
    fetch_details: bool = True,
    cache_dir: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> list[dict]:
    """Scrape all sessions from a public sched.com event page.

    Args:
        sched_url: Base URL like "https://kccnceu2025.sched.com"
        fetch_details: If True, fetch each session's detail page for
                       abstracts, speakers, and track info.
        cache_dir: If set, cache each session detail as JSON in this dir.
                   Cached sessions are skipped on subsequent runs.
        max_workers: Number of concurrent requests for detail pages.

    Returns:
        List of session dicts with keys:
        title, abstract, speakers, track, format, time, tags, venue, sched_url
    """
    base = sched_url.rstrip("/")
    http_session = _get_session(base)
    sessions = _scrape_listing(base, http_session)

    if fetch_details and sessions:
        sessions = _fetch_all_details(
            sessions, http_session=http_session, cache_dir=cache_dir, max_workers=max_workers
        )

    return sessions


def _sched_id_from_url(url: str) -> str:
    """Extract the sched event ID from a detail URL.

    e.g. ".../event/1tfF2/some-slug" → "1tfF2"
    """
    match = re.search(r"/event/([^/]+)", url)
    return match.group(1) if match else ""


def _fetch_with_retry(url: str, session: requests.Session) -> requests.Response:
    """Fetch a URL with exponential backoff on 429, 5xx, and connection errors."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            delay = BACKOFF_SCHEDULE[min(attempt, len(BACKOFF_SCHEDULE) - 1)]
            time.sleep(delay)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            delay = BACKOFF_SCHEDULE[min(attempt, len(BACKOFF_SCHEDULE) - 1)]
            time.sleep(delay)
            continue

        resp.raise_for_status()
        return resp

    # Final attempt — let it raise
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp


def _fetch_all_details(
    sessions: list[dict],
    *,
    http_session: requests.Session,
    cache_dir: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> list[dict]:
    """Fetch detail pages for all sessions, with caching and concurrency."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    total = len(sessions)
    to_fetch: list[tuple[int, dict]] = []
    cached_count = 0

    # Check cache and separate cached vs. to-fetch
    for idx, sess in enumerate(sessions):
        detail_url = sess.get("_detail_url", "")
        if not detail_url:
            continue

        sched_id = _sched_id_from_url(detail_url)
        if cache_dir and sched_id:
            cache_file = cache_dir / f"{sched_id}.json"
            if cache_file.exists():
                cached = json.loads(cache_file.read_text())
                sess.update(cached)
                sess.pop("_detail_url", None)
                cached_count += 1
                continue

        to_fetch.append((idx, sess))

    if cached_count:
        console.print(
            f"{tag('sched')} {cached_count}/{total} sessions cached, {len(to_fetch)} to fetch"
        )

    if not to_fetch:
        console.print(f"{tag('sched')} All {total} sessions cached.")
        return sessions

    console.print(
        f"{tag('sched')} Fetching details for {len(to_fetch)} sessions "
        f"({max_workers} concurrent)..."
    )

    # Thread-local sessions (requests.Session is not thread-safe)
    _local = threading.local()
    shared_headers = dict(http_session.headers)
    shared_cookies = dict(http_session.cookies)

    def _get_thread_session() -> requests.Session:
        if not hasattr(_local, "session"):
            s = requests.Session()
            s.headers.update(shared_headers)
            s.cookies.update(shared_cookies)
            _local.session = s
        return _local.session

    def _fetch_one(idx_sess: tuple[int, dict]) -> tuple[int, dict | None, str]:
        idx, sess = idx_sess
        detail_url = sess.get("_detail_url", "")
        try:
            detail = _scrape_detail(detail_url, _get_thread_session())
            return idx, detail, sess["title"]
        except Exception as e:
            return idx, None, f"failed: {e}"

    def _handle_result(future, pb, task):
        idx, detail, title_or_err = future.result()
        if detail is not None:
            sessions[idx].update(detail)
            sessions[idx].pop("_detail_url", None)
            if cache_dir:
                detail_url = sessions[idx].get("sched_url", "")
                sched_id = _sched_id_from_url(detail_url)
                if sched_id:
                    cache_file = cache_dir / f"{sched_id}.json"
                    cache_file.write_text(json.dumps(sessions[idx], indent=2, ensure_ascii=False))
        pb.update(task, advance=1, description=f"{tag('sched')} {title_or_err[:50]}")

    with progress_bar() as pb:
        task = pb.add_task(f"{tag('sched')} Fetching sessions", total=len(to_fetch))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pending: dict = {}
            for item in to_fetch:
                pending[pool.submit(_fetch_one, item)] = item
                time.sleep(REQUEST_SPACING)
                # Drain completed futures during submission
                done = [f for f in pending if f.done()]
                for f in done:
                    del pending[f]
                    _handle_result(f, pb, task)

            # Process remaining futures
            for future in as_completed(pending):
                _handle_result(future, pb, task)

    return sessions


def _scrape_listing(base_url: str, session: requests.Session) -> list[dict]:
    """Scrape the main schedule listing page for session links and titles."""
    resp = session.get(base_url, timeout=30)
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


def _scrape_detail(url: str, session: requests.Session) -> dict:
    """Scrape a single session detail page for abstract, speakers, track, time."""
    resp = _fetch_with_retry(url, session)
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
