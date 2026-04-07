"""Shared provider resolution for URL-based service dispatch."""

from urllib.parse import urlparse


def resolve_provider(
    url: str, patterns: list[tuple[str, str]]
) -> tuple[str, str] | None:
    """Match a URL hostname to a provider from a pattern list.

    Returns (provider_name, module_path) or None.
    """
    hostname = urlparse(url).hostname or ""
    for domain, module in patterns:
        if hostname == domain or hostname.endswith("." + domain):
            return domain, module
    return None
