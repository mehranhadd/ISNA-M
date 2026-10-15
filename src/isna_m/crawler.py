"""Scrape article metadata and text from the ISNA news website.

The website's HTML structure may change. Network access is only performed when
``crawl()`` or the command-line entry point is explicitly invoked.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, SoupStrainer

from .paths import DEFAULT_RAW_DATASET, ensure_project_directories

BASE_URL = "https://www.isna.ir/"
DEFAULT_PAGE_COUNT = 18
REQUEST_TIMEOUT_SECONDS = 30
ARTICLE_COLUMNS = [
    "Content", "Title", "Category", "Source", "Journalist", "Secretary",
    "Tags", "Week_Day", "Date", "Time", "URL",
]
MAIN_ONLY = SoupStrainer("main")
HEADER_ONLY = SoupStrainer("header")


def get_html(
    url: str,
    only: SoupStrainer | None = None,
    *,
    session: requests.Session | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> tuple[str, BeautifulSoup]:
    """Fetch a page and return its title and parsed HTML."""
    client = session or requests
    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    html_text = response.text
    title_match = re.search(r"<title\b[^>]*>(.*?)</title\s*>", html_text, re.I | re.S)
    title = BeautifulSoup(title_match.group(1), "html.parser").get_text(" ", strip=True) if title_match else ""
    soup = BeautifulSoup(html_text, "html.parser", parse_only=only)
    return title, soup


def get_archive_link(*, session: requests.Session | None = None) -> str:
    """Find the ISNA archive link from the page header."""
    _, home_soup = get_html(BASE_URL, only=HEADER_ONLY, session=session)
    for li in home_soup.find_all("li"):
        anchor = li.find("a", href=True)
        if not anchor:
            continue
        href = anchor["href"]
        data_id = str(li.get("data-id", ""))
        if data_id.startswith("10") or re.search(r"archive", href, re.I):
            if re.search(r"archive", href, re.I):
                return urljoin(BASE_URL, href)
    raise RuntimeError("Could not find the ISNA archive link; the website HTML may have changed.")


def _story_links_from_page(soup: BeautifulSoup) -> list[str]:
    """Extract article URLs from an archive page, ignoring photo reports."""
    links: list[str] = []
    for container in soup.find_all("div", class_="items"):
        for figure in container.find_all("figure"):
            anchor = figure.find("a", href=True)
            if not anchor:
                continue
            href = anchor["href"]
            if re.search(r"^/photo(?:/|$)", href):
                continue
            links.append(urljoin(BASE_URL, href))
        if links:
            break
    return links


def _next_archive_page(soup: BeautifulSoup, current_url: str) -> str | None:
    """Return the next archive page URL, if it is present."""
    footer = soup.find("footer")
    if footer:
        for anchor in footer.find_all("a", href=True):
            if "بعدی" in anchor.get_text(" ", strip=True):
                return urljoin(current_url, anchor["href"])
    return None


def get_news_links(
    archive: str,
    page_num: int = 10,
    *,
    session: requests.Session | None = None,
    delay_seconds: float = 2.0,
) -> list[str]:
    """Collect article URLs from up to ``page_num`` archive pages."""
    if page_num < 1:
        raise ValueError("page_num must be at least 1")
    news_links: list[str] = []
    current_url: str | None = archive
    for page_index in range(page_num):
        if not current_url:
            break
        _, soup = get_html(current_url, only=MAIN_ONLY, session=session)
        news_links.extend(_story_links_from_page(soup))
        next_url = _next_archive_page(soup, current_url)
        current_url = next_url
        if page_index < page_num - 1 and current_url and delay_seconds > 0:
            time.sleep(delay_seconds)
    # Preserve order while preventing duplicate URLs across archive pages.
    return list(dict.fromkeys(news_links))


def extract_li_information(soup: BeautifulSoup) -> tuple[str | None, ...]:
    """Extract weekday, date, time, category, source, journalist and secretary."""
    weekday = date = publish_time = category = source = journalist = secretary = None
    for li in soup.find_all("li"):
        icon = li.find("i")
        icon_classes = " ".join(icon.get("class", [])) if icon else ""
        spans = li.find_all("span")
        if "calendar" in icon_classes:
            for span in spans:
                classes = span.get("class", [])
                if "title-meta" in classes:
                    weekday = span.get_text(" ", strip=True)
                elif "text-meta" in classes:
                    parts = span.get_text(" ", strip=True).split("/", 1)
                    if len(parts) == 2:
                        date, publish_time = (part.strip() for part in parts)
            continue
        if "folder" in icon_classes:
            category_span = next((s for s in spans if "text-meta" in s.get("class", [])), None)
            if category_span:
                category = category_span.get_text(" ", strip=True)
            continue
        if re.search(r"microphone$", icon_classes):
            title_span = next((s for s in spans if "title-meta" in s.get("class", [])), None)
            value_span = next((s for s in spans if "text-meta" in s.get("class", [])), None)
            if title_span and title_span.get_text(" ", strip=True) != "منبع :":
                source = "Journalist"
            elif value_span:
                source = value_span.get_text(" ", strip=True)
            continue
        if re.search(r"edit$", icon_classes):
            strong = li.find("strong")
            journalist = strong.get_text(" ", strip=True) if strong else journalist
            continue
        if re.search(r"check-square$", icon_classes):
            strong = li.find("strong")
            secretary = strong.get_text(" ", strip=True) if strong else secretary
    return weekday, date, publish_time, category, source, journalist, secretary


def extract_news_content(soup: BeautifulSoup) -> str:
    """Join paragraph text found on an article page."""
    return " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p")).strip()


def extract_news_tags(soup: BeautifulSoup) -> list[str]:
    """Extract article tags from the page footer."""
    footer = soup.find("footer")
    tag_list = footer.find("ul") if footer else None
    if not tag_list:
        return []
    return [item.get_text(" ", strip=True) for item in tag_list.find_all("li") if item.get_text(" ", strip=True)]


def extract_article(url: str, *, session: requests.Session | None = None) -> dict[str, object]:
    """Fetch a single article and return it in the dataset's column schema."""
    title, soup = get_html(url, only=MAIN_ONLY, session=session)
    weekday, date, publish_time, category, source, journalist, secretary = extract_li_information(soup)
    return {
        "Content": extract_news_content(soup),
        "Title": title,
        "Category": category,
        "Source": source,
        "Journalist": journalist,
        "Secretary": secretary,
        "Tags": extract_news_tags(soup),
        "Week_Day": weekday,
        "Date": date,
        "Time": publish_time,
        "URL": url,
    }


def extract_data(
    news_links: Iterable[str],
    *,
    session: requests.Session | None = None,
    delay_seconds: float = 2.0,
) -> Iterable[dict[str, object]]:
    """Yield article records in URL order, with a delay between requests."""
    for url in news_links:
        yield extract_article(url, session=session)
        if delay_seconds > 0:
            time.sleep(delay_seconds)


def crawl(
    output_path: str | Path = DEFAULT_RAW_DATASET,
    *,
    page_num: int = DEFAULT_PAGE_COUNT,
    delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """Crawl the archive and save the resulting raw CSV dataset."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        archive_url = get_archive_link(session=session)
        links = get_news_links(archive_url, page_num, session=session, delay_seconds=delay_seconds)
        records = list(extract_data(links, session=session, delay_seconds=delay_seconds))
    dataframe = pd.DataFrame.from_records(records, columns=ARTICLE_COLUMNS)
    dataframe.to_csv(output_path, index=False)
    print(f"Saved {len(dataframe)} articles to {output_path}")
    return dataframe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect article metadata and content from ISNA.")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGE_COUNT, help="Maximum archive pages to crawl (default: 18).")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait between requests (default: 2).")
    parser.add_argument("--output", type=Path, default=DEFAULT_RAW_DATASET, help="Destination CSV path.")
    args = parser.parse_args(argv)
    ensure_project_directories()
    crawl(args.output, page_num=args.pages, delay_seconds=args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
