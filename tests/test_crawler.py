from bs4 import BeautifulSoup

from isna_m import crawler


def test_archive_parser_skips_photo_reports():
    soup = BeautifulSoup(
        '''<main><div class="items">
        <figure><a href="/news/123">story</a></figure>
        <figure><a href="/photo/456">photo report</a></figure>
        <figure><a href="https://example.org/news/789">external</a></figure>
        </div></main>''',
        "html.parser",
    )

    assert crawler._story_links_from_page(soup) == [
        "https://www.isna.ir/news/123",
        "https://example.org/news/789",
    ]


def test_next_archive_page_uses_relative_url():
    soup = BeautifulSoup('<footer><a href="/archive?page=2">صفحه بعدی</a></footer>', "html.parser")
    assert crawler._next_archive_page(soup, "https://www.isna.ir/archive?page=1") == (
        "https://www.isna.ir/archive?page=2"
    )


def test_empty_article_metadata_does_not_crash():
    soup = BeautifulSoup("<main><p>body text</p></main>", "html.parser")
    assert crawler.extract_li_information(soup) == (None, None, None, None, None, None, None)
    assert crawler.extract_news_content(soup) == "body text"
    assert crawler.extract_news_tags(soup) == []
