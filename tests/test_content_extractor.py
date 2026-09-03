from src.extraction.content_extractor import ContentExtractor


ARTICLE_HTML = """
<html>
  <head><title>Test</title></head>
  <body>
    <nav>Menu</nav>
    <article>
      <h1>Retrieval Augmented Generation</h1>
      <p>RAG combines retrieval and generation to ground answers in external
      evidence instead of relying only on model memory.</p>
    </article>
    <footer>Copyright</footer>
  </body>
</html>
"""


def test_content_extractor_finds_article_and_removes_page_chrome() -> None:
    document = ContentExtractor(min_content_length=20).extract(
        ARTICLE_HTML, "https://example.com/article"
    )

    assert document is not None
    assert document.title == "Test"
    assert "Retrieval Augmented Generation" in document.text
    assert "RAG combines retrieval and generation" in document.text
    assert "Menu" not in document.text
    assert "Copyright" not in document.text


def test_beautifulsoup_fallback_removes_noise(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.extraction.content_extractor.trafilatura.extract", lambda *a, **k: None
    )
    document = ContentExtractor(min_content_length=20).extract(
        ARTICLE_HTML, "https://example.com/article"
    )

    assert document is not None
    assert "Menu" not in document.text
    assert "Copyright" not in document.text


def test_content_extractor_rejects_short_content() -> None:
    html = "<html><body><article><p>Too short</p></article></body></html>"
    assert (
        ContentExtractor(min_content_length=200).extract(
            html, "https://example.com/short"
        )
        is None
    )

