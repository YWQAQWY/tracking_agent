import asyncio

import httpx

from src.crawling.crawler import WebCrawler


def test_crawler_returns_none_for_http_error() -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(404, request=request)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await WebCrawler(client=client).fetch("https://example.com/missing")
        assert result is None

    asyncio.run(scenario())


def test_crawler_returns_none_for_timeout() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
            result = await WebCrawler(client=client).fetch("https://example.com/slow")
        assert result is None

    asyncio.run(scenario())


def test_crawler_fetches_html_and_follows_normal_contract() -> None:
    html = "<html><body><article>Useful text</article></body></html>"

    async def scenario() -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text=html,
                request=request,
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await WebCrawler(client=client).fetch("https://example.com/page")
        assert result is not None
        assert result.html == html
        assert result.status_code == 200

    asyncio.run(scenario())


def test_crawler_skips_non_html_and_oversized_pages() -> None:
    async def scenario() -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/pdf":
                return httpx.Response(
                    200,
                    headers={"content-type": "application/pdf"},
                    content=b"pdf",
                    request=request,
                )
            return httpx.Response(
                200,
                headers={"content-type": "text/html", "content-length": "999"},
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            crawler = WebCrawler(max_page_bytes=100, client=client)
            assert await crawler.fetch("https://example.com/pdf") is None
            assert await crawler.fetch("https://example.com/large") is None

    asyncio.run(scenario())

