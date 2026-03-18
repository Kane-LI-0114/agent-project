"""
core/search.py
==============
Lightweight live-search helpers for the SmartTutor web app.

The service combines a few free/publicly accessible information sources:
- DuckDuckGo Instant Answer API
- Wikipedia search + summary API
- OpenAlex works API
- arXiv API
- PubMed E-utilities API
- Optional scraping of configured knowledge pages

The collected snippets are converted into a compact system message so the
LLM can answer with fresher, source-backed context when search is enabled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from xml.etree import ElementTree
from typing import Iterable, List, Literal
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from config.settings import SEARCH_ENABLED, SEARCH_KNOWLEDGE_PAGES

logger = logging.getLogger(__name__)

SearchMode = Literal["auto", "on", "off"]

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36 SmartTutor/1.0"
    )
}
_MAX_SOURCE_COUNT = 6
_MAX_SNIPPET_CHARS = 500
_ACADEMIC_KEYWORDS = (
    "paper",
    "papers",
    "research",
    "academic",
    "journal",
    "study",
    "studies",
    "citation",
    "citations",
    "reference",
    "references",
    "arxiv",
    "pubmed",
    "openalex",
    "preprint",
    "literature",
    "dataset",
    "experiment",
    "theorem",
    "proof",
    "calculus",
    "algebra",
    "geometry",
    "statistics",
    "chemistry",
    "finance",
    "economics",
    "philosophy",
)


@dataclass
class SearchSource:
    """Normalized search evidence item."""

    title: str
    url: str
    snippet: str
    provider: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the source for API responses."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "provider": self.provider,
        }


@dataclass
class SearchResult:
    """Bundled result of a search execution."""

    query: str
    sources: List[SearchSource]

    def to_system_message(self) -> str:
        """Serialize the search evidence into a compact system instruction."""
        lines = [
            "You have live search context for the current user message.",
            "Use the information below only when it is relevant and helpful.",
            "Prefer the search evidence over stale world knowledge for factual details.",
            "Do not add a separate Sources section to the final answer.",
            "",
            f"Search query: {self.query}",
            "",
            "Retrieved sources:",
        ]
        for index, source in enumerate(self.sources, start=1):
            lines.append(f"[{index}] {source.title} ({source.provider})")
            lines.append(f"URL: {source.url}")
            lines.append(f"Snippet: {source.snippet}")
            lines.append("")
        return "\n".join(lines).strip()

    def to_dict_list(self) -> List[dict[str, str]]:
        """Return structured sources for the frontend."""
        return [source.to_dict() for source in self.sources]


class _ParagraphExtractor(HTMLParser):
    """Best-effort extractor for visible paragraph/list text from static pages."""

    def __init__(self) -> None:
        super().__init__()
        self._capture_depth = 0
        self._chunks: List[str] = []
        self.texts: List[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in {"p", "li", "article", "section"}:
            self._capture_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"p", "li", "article", "section"} and self._capture_depth > 0:
            self._capture_depth -= 1
            text = self._flush_chunks()
            if text:
                self.texts.append(text)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._capture_depth > 0:
            self._chunks.append(cleaned)
        if self._in_title:
            self.title = f"{self.title} {cleaned}".strip()

    def close(self) -> None:
        super().close()
        text = self._flush_chunks()
        if text:
            self.texts.append(text)

    def _flush_chunks(self) -> str:
        if not self._chunks:
            return ""
        text = " ".join(self._chunks).strip()
        self._chunks.clear()
        return text


class SearchService:
    """Runs optional web search for a single user query."""

    def should_execute(self, query: str, mode: SearchMode) -> bool:
        """Return whether a query should trigger live search."""
        if not SEARCH_ENABLED:
            return False
        normalized_mode: SearchMode = mode if mode in {"auto", "on", "off"} else "auto"
        if normalized_mode == "off":
            return False
        if normalized_mode == "on":
            return True
        return self.should_search(query)

    def get_pending_sources(self, query: str) -> List[dict[str, str]]:
        """Return the provider list shown while search is running."""
        pending = [
            {
                "title": "DuckDuckGo Instant Answers",
                "url": "https://duckduckgo.com/",
                "snippet": "Searching quick web facts and related topics.",
                "provider": "DuckDuckGo",
            },
            {
                "title": "Wikipedia",
                "url": "https://www.wikipedia.org/",
                "snippet": "Searching encyclopedia summaries and article extracts.",
                "provider": "Wikipedia",
            },
        ]
        if self._looks_academic(query):
            pending.extend(
                [
                    {
                        "title": "OpenAlex Works",
                        "url": "https://openalex.org/",
                        "snippet": "Searching scholarly works and citation metadata.",
                        "provider": "OpenAlex",
                    },
                    {
                        "title": "arXiv",
                        "url": "https://arxiv.org/",
                        "snippet": "Searching open-access preprints and abstracts.",
                        "provider": "arXiv",
                    },
                    {
                        "title": "PubMed",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/",
                        "snippet": "Searching biomedical and life-science article summaries.",
                        "provider": "PubMed",
                    },
                ]
            )
        if SEARCH_KNOWLEDGE_PAGES:
            pending.append(
                {
                    "title": "Knowledge Pages",
                    "url": SEARCH_KNOWLEDGE_PAGES[0].url,
                    "snippet": "Scanning configured course-friendly reference pages.",
                    "provider": "Knowledge Page",
                }
            )
        return pending

    async def maybe_search(self, query: str, mode: SearchMode) -> SearchResult | None:
        """Return search evidence when the mode requires it."""
        if not self.should_execute(query, mode):
            return None
        return await self.search(query)

    def should_search(self, query: str) -> bool:
        """Heuristic for deciding whether a question benefits from live search."""
        lower = query.lower()

        search_triggers = (
            "search",
            "look up",
            "internet",
            "online",
            "source",
            "citation",
            "cite",
            "wikipedia",
            "latest",
            "current",
            "today",
            "recent",
        )
        history_triggers = (
            "who",
            "when",
            "where",
            "which",
            "date",
            "year",
            "president",
            "war",
            "revolution",
            "battle",
            "treaty",
            "empire",
            "dynasty",
            "france",
            "rome",
        )

        if any(token in lower for token in search_triggers):
            return True

        if "?" in query and any(token in lower for token in history_triggers):
            return True

        if re.search(r"\b(19|20)\d{2}\b", lower):
            return True

        return False

    def _looks_academic(self, query: str) -> bool:
        lower = query.lower()
        return any(keyword in lower for keyword in _ACADEMIC_KEYWORDS)

    async def search(self, query: str) -> SearchResult | None:
        """Collect best-effort search evidence from multiple free sources."""
        tasks = [
            asyncio.to_thread(self._duckduckgo_instant_answer, query),
            asyncio.to_thread(self._wikipedia_search, query),
            asyncio.to_thread(self._knowledge_page_search, query),
        ]
        if self._looks_academic(query):
            tasks.extend(
                [
                    asyncio.to_thread(self._openalex_search, query),
                    asyncio.to_thread(self._arxiv_search, query),
                    asyncio.to_thread(self._pubmed_search, query),
                ]
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: List[SearchSource] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Search source failed for query %r: %s", query, result)
                continue
            merged.extend(result)

        deduped = self._dedupe_sources(merged)[:_MAX_SOURCE_COUNT]
        if not deduped:
            return None
        return SearchResult(query=query, sources=deduped)

    def _duckduckgo_instant_answer(self, query: str) -> List[SearchSource]:
        params = urlencode(
            {
                "q": query,
                "format": "json",
                "no_html": "1",
                "no_redirect": "1",
                "skip_disambig": "1",
            }
        )
        url = f"https://api.duckduckgo.com/?{params}"
        payload = self._fetch_json(url)

        sources: List[SearchSource] = []
        abstract = self._clean_text(payload.get("AbstractText", ""))
        abstract_url = payload.get("AbstractURL", "")
        heading = self._clean_text(payload.get("Heading", "")) or "DuckDuckGo instant answer"
        if abstract and abstract_url:
            sources.append(
                SearchSource(
                    title=heading,
                    url=abstract_url,
                    snippet=self._truncate(abstract),
                    provider="DuckDuckGo",
                )
            )

        for topic in payload.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text") and topic.get("FirstURL"):
                title = self._clean_text(topic.get("Text", "").split(" - ")[0]) or "DuckDuckGo related topic"
                sources.append(
                    SearchSource(
                        title=title,
                        url=topic["FirstURL"],
                        snippet=self._truncate(self._clean_text(topic["Text"])),
                        provider="DuckDuckGo",
                    )
                )
            elif isinstance(topic, dict):
                for nested in topic.get("Topics", [])[:2]:
                    if nested.get("Text") and nested.get("FirstURL"):
                        title = self._clean_text(nested.get("Text", "").split(" - ")[0]) or "DuckDuckGo related topic"
                        sources.append(
                            SearchSource(
                                title=title,
                                url=nested["FirstURL"],
                                snippet=self._truncate(self._clean_text(nested["Text"])),
                                provider="DuckDuckGo",
                            )
                        )
        return sources

    def _wikipedia_search(self, query: str) -> List[SearchSource]:
        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            + urlencode(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "utf8": "1",
                    "format": "json",
                    "srlimit": "2",
                }
            )
        )
        payload = self._fetch_json(search_url)
        items = payload.get("query", {}).get("search", [])

        sources: List[SearchSource] = []
        for item in items[:2]:
            title = item.get("title")
            if not title:
                continue
            summary_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + quote(title.replace(" ", "_"))
            )
            try:
                summary_payload = self._fetch_json(summary_url)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Wikipedia summary fetch failed for %s: %s", title, exc)
                continue
            extract = self._clean_text(summary_payload.get("extract", ""))
            content_urls = summary_payload.get("content_urls", {}).get("desktop", {})
            page_url = content_urls.get("page") or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            if extract:
                sources.append(
                    SearchSource(
                        title=summary_payload.get("title", title),
                        url=page_url,
                        snippet=self._truncate(extract),
                        provider="Wikipedia",
                    )
                )
        return sources

    def _knowledge_page_search(self, query: str) -> List[SearchSource]:
        lower = query.lower()
        matched_pages = [
            page for page in SEARCH_KNOWLEDGE_PAGES
            if any(keyword.lower() in lower for keyword in page.keywords)
        ]
        sources: List[SearchSource] = []
        for page in matched_pages[:2]:
            try:
                html = self._fetch_text(page.url)
                snippet = self._extract_relevant_text(html, query)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Knowledge page fetch failed for %s: %s", page.url, exc)
                continue
            if snippet:
                sources.append(
                    SearchSource(
                        title=page.name,
                        url=page.url,
                        snippet=self._truncate(snippet),
                        provider="Knowledge Page",
                    )
                )
        return sources

    def _openalex_search(self, query: str) -> List[SearchSource]:
        url = (
            "https://api.openalex.org/works?"
            + urlencode(
                {
                    "search": query,
                    "per-page": "2",
                    "sort": "relevance_score:desc",
                }
            )
        )
        payload = self._fetch_json(url)
        results = payload.get("results", [])
        sources: List[SearchSource] = []
        for item in results[:2]:
            title = self._clean_text(item.get("display_name", ""))
            if not title:
                continue
            abstract = self._openalex_abstract(item.get("abstract_inverted_index"))
            venue = self._clean_text(
                item.get("primary_location", {}).get("source", {}).get("display_name", "")
            )
            year = item.get("publication_year")
            snippet_parts = [part for part in [venue, str(year) if year else ""] if part]
            if abstract:
                snippet_parts.append(abstract)
            url = item.get("primary_location", {}).get("landing_page_url") or item.get("id", "")
            if not url:
                continue
            sources.append(
                SearchSource(
                    title=title,
                    url=url,
                    snippet=self._truncate(" | ".join(snippet_parts) or "OpenAlex scholarly work"),
                    provider="OpenAlex",
                )
            )
        return sources

    def _arxiv_search(self, query: str) -> List[SearchSource]:
        url = (
            "https://export.arxiv.org/api/query?"
            + urlencode(
                {
                    "search_query": f"all:{query}",
                    "start": "0",
                    "max_results": "2",
                }
            )
        )
        xml_text = self._fetch_text(url)
        root = ElementTree.fromstring(xml_text)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        sources: List[SearchSource] = []
        for entry in root.findall("atom:entry", namespace)[:2]:
            title = self._clean_text(entry.findtext("atom:title", default="", namespaces=namespace))
            summary = self._clean_text(entry.findtext("atom:summary", default="", namespaces=namespace))
            link = self._clean_text(entry.findtext("atom:id", default="", namespaces=namespace))
            authors = [
                self._clean_text(author.findtext("atom:name", default="", namespaces=namespace))
                for author in entry.findall("atom:author", namespace)[:2]
            ]
            author_text = ", ".join([author for author in authors if author])
            if title and link:
                sources.append(
                    SearchSource(
                        title=title,
                        url=link,
                        snippet=self._truncate(" | ".join([part for part in [author_text, summary] if part])),
                        provider="arXiv",
                    )
                )
        return sources

    def _pubmed_search(self, query: str) -> List[SearchSource]:
        search_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            + urlencode(
                {
                    "db": "pubmed",
                    "retmode": "json",
                    "retmax": "2",
                    "sort": "relevance",
                    "term": query,
                }
            )
        )
        search_payload = self._fetch_json(search_url)
        id_list = search_payload.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        summary_url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
            + urlencode(
                {
                    "db": "pubmed",
                    "retmode": "json",
                    "id": ",".join(id_list[:2]),
                }
            )
        )
        summary_payload = self._fetch_json(summary_url)
        result_map = summary_payload.get("result", {})
        sources: List[SearchSource] = []
        for pmid in id_list[:2]:
            item = result_map.get(pmid, {})
            title = self._clean_text(item.get("title", ""))
            if not title:
                continue
            authors = [
                self._clean_text(author.get("name", ""))
                for author in item.get("authors", [])[:2]
                if isinstance(author, dict)
            ]
            journal = self._clean_text(item.get("source", ""))
            pubdate = self._clean_text(item.get("pubdate", ""))
            snippet = " | ".join(
                [part for part in [", ".join([name for name in authors if name]), journal, pubdate] if part]
            )
            sources.append(
                SearchSource(
                    title=title,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    snippet=self._truncate(snippet or "PubMed article"),
                    provider="PubMed",
                )
            )
        return sources

    def _extract_relevant_text(self, html: str, query: str) -> str:
        parser = _ParagraphExtractor()
        parser.feed(html)
        parser.close()

        terms = self._meaningful_terms(query)
        if not parser.texts:
            fallback = self._clean_text(re.sub(r"<[^>]+>", " ", html))
            return self._truncate(fallback)

        scored = sorted(
            parser.texts,
            key=lambda text: self._score_text(text, terms),
            reverse=True,
        )
        best = [text for text in scored if text][:2]
        return " ".join(best).strip()

    def _score_text(self, text: str, terms: Iterable[str]) -> int:
        lower = text.lower()
        score = 0
        for term in terms:
            if term in lower:
                score += 3
        score += max(0, 2 - abs(len(text) - 240) // 120)
        return score

    def _meaningful_terms(self, query: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", query.lower())
        stopwords = {
            "what", "when", "where", "which", "with", "from", "that", "this",
            "your", "about", "into", "have", "would", "could", "should",
            "history", "homework", "question", "explain",
        }
        seen = []
        for token in tokens:
            if token in stopwords or token in seen:
                continue
            seen.append(token)
        return seen[:8]

    def _dedupe_sources(self, sources: List[SearchSource]) -> List[SearchSource]:
        seen = set()
        deduped: List[SearchSource] = []
        for source in sources:
            key = (source.url.strip().lower(), source.title.strip().lower())
            if not source.snippet or key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped

    def _openalex_abstract(self, inverted_index: object) -> str:
        if not isinstance(inverted_index, dict):
            return ""
        positions: List[tuple[int, str]] = []
        for word, indexes in inverted_index.items():
            if not isinstance(word, str) or not isinstance(indexes, list):
                continue
            for index in indexes:
                if isinstance(index, int):
                    positions.append((index, word))
        if not positions:
            return ""
        ordered_words = [word for _, word in sorted(positions, key=lambda item: item[0])]
        return self._clean_text(" ".join(ordered_words))

    def _fetch_json(self, url: str) -> dict:
        payload = self._fetch_text(url)
        return json.loads(payload)

    def _fetch_text(self, url: str) -> str:
        request = Request(url, headers=_REQUEST_HEADERS)
        with urlopen(request, timeout=8) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read().decode(charset, errors="replace")
        return body

    def _clean_text(self, text: str) -> str:
        text = unescape(text or "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _truncate(self, text: str) -> str:
        cleaned = self._clean_text(text)
        if len(cleaned) <= _MAX_SNIPPET_CHARS:
            return cleaned
        return cleaned[: _MAX_SNIPPET_CHARS - 1].rstrip() + "…"
