"""Search providers behind one interface: `search(query, k) -> list[SearchResult]`.

    tavily     web search API built for agents (TAVILY_API_KEY), returns cleaned page content
    wikipedia  MediaWiki REST search + page extracts (free, no key)
    arxiv      arXiv Atom API: titles + abstracts of papers (free, no key)
    fake       a small offline corpus (tests, CI)

Several providers can be combined (`SEARCH_PROVIDERS=wikipedia,arxiv`); results are interleaved
and de-duplicated by URL. All network calls have timeouts and retries; a failing provider
returns no results instead of failing the whole research run.
"""
from __future__ import annotations

import html
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Protocol

import httpx

log = logging.getLogger(__name__)
USER_AGENT = "deep-research-agent/0.1 (research tool; https://github.com/)"


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    provider: str


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[SearchResult]: ...


def _get(client: httpx.Client, url: str, retries: int = 3, **kw) -> httpx.Response:
    for attempt in range(retries):
        try:
            r = client.get(url, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            return r
        except httpx.HTTPError:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * 2 ** attempt)
    raise RuntimeError("unreachable")


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip()


class TavilySearch:
    name = "tavily"

    def __init__(self, api_key: str | None = None, timeout: float = 30):
        self.api_key = api_key or os.environ["TAVILY_API_KEY"]
        self.client = httpx.Client(timeout=timeout)

    def search(self, query: str, k: int) -> list[SearchResult]:
        r = self.client.post("https://api.tavily.com/search", json={
            "query": query, "max_results": k, "search_depth": "advanced", "include_raw_content": False},
            headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        return [SearchResult(x.get("title", ""), x["url"], _clean(x.get("content", "")), self.name)
                for x in r.json().get("results", [])]


class WikipediaSearch:
    name = "wikipedia"

    def __init__(self, lang: str = "en", timeout: float = 20):
        self.base = f"https://{lang}.wikipedia.org"
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})

    def search(self, query: str, k: int) -> list[SearchResult]:
        pages = _get(self.client, f"{self.base}/w/rest.php/v1/search/page", params={"q": query, "limit": k}).json()
        titles = [p["key"] for p in pages.get("pages", [])][:k]
        if not titles:
            return []
        extracts = _get(self.client, f"{self.base}/w/api.php", params={
            "action": "query", "prop": "extracts", "explaintext": 1, "exintro": 0, "exchars": 2500,
            "titles": "|".join(titles), "format": "json", "formatversion": 2}).json()
        by_title = {p["title"].replace(" ", "_"): p.get("extract", "") for p in extracts.get("query", {}).get("pages", [])}
        return [SearchResult(t.replace("_", " "), f"{self.base}/wiki/{t}", _clean(by_title.get(t, "")), self.name)
                for t in titles if by_title.get(t)]


class ArxivSearch:
    name = "arxiv"
    NS = {"a": "http://www.w3.org/2005/Atom"}

    def __init__(self, timeout: float = 30):
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True)

    def search(self, query: str, k: int) -> list[SearchResult]:
        r = _get(self.client, "https://export.arxiv.org/api/query",
                 params={"search_query": f"all:{query}", "max_results": k, "sortBy": "relevance"})
        out = []
        for entry in ET.fromstring(r.text).findall("a:entry", self.NS):
            title = _clean(entry.findtext("a:title", "", self.NS))
            url = entry.findtext("a:id", "", self.NS).replace("http://", "https://")
            abstract = _clean(entry.findtext("a:summary", "", self.NS))
            year = (entry.findtext("a:published", "", self.NS) or "")[:4]
            out.append(SearchResult(f"{title} ({year})" if year else title, url, abstract, self.name))
        return out


FAKE_CORPUS = [
    ("Retrieval-augmented generation", "https://example.org/rag",
     "Retrieval-augmented generation (RAG) combines a retriever with a language model. Lewis et al. (2020) "
     "introduced RAG, which reduced hallucination on knowledge-intensive tasks compared with closed-book models."),
    ("BM25", "https://example.org/bm25",
     "BM25 is a lexical ranking function based on term frequency and inverse document frequency. It remains a "
     "strong zero-shot baseline on the BEIR benchmark."),
    ("Dense retrieval", "https://example.org/dense",
     "Dense retrievers embed queries and documents with neural encoders. DPR (2020) outperformed BM25 on open-domain "
     "QA, but dense models can generalise worse to new domains than BM25."),
    ("Hybrid search", "https://example.org/hybrid",
     "Hybrid search fuses lexical and dense rankings, often with reciprocal rank fusion, and usually beats either "
     "alone on BEIR."),
    ("Cross-encoder reranking", "https://example.org/rerank",
     "Cross-encoders read the query and the document together. Reranking the top 100 results with a cross-encoder "
     "improves nDCG@10 substantially at a higher latency cost."),
    ("Hallucination in LLMs", "https://example.org/hallucination",
     "Language models can produce fluent but unsupported statements. Grounding answers in retrieved sources and "
     "checking citations reduces, but does not eliminate, hallucination."),
]


class FakeSearch:
    """Keyword-overlap search over FAKE_CORPUS."""

    name = "fake"

    def search(self, query: str, k: int) -> list[SearchResult]:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for title, url, text in FAKE_CORPUS:
            words = set(re.findall(r"[a-z0-9]+", f"{title} {text}".lower()))
            if overlap := len(q & words):
                scored.append((overlap, SearchResult(title, url, text, self.name)))
        return [r for _, r in sorted(scored, key=lambda x: -x[0])][:k]


PROVIDERS = {"tavily": TavilySearch, "wikipedia": WikipediaSearch, "arxiv": ArxivSearch, "fake": FakeSearch}


class MultiSearch:
    def __init__(self, providers: list[SearchProvider]):
        self.providers = providers
        self.name = "+".join(p.name for p in providers)

    def search(self, query: str, k: int) -> list[SearchResult]:
        per_provider = []
        for p in self.providers:
            try:
                per_provider.append(p.search(query, k))
            except Exception as e:  # one provider down must not stop the run
                log.warning("search provider %s failed for %r: %s", p.name, query, e)
                per_provider.append([])
        seen, out = set(), []
        for rank in range(max((len(r) for r in per_provider), default=0)):  # interleave providers
            for results in per_provider:
                if rank < len(results) and results[rank].url not in seen:
                    seen.add(results[rank].url)
                    out.append(results[rank])
        return out[:k]


def build_search(spec: str) -> MultiSearch:
    names = [n.strip().lower() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in PROVIDERS]
    if unknown or not names:
        raise ValueError(f"unknown search provider(s) {unknown}; use {', '.join(PROVIDERS)}")
    return MultiSearch([PROVIDERS[n]() for n in names])
