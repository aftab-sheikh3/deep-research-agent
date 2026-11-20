from deep_research import citations
from deep_research.schemas import Source
from deep_research.search import FakeSearch, MultiSearch, SearchResult, build_search

SOURCES = {n: Source(n=n, title=f"T{n}", url=f"https://x/{n}", provider="fake") for n in (1, 2, 3)}


def test_check_removes_invalid_and_builds_references():
    report = ("# Title\n\nShort.\n\nA long enough paragraph with a claim that needs support from a source [1, 7].\n\n"
              "Another long paragraph with more claims that are not supported by any source at all here.\n\n"
              "## References\n[1] old list")
    final, rep = citations.check(report, SOURCES)
    assert "[7]" not in final and "[1]" in final and "old list" not in final
    assert rep.cited == [1] and rep.invalid == [7] and rep.unused == [2, 3]
    assert rep.paragraphs == 2 and rep.uncited_paragraphs == 1 and rep.valid_rate == 0.5
    assert final.rstrip().endswith("[1] T1. https://x/1")


def test_no_citations_no_reference_section():
    final, rep = citations.check("Plain text.", SOURCES)
    assert "References" not in final and rep.valid_rate == 0.0


class Boom:
    name = "boom"

    def search(self, q, k):
        raise ConnectionError("down")


class Static:
    def __init__(self, name, urls):
        self.name, self.urls = name, urls

    def search(self, q, k):
        return [SearchResult(u, u, "text", self.name) for u in self.urls]


def test_multisearch_interleaves_dedupes_and_survives_failures():
    s = MultiSearch([Static("a", ["u1", "u2", "u3"]), Boom(), Static("b", ["u2", "u4"])])
    assert [r.url for r in s.search("q", 10)] == ["u1", "u2", "u4", "u3"]


def test_fake_search_and_builder():
    assert FakeSearch().search("cross-encoder reranking", 2)[0].url.endswith("/rerank")
    assert build_search("fake").name == "fake"
    try:
        build_search("bing")
    except ValueError as e:
        assert "unknown" in str(e)
    else:
        raise AssertionError
