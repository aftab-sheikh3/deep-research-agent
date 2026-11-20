import pytest

from deep_research.config import Settings
from deep_research.llm import ScriptedChatModel
from deep_research.search import FakeSearch, MultiSearch

TOPIC = "How does hybrid search with reranking improve retrieval-augmented generation?"
PLAN = {"sub_questions": [
    {"question": "What is hybrid search?", "queries": ["hybrid search BM25 dense"]},
    {"question": "What does cross-encoder reranking add?", "queries": ["cross-encoder reranking"]}]}
DRAFT = ("Hybrid search fuses lexical and dense rankings and usually beats either alone [1]. Rerankers help further [2].\n\n"
         "## What is hybrid search?\n\nHybrid search combines BM25 with dense retrievers using reciprocal rank fusion, "
         "which usually beats either method alone on the BEIR benchmark [1][9].\n\n"
         "## Reranking\n\nCross-encoders read the query and document together and improve nDCG@10 at a latency cost, "
         "according to the available evidence on reranking the top 100 results.")


def notes(prompt) -> dict:
    """Fake extraction: one finding citing result [1] of whatever the worker searched."""
    return {"findings": [{"claim": "A finding from the first search result.", "sources": [1]}]}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(provider="fake", search="fake", max_rounds=2, quality_bar=8,
                    checkpoint_db=str(tmp_path / "ck.sqlite"))


@pytest.fixture
def search() -> MultiSearch:
    return MultiSearch([FakeSearch()])


def scripted(critiques, drafts=None) -> ScriptedChatModel:
    return ScriptedChatModel(script=list(drafts or [DRAFT] * 3),
                             structured={"Plan": PLAN, "Notes": notes, "Critique": list(critiques)})
