"""Baseline for the evaluation: one search on the topic, one LLM call, the same citation check.

This is what a plain "search + answer" assistant does. The agent has to beat it to justify its cost.
"""
from __future__ import annotations

import time

from langchain_core.language_models.chat_models import BaseChatModel

from . import citations
from .config import Settings
from .graph import number_sources, strip_thinking
from .schemas import ResearchStatus
from .search import MultiSearch

BASELINE_PROMPT = """Write a research report in Markdown on: {topic}
Use only the numbered sources below and cite them like [2] in every paragraph.
Start with a 2-3 sentence summary, then a few ## sections. Do not write a References section.

Sources:
{sources}"""


def baseline_report(topic: str, llm: BaseChatModel, search: MultiSearch, settings: Settings) -> ResearchStatus:
    start = time.time()
    results = search.search(topic, settings.results_per_query * 2)
    sources = number_sources([{"title": r.title, "url": r.url, "provider": r.provider,
                               "content": r.content[: settings.source_chars]} for r in results])
    numbered = "\n\n".join(f"[{n}] {s.title} ({s.url})\n{s.content}" for n, s in sources.items())
    draft = strip_thinking(str(llm.invoke(BASELINE_PROMPT.format(topic=topic, sources=numbered or "(none)")).content))
    report, check = citations.check(draft, sources)
    return ResearchStatus(thread_id="baseline", status="done", topic=topic, report=report,
                          sources=list(sources.values()), citations=check, seconds=round(time.time() - start, 1))
