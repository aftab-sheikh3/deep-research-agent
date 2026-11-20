"""The research workflow as a LangGraph graph (orchestrator-workers + reflexion).

    plan -> review_plan (optional human approval: interrupt) ─Send×N─> research (parallel workers)
                                                                           │
               ┌──────────────── Send×M (follow-up queries) ───────────────┤
               │                                                           v
            critique <───────────────────────────────────────────────── write
               │ score >= bar, no follow-ups, or out of rounds
               v
            finalize (deterministic citation check + References) -> END

- plan:      structured Plan (sub-questions + search queries)
- research:  one worker per sub-question: search -> extract numbered findings (structured Notes);
             sources and findings are merged into the shared state with list reducers
- write:     markdown report citing global source numbers [n]; on later rounds it also gets the critique
- critique:  structured Critique (score 1-10, gaps, unsupported claims, follow-up queries)
- finalize:  removes invalid [n], counts uncited paragraphs, rebuilds the References section

Every step is checkpointed (SQLite), so a run can be inspected, resumed after a crash, or paused
for plan approval and continued later with the same thread id.
"""
from __future__ import annotations

import operator
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from . import citations
from .config import Settings, get_settings
from .llm import get_chat_model
from .schemas import Critique, Notes, Plan, ResearchStatus, Source, SubQuestion
from .search import MultiSearch, build_search

PLAN_PROMPT = """You are planning a research report on the topic below.
Break it into {n} complementary sub-questions that together give a complete, balanced answer
(background, how it works, evidence/results, limitations or open debates). For each, give 1-2 short
search-engine queries.

Topic: {topic}"""

NOTES_PROMPT = """Extract the facts relevant to the question from the numbered search results.
Only use what the results say. Keep numbers, dates and names. Cite result numbers for every finding.

Question: {question}

Search results:
{results}"""

WRITE_PROMPT = """Write a research report in Markdown on: {topic}

Rules:
- Use only the findings below. Cite them with their source numbers, like [3] or [1][4], in every paragraph.
- Structure: a 2-3 sentence summary, then one section (## heading) per sub-question, then "## Limitations and open questions".
- Be specific (numbers, dates, names); say where sources disagree; no filler.
- Do not write a References section (it is added automatically).
{revision}
Sub-questions:
{plan}

Findings (with source numbers):
{findings}

Sources:
{sources}"""

REVISION = """
This is a revision. A reviewer scored the previous draft {score}/10 and said:
- Missing: {missing}
- Unsupported: {unsupported}
New findings from follow-up searches are included below. Fix every point.

Previous draft:
{draft}
"""

CRITIQUE_PROMPT = """You are a demanding reviewer of research reports. Score the report 1-10 for coverage of the
topic, accuracy against its sources, citation support for every claim, and clarity. List what is missing, any
statement not supported by a cited source, and up to 3 search queries that would fill the gaps (none if the
report is already excellent).

Topic: {topic}

Report:
{draft}

Available sources:
{sources}"""


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models (e.g. qwen3) emit before the answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


class ResearchState(TypedDict, total=False):
    topic: str
    plan: list[dict]
    sources: Annotated[list[dict], operator.add]
    findings: Annotated[list[dict], operator.add]
    critiques: Annotated[list[dict], operator.add]
    draft: str
    rounds: int
    report: str
    citations: dict
    started: float
    finished: float


class WorkerState(TypedDict):
    question: str
    queries: list[str]


def number_sources(raw: list[dict]) -> dict[int, Source]:
    """Global source numbers in order of first appearance, de-duplicated by URL."""
    out: dict[int, Source] = {}
    seen: dict[str, int] = {}
    for s in raw:
        if s["url"] not in seen:
            seen[s["url"]] = len(out) + 1
            out[seen[s["url"]]] = Source(n=seen[s["url"]], **s)
    return out


def make_checkpointer(path: str | None):
    if not path or path == ":memory:":
        return InMemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


class Researcher:
    def __init__(self, settings: Settings | None = None, llm: BaseChatModel | None = None,
                 search: MultiSearch | None = None, checkpointer=None):
        self.settings = s = settings or get_settings()
        self.llm = llm or get_chat_model(s)
        self.search = search or build_search(s.search)
        self.checkpointer = checkpointer if checkpointer is not None else make_checkpointer(s.checkpoint_db)
        self.graph = self._build().compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------ nodes
    def _plan(self, state: ResearchState) -> dict:
        plan = self.llm.with_structured_output(Plan).invoke(
            PLAN_PROMPT.format(n=f"3-{self.settings.max_sub_questions}", topic=state["topic"]))
        subs = [sq for sq in plan.sub_questions if sq.question.strip()][: self.settings.max_sub_questions]
        if not subs:  # a weak model returned nothing usable: research the topic itself
            subs = [SubQuestion(question=state["topic"], queries=[state["topic"]])]
        return {"plan": [sq.model_dump() for sq in subs], "rounds": 0, "started": time.time()}

    def _review_plan(self, state: ResearchState) -> Command:
        plan = state["plan"]
        if self.settings.approve_plan:
            decision = interrupt({"plan": plan, "hint": "resume with {'approved': true} or {'sub_questions': [...]}"})
            if isinstance(decision, dict) and decision.get("sub_questions"):
                plan = [{"question": q, "queries": [q]} for q in decision["sub_questions"] if q.strip()]
            elif isinstance(decision, dict) and decision.get("approved") is False:
                return Command(goto=END, update={"report": "Research cancelled at plan review."})
        sends = [Send("research", {"question": sq["question"], "queries": sq["queries"] or [sq["question"]]})
                 for sq in plan]
        return Command(goto=sends, update={"plan": plan})

    def _research(self, worker: WorkerState) -> dict:
        results, seen = [], set()
        for query in worker["queries"][:2]:
            for r in self.search.search(query, self.settings.results_per_query):
                if r.url not in seen:
                    seen.add(r.url)
                    results.append(r)
        if not results:
            return {"sources": [], "findings": []}
        numbered = "\n\n".join(f"[{i}] {r.title} ({r.url})\n{r.content[: self.settings.source_chars]}"
                               for i, r in enumerate(results, 1))
        try:
            notes = self.llm.with_structured_output(Notes).invoke(
                NOTES_PROMPT.format(question=worker["question"], results=numbered))
            findings = notes.findings
        except Exception:  # keep the sources even if extraction fails
            findings = []
        sources = [{"title": r.title, "url": r.url, "provider": r.provider,
                    "content": r.content[: self.settings.source_chars]} for r in results]
        return {"sources": sources, "findings": [
            {"question": worker["question"], "claim": f.claim,
             "urls": [results[n - 1].url for n in f.sources if 1 <= n <= len(results)]} for f in findings]}

    def _write(self, state: ResearchState) -> dict:
        sources = number_sources(state.get("sources", []))
        by_url = {s.url: n for n, s in sources.items()}
        lines = []
        for f in state.get("findings", []):
            refs = "".join(f"[{by_url[u]}]" for u in f["urls"] if u in by_url)
            if refs:
                lines.append(f"- ({f['question']}) {f['claim']} {refs}")
        revision = ""
        if state.get("critiques"):
            last = state["critiques"][-1]
            revision = REVISION.format(score=last["score"], missing="; ".join(last["missing"]) or "-",
                                       unsupported="; ".join(last["unsupported"]) or "-", draft=state.get("draft", ""))
        prompt = WRITE_PROMPT.format(
            topic=state["topic"], revision=revision, plan="\n".join(f"- {sq['question']}" for sq in state["plan"]),
            findings="\n".join(lines) or "(no findings)",
            sources="\n".join(f"[{n}] {s.title} - {s.url}" for n, s in sources.items()))
        return {"draft": strip_thinking(str(self.llm.invoke(prompt).content))}

    def _critique(self, state: ResearchState) -> dict:
        sources = number_sources(state.get("sources", []))
        try:
            c = self.llm.with_structured_output(Critique).invoke(CRITIQUE_PROMPT.format(
                topic=state["topic"], draft=state["draft"],
                sources="\n".join(f"[{n}] {s.title}" for n, s in sources.items())))
        except Exception:  # an unusable critique ends the loop rather than the run
            c = Critique(score=self.settings.quality_bar, missing=[], unsupported=[], follow_up_queries=[])
        return {"critiques": [c.model_dump()], "rounds": state.get("rounds", 0) + 1}

    def _after_critique(self, state: ResearchState):
        last = state["critiques"][-1]
        done = (last["score"] >= self.settings.quality_bar or not last["follow_up_queries"]
                or state["rounds"] >= self.settings.max_rounds)
        if done:
            return "finalize"
        return [Send("research", {"question": q, "queries": [q]}) for q in last["follow_up_queries"][:3]]

    def _finalize(self, state: ResearchState) -> dict:
        report, check = citations.check(state.get("draft", ""), number_sources(state.get("sources", [])))
        return {"report": report, "citations": check.model_dump(), "finished": time.time()}

    def _build(self) -> StateGraph:
        g = StateGraph(ResearchState)
        g.add_node("plan", self._plan)
        g.add_node("review_plan", self._review_plan, destinations=("research", END))
        g.add_node("research", self._research)
        g.add_node("write", self._write)
        g.add_node("critique", self._critique)
        g.add_node("finalize", self._finalize)
        g.add_edge(START, "plan")
        g.add_edge("plan", "review_plan")
        g.add_edge("research", "write")  # runs once all parallel workers of the step are done
        g.add_edge("write", "critique")
        g.add_conditional_edges("critique", self._after_critique, ["research", "finalize"])
        g.add_edge("finalize", END)
        return g

    # ------------------------------------------------------------------ public API
    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}

    def start(self, topic: str, thread_id: str | None = None) -> ResearchStatus:
        thread_id = thread_id or uuid.uuid4().hex[:12]
        if self.graph.get_state(self._config(thread_id)).values:
            raise ValueError(f"thread {thread_id!r} already exists: use resume() or show, or pick a new id")
        self.graph.invoke({"topic": topic}, self._config(thread_id))
        return self.status(thread_id)

    def resume(self, thread_id: str, decision: Any = None) -> ResearchStatus:
        """Continue a paused (plan approval) or interrupted run from its last checkpoint."""
        snapshot = self.graph.get_state(self._config(thread_id))
        if snapshot.interrupts:
            self.graph.invoke(Command(resume=decision if decision is not None else {"approved": True}),
                              self._config(thread_id))
        elif snapshot.next:  # crashed mid-run: re-run the pending steps
            self.graph.invoke(None, self._config(thread_id))
        return self.status(thread_id)

    def status(self, thread_id: str) -> ResearchStatus:
        snapshot = self.graph.get_state(self._config(thread_id))
        v = snapshot.values
        if snapshot.interrupts:
            state = "awaiting_approval"
        elif snapshot.next:
            state = "running"
        else:
            state = "done"
        sources = number_sources(v.get("sources", []))
        return ResearchStatus(
            thread_id=thread_id, status=state, topic=v.get("topic", ""),
            plan=[SubQuestion(**sq) for sq in v.get("plan", [])], report=v.get("report", ""),
            sources=list(sources.values()), critiques=[Critique(**c) for c in v.get("critiques", [])],
            citations=v.get("citations"), rounds=v.get("rounds", 0),
            seconds=round(v["finished"] - v["started"], 1) if v.get("finished") and v.get("started") else 0.0)
