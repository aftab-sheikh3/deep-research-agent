"""Agent vs baseline on research topics with hand-written key points.

    python -m evals.run_eval              # or: research eval / make eval
    python -m evals.run_eval --limit 2

For each topic, both systems write a report (baseline = one search + one LLM call, see
deep_research/baseline.py). Metrics:
- key-point coverage: an LLM judge marks which of the topic's 5 key points each report covers
- citation validity: share of [n] references that point to a real source (deterministic)
- uncited paragraphs: share of body paragraphs with no citation (deterministic)
- sources cited, critique rounds, latency
- pairwise preference: the judge compares the two reports twice with the order swapped; a win
  counts only if both orders agree (this controls for position bias), otherwise it's a tie

The judge is the configured model unless JUDGE_PROVIDER / JUDGE_MODEL are set (using a stronger,
different model than the one under test is better practice).
Writes evals/results.json and evals/results.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import time
from pathlib import Path
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from deep_research.baseline import baseline_report
from deep_research.config import Settings, get_settings
from deep_research.graph import Researcher, make_checkpointer
from deep_research.llm import get_chat_model
from deep_research.schemas import ResearchStatus
from deep_research.search import build_search

HERE = Path(__file__).parent


class Coverage(BaseModel):
    covered: list[bool] = Field(description="For each key point, in order: does the report cover it?")


class Preference(BaseModel):
    winner: Literal["A", "B", "tie"] = Field(description="The better research report overall")
    reason: str = ""


COVERAGE_PROMPT = """For each numbered key point, answer whether the report below covers it (true/false).
Key points:
{points}

Report:
{report}"""

PREFERENCE_PROMPT = """Two research reports answer the same topic. Which is better overall: more complete, more
accurate, better supported by citations, clearer? Answer A, B or tie.

Topic: {topic}

Report A:
{a}

Report B:
{b}"""


def load_topics(limit: int | None = None) -> list[dict]:
    return [json.loads(x) for x in (HERE / "topics.jsonl").read_text().splitlines() if x.strip()][:limit]


def coverage(judge: BaseChatModel, report: str, points: list[str]) -> float | None:
    try:
        out = judge.with_structured_output(Coverage).invoke(COVERAGE_PROMPT.format(
            points="\n".join(f"{i}. {p}" for i, p in enumerate(points, 1)), report=report[:12000]))
    except Exception:
        return None
    flags = (list(out.covered) + [False] * len(points))[: len(points)]
    return sum(flags) / len(points)


def prefer(judge: BaseChatModel, topic: str, agent: str, baseline: str) -> str:
    """'agent', 'baseline' or 'tie'; both presentation orders must agree for a win."""
    try:
        first = judge.with_structured_output(Preference).invoke(
            PREFERENCE_PROMPT.format(topic=topic, a=agent[:10000], b=baseline[:10000])).winner
        second = judge.with_structured_output(Preference).invoke(
            PREFERENCE_PROMPT.format(topic=topic, a=baseline[:10000], b=agent[:10000])).winner
    except Exception:
        return "tie"
    if first == "A" and second == "B":
        return "agent"
    if first == "B" and second == "A":
        return "baseline"
    return "tie"


def report_metrics(st: ResearchStatus) -> dict:
    c = st.citations
    return {"valid_citation_rate": c.valid_rate if c else 0.0,
            "uncited_paragraph_rate": (c.uncited_paragraphs / c.paragraphs) if c and c.paragraphs else None,
            "sources_cited": len(c.cited) if c else 0, "seconds": st.seconds, "rounds": st.rounds,
            "words": len(st.report.split())}


def _mean(rows: list[dict], key: str):
    values = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(values), 3) if values else None


def judge_model(settings: Settings) -> BaseChatModel:
    if os.getenv("JUDGE_PROVIDER"):
        return get_chat_model(dataclasses.replace(settings, provider=os.environ["JUDGE_PROVIDER"],
                                                  model=os.getenv("JUDGE_MODEL") or None))
    return get_chat_model(settings)


def run(limit: int | None = None, settings: Settings | None = None, llm: BaseChatModel | None = None,
        judge: BaseChatModel | None = None, search=None) -> dict:
    settings = settings or get_settings()
    llm = llm or get_chat_model(settings)
    judge = judge or judge_model(settings)
    search = search or build_search(settings.search)
    agent = Researcher(dataclasses.replace(settings, approve_plan=False), llm=llm, search=search,
                       checkpointer=make_checkpointer(":memory:"))
    rows = []
    for t in load_topics(limit):
        row: dict = {"id": t["id"], "topic": t["topic"]}
        for system in ("agent", "baseline"):
            start = time.time()
            try:
                st = agent.start(t["topic"], f"eval-{t['id']}-{int(start)}") if system == "agent" \
                    else baseline_report(t["topic"], llm, search, settings)
                row[system] = {**report_metrics(st), "coverage": coverage(judge, st.report, t["key_points"]),
                               "report": st.report}
            except Exception as e:
                row[system] = {"error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - start, 1)}
        if "report" in row["agent"] and "report" in row["baseline"]:
            row["preference"] = prefer(judge, t["topic"], row["agent"]["report"], row["baseline"]["report"])
        rows.append(row)
        a, b = row["agent"], row["baseline"]
        print(f"{t['id']}: coverage agent {a.get('coverage')} vs baseline {b.get('coverage')} · "
              f"preferred {row.get('preference', '-')} · {a.get('seconds')}s vs {b.get('seconds')}s")

    def side(name: str) -> dict:
        data = [r[name] for r in rows if "error" not in r[name]]
        return {k: _mean(data, k) for k in ("coverage", "valid_citation_rate", "uncited_paragraph_rate",
                                             "sources_cited", "rounds", "words", "seconds")} | {
            "errors": sum("error" in r[name] for r in rows)}

    prefs = [r.get("preference", "tie") for r in rows]
    summary = {"agent": side("agent"), "baseline": side("baseline"), "n": len(rows),
               "wins": {k: prefs.count(k) for k in ("agent", "baseline", "tie")}}
    out = {"provider": settings.provider, "model": settings.model_name, "search": settings.search,
           "summary": summary, "topics": rows}
    (HERE / "results.json").write_text(json.dumps(out, indent=2, default=str))
    md = to_markdown(summary, settings)
    (HERE / "results.md").write_text(md)
    print("\n" + md)
    return out


def to_markdown(s: dict, settings: Settings) -> str:
    def pct(x):
        return "-" if x is None else f"{x:.0%}"

    def val(x):
        return "-" if x is None else x

    a, b = s["agent"], s["baseline"]
    rows = [("Key-point coverage (judge)", pct(a["coverage"]), pct(b["coverage"])),
            ("Valid citations", pct(a["valid_citation_rate"]), pct(b["valid_citation_rate"])),
            ("Uncited paragraphs", pct(a["uncited_paragraph_rate"]), pct(b["uncited_paragraph_rate"])),
            ("Sources cited", val(a["sources_cited"]), val(b["sources_cited"])),
            ("Critique rounds", val(a["rounds"]), "-"),
            ("Words", val(a["words"]), val(b["words"])),
            ("Seconds per report", val(a["seconds"]), val(b["seconds"])),
            ("Errors", a["errors"], b["errors"])]
    w = s["wins"]
    return "\n".join([
        f"Model `{settings.provider}:{settings.model_name}` · search `{settings.search}` · {s['n']} topics · "
        f"run {time.strftime('%Y-%m-%d')}", "",
        "| Metric | Agent | Baseline |", "|---|---|---|", *[f"| {m} | {x} | {y} |" for m, x, y in rows], "",
        f"Pairwise preference (both orders must agree): agent {w['agent']} · baseline {w['baseline']} · tie {w['tie']}",
        ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Research Agent: agent vs baseline evaluation")
    parser.add_argument("--limit", type=int, help="first N topics")
    run(parser.parse_args().limit)


if __name__ == "__main__":
    main()
