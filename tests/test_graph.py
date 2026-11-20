import dataclasses

from deep_research.graph import Researcher, number_sources, strip_thinking
from tests.conftest import TOPIC, scripted

GOOD = {"score": 9, "missing": [], "unsupported": [], "follow_up_queries": []}
WEAK = {"score": 4, "missing": ["latency numbers"], "unsupported": [], "follow_up_queries": ["hallucination citations"]}


def test_full_run_writes_checked_report(settings, search):
    llm = scripted([GOOD])
    st = Researcher(settings, llm=llm, search=search).start(TOPIC, "t1")
    assert st.status == "done" and st.rounds == 1 and len(st.plan) == 2
    assert llm.calls.count("Notes") == 2  # one worker per sub-question
    assert "[9]" not in st.report and st.citations.invalid == [9]
    assert "## References" in st.report and st.citations.cited == [1, 2]
    assert st.citations.uncited_paragraphs == 1  # the reranking paragraph cites nothing


def test_low_score_triggers_follow_up_research_and_revision(settings, search):
    llm = scripted([WEAK, GOOD])
    st = Researcher(settings, llm=llm, search=search).start(TOPIC, "t2")
    assert st.rounds == 2 and [c.score for c in st.critiques] == [4, 9]
    assert llm.calls.count("Notes") == 3  # 2 planned workers + 1 follow-up worker
    assert any("hallucination" in s.url for s in st.sources)


def test_rounds_are_capped(settings, search):
    st = Researcher(dataclasses.replace(settings, max_rounds=1), llm=scripted([WEAK]), search=search).start(TOPIC, "t3")
    assert st.rounds == 1 and st.status == "done"


def test_plan_approval_pauses_and_resumes_in_a_new_process(settings, search):
    s = dataclasses.replace(settings, approve_plan=True)
    first = Researcher(s, llm=scripted([GOOD]), search=search)
    st = first.start(TOPIC, "t4")
    assert st.status == "awaiting_approval" and st.report == ""
    # a new Researcher with the same SQLite checkpoint file = a restarted process
    second = Researcher(s, llm=scripted([GOOD]), search=search)
    st = second.resume("t4", {"sub_questions": ["What is BM25?"]})
    assert st.status == "done" and [sq.question for sq in st.plan] == ["What is BM25?"]
    assert second.llm.calls.count("Notes") == 1


def test_rejected_plan_stops(settings, search):
    s = dataclasses.replace(settings, approve_plan=True)
    r = Researcher(s, llm=scripted([GOOD]), search=search)
    r.start(TOPIC, "t5")
    st = r.resume("t5", {"approved": False})
    assert st.status == "done" and "cancelled" in st.report and st.critiques == []


def test_helpers():
    assert strip_thinking("<think>hmm\n</think>\nAnswer") == "Answer"
    raw = [{"title": "a", "url": "u1", "provider": "x"}, {"title": "b", "url": "u2", "provider": "x"},
           {"title": "a again", "url": "u1", "provider": "x"}]
    assert {n: s.url for n, s in number_sources(raw).items()} == {1: "u1", 2: "u2"}


def test_thread_ids_are_not_reused(settings, search):
    import pytest

    r = Researcher(settings, llm=scripted([GOOD]), search=search)
    r.start(TOPIC, "t6")
    with pytest.raises(ValueError, match="already exists"):
        r.start(TOPIC, "t6")
