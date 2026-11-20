from deep_research.llm import ScriptedChatModel
from evals.run_eval import load_topics, prefer, run
from tests.conftest import scripted


def judge(winners):
    return ScriptedChatModel(structured={"Coverage": {"covered": [True, True, False, False, False]},
                                         "Preference": [{"winner": w} for w in winners]})


def test_topics_are_valid():
    topics = load_topics()
    assert len(topics) >= 5 and all(len(t["key_points"]) == 5 for t in topics)


def test_preference_needs_both_orders_to_agree():
    assert prefer(judge(["A", "B"]), "t", "agent", "base") == "agent"
    assert prefer(judge(["B", "A"]), "t", "agent", "base") == "baseline"
    assert prefer(judge(["A", "A"]), "t", "agent", "base") == "tie"  # position bias -> tie


def test_eval_runs_end_to_end(settings, search):
    good = {"score": 9, "missing": [], "unsupported": [], "follow_up_queries": []}
    out = run(limit=1, settings=settings, llm=scripted([good], drafts=["Report [1]."] * 4),
              judge=judge(["A", "B"]), search=search)
    s = out["summary"]
    assert s["agent"]["coverage"] == 0.4 and s["wins"]["agent"] == 1
    assert s["agent"]["valid_citation_rate"] == 1.0
