"""Command line.

    research run "topic" [--approve-plan] [--thread ID] [--out report.md]
    research resume ID [--sub-question "..."]...     continue a paused or interrupted run
    research show ID                                 print a stored run
    research eval [--limit N]
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path


def _print(status, out: str | None) -> None:
    if status.status == "awaiting_approval":
        print(f"Plan for thread {status.thread_id} (awaiting approval):")
        for i, sq in enumerate(status.plan, 1):
            print(f"  {i}. {sq.question}   queries: {sq.queries}")
        print(f"\nApprove:  research resume {status.thread_id}\n"
              f"Edit:     research resume {status.thread_id} --sub-question '...' --sub-question '...'")
        return
    print(status.report)
    scores = " -> ".join(str(c.score) for c in status.critiques) or "-"
    c = status.citations
    print(f"\n[thread {status.thread_id} · {status.rounds} critique round(s), scores {scores} · {len(status.sources)} "
          f"sources · {len(c.cited) if c else 0} cited · {status.seconds}s]", file=sys.stderr)
    if out:
        Path(out).write_text(status.report)
        print(f"saved {out}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="research", description="Deep Research Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="research a topic")
    run.add_argument("topic", nargs="+")
    run.add_argument("--approve-plan", action="store_true", help="pause after planning for your approval")
    run.add_argument("--thread", help="thread id (default: random)")
    run.add_argument("--out", help="also save the report to this file")
    res = sub.add_parser("resume", help="continue a paused or interrupted run")
    res.add_argument("thread")
    res.add_argument("--sub-question", action="append", help="replace the plan (repeatable)")
    res.add_argument("--reject", action="store_true", help="cancel at plan review")
    res.add_argument("--out")
    show = sub.add_parser("show", help="print a stored run")
    show.add_argument("thread")
    ev = sub.add_parser("eval", help="agent vs baseline evaluation")
    ev.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    if args.cmd == "eval":
        from .config import ROOT

        sys.path.insert(0, str(ROOT))  # evals/ lives next to src/
        from evals.run_eval import run as run_eval

        run_eval(limit=args.limit)
        return

    from .config import get_settings
    from .graph import Researcher

    settings = get_settings()
    if args.cmd == "run":
        settings = dataclasses.replace(settings, approve_plan=args.approve_plan or settings.approve_plan)
        _print(Researcher(settings).start(" ".join(args.topic), args.thread), args.out)
    elif args.cmd == "resume":
        settings = dataclasses.replace(settings, approve_plan=True)
        decision = {"approved": not args.reject}
        if args.sub_question:
            decision["sub_questions"] = args.sub_question
        _print(Researcher(settings).resume(args.thread, decision), args.out)
    else:
        _print(Researcher(settings).status(args.thread), None)


if __name__ == "__main__":
    main()
