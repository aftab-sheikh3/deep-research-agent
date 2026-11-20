"""Deterministic citation checks on a markdown report with [n] references.

- invalid references (no such source) are removed from the text and reported;
- every body paragraph should cite something (headings, lists of references and short
  connective lines are exempt);
- the References section is (re)generated from the sources actually cited, numbered as cited.
"""
from __future__ import annotations

import re

from .schemas import CitationReport, Source

REF = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _numbers(match: str) -> list[int]:
    return [int(x) for x in re.split(r"\s*,\s*", match)]


def strip_references_section(report: str) -> str:
    return re.split(r"\n#{1,3}\s*(References|Sources)\s*\n", report, maxsplit=1, flags=re.I)[0].rstrip()


def check(report: str, sources: dict[int, Source]) -> tuple[str, CitationReport]:
    body = strip_references_section(report)
    cited: set[int] = set()
    invalid: set[int] = set()

    def fix(m: re.Match) -> str:
        nums = _numbers(m.group(1))
        good = [n for n in nums if n in sources]
        invalid.update(n for n in nums if n not in sources)
        cited.update(good)
        return "".join(f"[{n}]" for n in good)

    body = REF.sub(fix, body)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    content = [p for p in paragraphs if not p.startswith("#") and len(p.split()) >= 12]
    uncited = sum(1 for p in content if not REF.search(p))
    refs = "\n".join(f"[{n}] {sources[n].title}. {sources[n].url}" for n in sorted(cited))
    final = f"{body}\n\n## References\n\n{refs}\n" if refs else body + "\n"
    return final, CitationReport(cited=sorted(cited), invalid=sorted(invalid),
                                 unused=sorted(set(sources) - cited), uncited_paragraphs=uncited,
                                 paragraphs=len(content))
