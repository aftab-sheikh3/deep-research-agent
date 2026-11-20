"""Structured outputs of the agent's LLM steps, and the API models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class SubQuestion(BaseModel):
    question: str = Field(description="A focused sub-question that the report must answer")
    queries: list[str] = Field(description="1-2 short search-engine queries for this sub-question")


class Plan(BaseModel):
    """A research plan: the sub-questions that together answer the topic."""

    sub_questions: list[SubQuestion] = Field(description="3-5 complementary sub-questions, no overlap")


class Finding(BaseModel):
    claim: str = Field(description="One factual statement taken from the sources, with numbers/dates if given")
    sources: list[int] = Field(description="Numbers of the search results that state this claim")


class Notes(BaseModel):
    """Facts extracted from search results for one sub-question."""

    findings: list[Finding] = Field(default_factory=list, description="Up to 6 findings relevant to the question")


class Critique(BaseModel):
    """A reviewer's assessment of a draft research report."""

    score: int = Field(ge=1, le=10, description="Overall quality 1-10: coverage, accuracy, citation support, clarity")
    missing: list[str] = Field(default_factory=list, description="Important aspects the report does not cover")
    unsupported: list[str] = Field(default_factory=list, description="Statements not backed by a cited source")
    follow_up_queries: list[str] = Field(default_factory=list, description="Up to 3 searches that would fill the gaps")


class Source(BaseModel):
    n: int
    title: str
    url: str
    provider: str
    content: str = ""


class CitationReport(BaseModel):
    cited: list[int] = []
    invalid: list[int] = []          # [n] with no such source (removed from the report)
    unused: list[int] = []           # sources never cited
    uncited_paragraphs: int = 0      # body paragraphs without any citation
    paragraphs: int = 0

    @computed_field
    @property
    def valid_rate(self) -> float:
        total = len(self.cited) + len(self.invalid)
        return len(self.cited) / total if total else 0.0


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=5, max_length=500, examples=["How do hybrid search and reranking improve RAG?"])
    approve_plan: bool = False


class ResearchStatus(BaseModel):
    thread_id: str
    status: Literal["running", "awaiting_approval", "done", "error"]
    topic: str = ""
    plan: list[SubQuestion] = []
    report: str = ""
    sources: list[Source] = []
    critiques: list[Critique] = []
    citations: CitationReport | None = None
    rounds: int = 0
    seconds: float = 0.0
    error: str = ""
