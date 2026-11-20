"""HTTP API. Research runs take minutes, so they run in the background and are polled.

    uvicorn deep_research.api:app --reload              # docs at http://127.0.0.1:8000/docs
    POST /research {"topic": "...", "approve_plan": false}  -> {"thread_id": ...}
    GET  /research/{thread_id}                              -> status, plan, report, sources, critiques
    POST /research/{thread_id}/approve {"approved": true}   -> continue a run paused for plan review
"""
from __future__ import annotations

import dataclasses
import threading
import uuid
from functools import lru_cache

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from . import __version__
from .config import get_settings
from .graph import Researcher
from .schemas import ResearchRequest, ResearchStatus

app = FastAPI(title="Deep Research Agent", version=__version__,
              description="Plans, searches, writes a cited report, critiques and revises it, checks citations.")
_errors: dict[str, str] = {}
_running: set[str] = set()
_lock = threading.Lock()


@lru_cache(maxsize=2)
def researcher(approve_plan: bool = False) -> Researcher:
    settings = dataclasses.replace(get_settings(), approve_plan=approve_plan)
    return Researcher(settings)


class Approval(BaseModel):
    approved: bool = True
    sub_questions: list[str] | None = None


def _run(fn, thread_id: str, *args) -> None:
    with _lock:
        _running.add(thread_id)
    try:
        fn(*args)
    except Exception as e:  # surfaced through GET /research/{id}
        _errors[thread_id] = f"{type(e).__name__}: {e}"
    finally:
        with _lock:
            _running.discard(thread_id)


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "provider": s.provider, "model": s.model_name, "search": s.search}


@app.post("/research", response_model=ResearchStatus, status_code=202)
def start(request: ResearchRequest, background: BackgroundTasks) -> ResearchStatus:
    thread_id = uuid.uuid4().hex[:12]
    r = researcher(request.approve_plan)
    background.add_task(_run, r.start, thread_id, request.topic, thread_id)
    return ResearchStatus(thread_id=thread_id, status="running", topic=request.topic)


def _find(thread_id: str) -> Researcher:
    for approve in (False, True):
        r = researcher(approve)
        if r.graph.get_state(r._config(thread_id)).values:
            return r
    raise HTTPException(404, f"unknown research thread {thread_id!r}")


@app.get("/research/{thread_id}", response_model=ResearchStatus)
def get(thread_id: str) -> ResearchStatus:
    if thread_id in _errors:
        return ResearchStatus(thread_id=thread_id, status="error", error=_errors[thread_id])
    try:
        status = _find(thread_id).status(thread_id)
    except HTTPException:
        if thread_id in _running:  # started, no checkpoint written yet
            return ResearchStatus(thread_id=thread_id, status="running")
        raise
    if thread_id in _running and status.status == "done":
        status.status = "running"
    return status


@app.post("/research/{thread_id}/approve", response_model=ResearchStatus, status_code=202)
def approve(thread_id: str, approval: Approval, background: BackgroundTasks) -> ResearchStatus:
    r = _find(thread_id)
    status = r.status(thread_id)
    if status.status != "awaiting_approval":
        raise HTTPException(409, f"thread is {status.status}, not awaiting approval")
    background.add_task(_run, r.resume, thread_id, thread_id, approval.model_dump(exclude_none=True))
    status.status = "running"
    return status
