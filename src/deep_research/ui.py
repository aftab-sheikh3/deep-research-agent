"""Streamlit front end (talks to the API):  streamlit run src/deep_research/ui.py"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Deep Research Agent", page_icon=":material/travel_explore:", layout="wide")
st.title("Deep Research Agent")

with st.form("start"):
    topic = st.text_input("Research topic", placeholder="How do hybrid search and reranking improve RAG?")
    approve = st.checkbox("Let me review the plan first")
    if st.form_submit_button("Research", type="primary") and topic:
        r = requests.post(f"{API}/research", json={"topic": topic, "approve_plan": approve}, timeout=30)
        st.session_state.thread = r.json()["thread_id"] if r.ok else None
        if not r.ok:
            st.error(r.text)

thread = st.session_state.get("thread")
if thread:
    status = requests.get(f"{API}/research/{thread}", timeout=30).json()
    st.caption(f"thread `{thread}` · {status['status']}")
    if status["status"] == "awaiting_approval":
        st.subheader("Proposed plan")
        edited = st.text_area("One sub-question per line", "\n".join(sq["question"] for sq in status["plan"]))
        if st.button("Approve and research", type="primary"):
            lines = [line for line in edited.splitlines() if line.strip()]
            requests.post(f"{API}/research/{thread}/approve", json={"approved": True, "sub_questions": lines}, timeout=30)
            st.rerun()
    elif status["status"] == "running":
        with st.spinner("Planning, searching, writing, critiquing..."):
            time.sleep(3)
        st.rerun()
    elif status["status"] == "error":
        st.error(status["error"])
    else:
        left, right = st.columns([3, 1])
        left.markdown(status["report"])
        with right:
            c = status.get("citations") or {}
            st.metric("Sources cited", f"{len(c.get('cited', []))} / {len(status['sources'])}")
            st.metric("Critique rounds", status["rounds"])
            for i, crit in enumerate(status["critiques"], 1):
                st.write(f"Round {i}: score **{crit['score']}/10**")
                if crit["missing"]:
                    st.caption("Missing: " + "; ".join(crit["missing"]))
            st.download_button("Download report.md", status["report"], file_name="report.md")
