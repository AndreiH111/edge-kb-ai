"""Lightweight Streamlit UI for manually smoke-testing the edge-kb-ai backend.

Run the API first (uvicorn main:app), then: streamlit run frontend/streamlit_app.py
"""
import json
from uuid import uuid4

import httpx
import streamlit as st

st.set_page_config(page_title="edge-kb-ai tester", layout="wide")

base = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000").rstrip("/") + "/ai"
client = httpx.Client(base_url=base, timeout=300)


def call(method: str, path: str, **kwargs):
    """Make a request and show any failure inline; returns parsed JSON or None."""
    try:
        resp = client.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"{exc.response.status_code}: {exc.response.text}")
    except httpx.HTTPError as exc:
        st.error(f"Request failed: {exc}")
    return None


def stream_answer(question: str, session_id: str, tools: list):
    """Yield answer tokens from POST /chat/stream (SSE), collecting tool names."""
    event = None
    with client.stream("POST", "/chat/stream",
                       json={"question": question, "session_id": session_id}) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "token":
                    yield data["text"]
                elif event == "tool":
                    tools.append(data["name"])
                elif event == "error":
                    yield f"\n\n**Error:** {data['message']}"


# --- Sidebar: health / agent status ---
if st.sidebar.button("Check agent status"):
    if (status := call("GET", "/agent-status")) is not None:
        st.sidebar.success("Backend reachable")
        st.sidebar.json(status)

chat_tab, files_tab, jobs_tab = st.tabs(["Chat", "Files", "Jobs"])

# --- Chat ---
with chat_tab:
    st.session_state.setdefault("session_id", str(uuid4()))
    st.session_state.setdefault("messages", [])
    col1, col2 = st.columns([3, 1])
    use_stream = col1.toggle("Stream (SSE)", value=True)
    if col2.button("New session"):
        st.session_state.session_id, st.session_state.messages = str(uuid4()), []
    st.caption(f"Session: `{st.session_state.session_id}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools"):
                st.caption("Tools: " + ", ".join(msg["tools"]))

    if question := st.chat_input("Ask about your documents or Jira…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            tools, answer = [], ""
            if use_stream:
                try:
                    answer = st.write_stream(
                        stream_answer(question, st.session_state.session_id, tools))
                except httpx.HTTPError as exc:
                    st.error(f"Stream failed: {exc}")
            elif (res := call("POST", "/chat", json={
                    "question": question, "session_id": st.session_state.session_id})):
                answer, tools = res["answer"], res.get("tools_used") or []
                st.markdown(answer)
            if tools:
                st.caption("Tools: " + ", ".join(tools))
        if answer:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "tools": tools})

# --- Files: list / upload / delete / reindex ---
with files_tab:
    with st.form("upload", clear_on_submit=True):
        up = st.file_uploader("Upload (Excel / PDF / PowerPoint)",
                              type=["xlsx", "xls", "xlsm", "pdf", "pptx", "pptm"])
        who = st.text_input("Uploaded by (optional)")
        if st.form_submit_button("Upload") and up:
            if res := call("POST", "/upload", files={"file": (up.name, up.getvalue())},
                           data={"uploaded_by": who} if who else None):
                st.success(f"Queued job {res['job_id']} for {res['filename']}")

    listing = call("GET", "/files")
    if listing:
        st.caption(f"Storage: {listing['destination']} — {listing['location']}")
        st.dataframe(listing["files"], width="stretch")
        names = [f["filename"] for f in listing["files"]]
        to_delete = st.multiselect("Delete files", names)
        if st.button("Delete selected", disabled=not to_delete, type="primary"):
            if res := call("DELETE", "/files", json={"filenames": to_delete}):
                st.success(res)
                st.rerun()

    if st.button("Reindex all") and (res := call("POST", "/reindex")):
        st.success(f"Queued reindex job {res['job_id']}")

# --- Jobs ---
with jobs_tab:
    st.button("Refresh")
    if (jobs := call("GET", "/jobs")) is not None:
        st.dataframe(jobs["jobs"], width="stretch")
