import os
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Support Agent", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------- State ----
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def response_error(response, fallback):
    try:
        payload = response.json()
    except ValueError:
        return f"{fallback} (backend returned HTTP {response.status_code})"
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return fallback


# ------------------------------------------------------------- Auth UI ----

def login_page():
    st.title("🤖 AI Customer Support Agent")
    st.caption("Chat with your documents, powered by RAG.")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                r = requests.post(
                    f"{BACKEND_URL}/login",
                    data={"username": username, "password": password},
                )
                if r.status_code == 200:
                    st.session_state.token = r.json()["access_token"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error(response_error(r, "Login failed"))

    with tab_register:
        with st.form("register_form"):
            username = st.text_input("Choose a username")
            password = st.text_input("Choose a password", type="password")
            submitted = st.form_submit_button("Register")
            if submitted:
                r = requests.post(
                    f"{BACKEND_URL}/register",
                    json={"username": username, "password": password},
                )
                if r.status_code == 200:
                    st.session_state.token = r.json()["access_token"]
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error(response_error(r, "Registration failed"))


# --------------------------------------------------------------- Helpers ----

def load_sessions():
    r = requests.get(f"{BACKEND_URL}/sessions", headers=auth_headers())
    return r.json() if r.status_code == 200 else []


def load_messages(session_id):
    r = requests.get(f"{BACKEND_URL}/sessions/{session_id}/messages", headers=auth_headers())
    return r.json() if r.status_code == 200 else []


def load_documents():
    r = requests.get(f"{BACKEND_URL}/documents", headers=auth_headers())
    return r.json() if r.status_code == 200 else []


# ---------------------------------------------------------------- Main ----

def main_app():
    with st.sidebar:
        st.markdown(f"### 👋 {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("📄 Documents")
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded is not None:
            if st.button("Ingest document"):
                with st.spinner("Processing PDF..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                    r = requests.post(f"{BACKEND_URL}/upload", headers=auth_headers(), files=files)
                if r.status_code == 200:
                    st.success(f"Added {r.json()['num_chunks']} chunks from {uploaded.name}")
                else:
                    st.error(r.json().get("detail", "Upload failed"))

        docs = load_documents()
        for d in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {d['filename']}  \n_{d['num_chunks']} chunks_")
            if col2.button("🗑️", key=f"del_{d['id']}"):
                requests.delete(f"{BACKEND_URL}/documents/{d['id']}", headers=auth_headers())
                st.rerun()

        st.divider()
        st.subheader("💬 Chat History")
        if st.button("➕ New Chat"):
            st.session_state.current_session_id = None
            st.session_state.messages = []
            st.rerun()

        sessions = load_sessions()
        for s in sessions:
            label = s["title"] or "New Chat"
            if st.button(label, key=f"session_{s['id']}", use_container_width=True):
                st.session_state.current_session_id = s["id"]
                st.session_state.messages = load_messages(s["id"])
                st.rerun()

    st.title("🤖 AI Customer Support Agent")

    if not docs:
        st.info("Upload a PDF in the sidebar to start chatting with your documents.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        page_info = f" (page {s['page']})" if s.get("page") else ""
                        st.markdown(f"**{s['source']}{page_info}**")
                        st.caption(s["snippet"])

    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query, "sources": None})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                r = requests.post(
                    f"{BACKEND_URL}/chat",
                    headers=auth_headers(),
                    json={"session_id": st.session_state.current_session_id, "query": query},
                )
            if r.status_code == 200:
                data = r.json()
                st.session_state.current_session_id = data["session_id"]
                st.markdown(data["answer"])
                if data["sources"]:
                    with st.expander("📚 Sources"):
                        for s in data["sources"]:
                            page_info = f" (page {s['page']})" if s.get("page") else ""
                            st.markdown(f"**{s['source']}{page_info}**")
                            st.caption(s["snippet"])
                st.session_state.messages.append({
                    "role": "assistant", "content": data["answer"], "sources": data["sources"]
                })
            else:
                st.error(r.json().get("detail", "Something went wrong"))


if st.session_state.token is None:
    login_page()
else:
    main_app()
