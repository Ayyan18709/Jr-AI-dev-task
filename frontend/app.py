import streamlit as st
import requests
import json
import uuid
import os

# --- Page Config ---
st.set_page_config(
    page_title="Production RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Custom CSS for aesthetic improvements ---
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ RAG Settings")
    st.markdown("---")
    
    # 1. System Health Check
    st.subheader("System Status")
    try:
        health_res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health_res.status_code == 200:
            health_data = health_res.json()
            st.success("🟢 API Online")
            st.caption(f"LLM Loaded: **{health_data.get('llm_loaded', False)}**")
            st.caption(f"Embedder Loaded: **{health_data.get('embedder_loaded', False)}**")
            st.caption(f"Chunks Indexed: **{health_data.get('chunks_indexed', 0)}**")
        else:
            st.error("🔴 API Offline")
    except requests.exceptions.ConnectionError:
        st.error("🔴 API Unreachable")
    
    st.markdown("---")
    
    # 2. Document Ingestion
    st.subheader("Upload Document (CV)")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], help="Upload a document to index it for RAG mode.")
    if st.button("Index Document", type="primary"):
        if uploaded_file is not None:
            with st.spinner("Indexing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    res = requests.post(f"{API_BASE_URL}/upload_cv", files=files)
                    if res.status_code == 200:
                        st.success(f"Successfully indexed! ({res.json().get('chunks_indexed', 0)} chunks)")
                    else:
                        st.error(f"Failed to index: {res.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please select a file first.")
            
    st.markdown("---")
    
    # 3. Chat Settings
    st.subheader("Chat Configuration")
    mode = st.radio("Mode", options=["chat", "rag"], index=1, format_func=lambda x: "General Chat" if x == "chat" else "RAG Mode", help="RAG Mode uses the uploaded document for context.")
    top_k = st.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- Main Content ---
st.title("🤖 Production RAG Chatbot")
st.caption(f"Powered by **Qwen2.5-1.5B-Instruct** | Session ID: `{st.session_state.session_id}`")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "meta" in message:
            st.caption(f"⏱️ Generation Latency: {message['meta'].get('latency_ms', 0):.2f} ms")
            if message['meta'].get('memory_interactions', 0) > 0:
                st.caption(f"🧠 Context Memory Used: {message['meta'].get('memory_interactions')} prior interactions")

# React to user input
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare payload for API
    payload = {
        "query": prompt,
        "session_id": st.session_state.session_id,
        "mode": mode,
        "top_k": top_k
    }

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                res = requests.post(f"{API_BASE_URL}/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer provided.")
                    latency = data.get("latency_ms", 0)
                    memory_interactions = len(data.get("memory_state", {}).get("interactions", []))
                    
                    message_placeholder.markdown(answer)
                    st.caption(f"⏱️ Generation Latency: {latency:.2f} ms")
                    
                    if memory_interactions > 0:
                        st.caption(f"🧠 Context Memory Used: {memory_interactions} prior interactions")
                    
                    # Store assistant message in history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "meta": {
                            "latency_ms": latency,
                            "memory_interactions": memory_interactions
                        }
                    })
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend API. Please make sure the FastAPI server is running.")
