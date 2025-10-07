import streamlit as st
from datetime import datetime, timedelta
from uuid import uuid4
import os
import base64
import logging
from typing import Any, Dict

import requests
from langchain_core.messages import AIMessage, HumanMessage

# API Configuration
API_BASE_URL = os.getenv("TRINX_API_BASE_URL", "http://127.0.0.1:8008")


# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Config ───────────────────────────
st.set_page_config(page_title="TrinX AI Chatbot", layout="centered")
st.title("🤖 TrinX AI Chatbot")

# ─── Session Management ────────────────────
SESSION_TIMEOUT = timedelta(minutes=10)
now = datetime.now()

if "thread_id" not in st.session_state or "last_active" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
    st.session_state.messages = []
    st.session_state.last_active = now
    st.session_state.uploaded_files_info = []  # Track uploaded files
    logger.info(f"New session started with thread_id: {st.session_state.thread_id}")
elif now - st.session_state.last_active > SESSION_TIMEOUT:
    logger.warning("Session timed out. Creating new session.")
    st.session_state.thread_id = str(uuid4())
    st.session_state.messages = []
    st.session_state.uploaded_files_info = []
    st.session_state.last_active = now
    logger.info(f"New session started with thread_id: {st.session_state.thread_id}")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid4())
    logger.info(f"New user_id assigned: {st.session_state.user_id}")

if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = []

if "pending_file_query" not in st.session_state:
    st.session_state.pending_file_query = False

st.session_state.last_active = now

# ─── Sidebar File Upload ───────────────────
with st.sidebar:
    st.header("📁 File Upload")
    uploaded_files = st.file_uploader(
        "Upload documents", 
        type=["pdf", "txt", "docx", "md"], 
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX files to ask questions about their content"
    )
    
    # Display uploaded files info
    if st.session_state.uploaded_files_info:
        st.subheader("📄 Uploaded Files")
        for file_info in st.session_state.uploaded_files_info:
            st.write(f"✅ {file_info['filename']}")
        
        if st.button("🗑️ Clear All Files"):
            st.session_state.uploaded_files_info = []
            st.session_state.pending_file_query = False
            st.rerun()
    
    st.divider()
    st.markdown("### 💡 How it works:")
    st.markdown("""
    **📁 File Mode (when files uploaded):**
    - All queries automatically analyze your uploaded files
    - Ask anything about the content: "Analyze this", "What role is best?", "Summarize"
    
    **🔍 Override file mode:**
    - "What is Trinity coin?" → Trinity info
    - "Bitcoin price?" → Live price data
    - "Latest crypto news?" → Web search
    
    **🎯 Normal Mode (no files):**
    - Trinity questions, crypto prices, general knowledge
    """)
    
    st.markdown("### 🎯 Example queries:")
    st.markdown("""
    **With files uploaded:**
    - "Analyze this resume"
    - "What role is best for this candidate?"
    - "Summarize the document"
    
    **Without files:**
    - "What is Trinity AI coin?"
    - "Bitcoin price now?"
    - "Latest crypto news"
    """)

# ─── File Status Display ───────────────────
if st.session_state.uploaded_files_info:
    with st.container():
        st.success(
            "📁 **Files Ready** - {} file(s) uploaded. Your next question will analyze them."
            .format(len(st.session_state.uploaded_files_info))
        )

# ─── Display Chat History ──────────────────
for msg in st.session_state.messages:
    role = "assistant" if isinstance(msg, AIMessage) else "user"       
    with st.chat_message(role):
        st.markdown(msg.content)

# ─── Chat Input ─────────────────────────────
user_query = st.chat_input(
    placeholder="Ask about Trinity AI or your uploaded documents…",
)

# ─── Handle File Upload (Process files when uploaded) ─────────────────
if uploaded_files:
    current_file_names = [f.name for f in uploaded_files]
    processed_file_names = [info['filename'] for info in st.session_state.uploaded_files_info]
    
    for file in uploaded_files:
        if file.name not in processed_file_names:
            try:
                encoded_file = base64.b64encode(file.getvalue()).decode()
                payload = {
                    "session_id": st.session_state.thread_id,
                    "user_id": st.session_state.user_id,
                    "filename": file.name,
                    "file_base64": encoded_file,
                    "content_type": file.type,
                }
                response = requests.post(
                    f"{API_BASE_URL}/api/chat-premium/upload-file",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()

                st.session_state.uploaded_files_info.append({
                    "filename": file.name,
                    "upload_time": now.strftime("%Y-%m-%d %H:%M:%S")
                })
                st.session_state.pending_file_query = True
                st.success(f"✅ File '{file.name}' uploaded and indexed successfully!")
                logger.info(f"Successfully uploaded and indexed via API: {file.name}")

            except requests.HTTPError as http_err:
                st.error(f"❌ API error processing {file.name}: {http_err.response.text}")
                logger.error(f"HTTP error processing {file.name}: {http_err}")
            except Exception as e:
                st.error(f"❌ Unexpected error processing {file.name}: {str(e)}")
                logger.error(f"Unexpected error processing {file.name}: {e}")

if user_query:
    user_msg = HumanMessage(content=user_query)
    
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append(user_msg)

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat-premium",
            json={
                "prompt": user_query,
                "session_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id,
                "is_file_upload": st.session_state.pending_file_query,
            },
            timeout=60,
        )
        response.raise_for_status()
        data: Dict[str, Any] = response.json()

        if st.session_state.pending_file_query:
            st.session_state.pending_file_query = False

        ai_content = data.get("response", "")
        ai_msg = AIMessage(content=ai_content)

        with st.chat_message("assistant"):
            st.markdown(ai_content)
        st.session_state.messages.append(ai_msg)
        logger.info("Chat response received from API")

    except requests.HTTPError as http_err:
        error_text = http_err.response.text if http_err.response else str(http_err)
        logger.error(f"HTTP error during chat: {error_text}")
        with st.chat_message("assistant"):
            st.markdown(f"❌ API Error: `{error_text}`")
    except Exception as e:
        logger.error(f"Unexpected error during chat: {e}")
        with st.chat_message("assistant"):
            st.markdown(f"❌ Error: `{e}`")
