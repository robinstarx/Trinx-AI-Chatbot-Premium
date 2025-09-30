import streamlit as st
from datetime import datetime, timedelta
from uuid import uuid4
import os
from langchain_community.document_loaders import PyPDFLoader
import logging
from src.rag.rag_store import add_docs
from src.rag.rag_store import vector_store
from src.agent.graph import graph_agent
from langchain_core.messages import AIMessage, HumanMessage

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
        type=["pdf", "txt", "docx"], 
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
    save_dir = "uploaded_files"
    os.makedirs(save_dir, exist_ok=True)
    
    # Check for new files that haven't been processed yet
    current_file_names = [f.name for f in uploaded_files]
    processed_file_names = [info['filename'] for info in st.session_state.uploaded_files_info]
    
    for file in uploaded_files:
        if file.name not in processed_file_names:
            try:
                save_path = os.path.join(save_dir, file.name)
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())

                doc_upload = add_docs(
                    save_path,
                    vector_store,
                    {"source": "upload", "filename": file.name, "user_id": st.session_state.user_id}
                )
                
                if doc_upload:
                    # Add to session state
                    st.session_state.uploaded_files_info.append({
                        "filename": file.name,
                        "upload_time": now.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.session_state.pending_file_query = True
                    st.success(f"✅ File '{file.name}' uploaded and indexed successfully!")
                    logger.info(f"Successfully uploaded and indexed: {file.name}")
                else:
                    st.warning(f"⚠️ No content extracted from {file.name}")
                    logger.warning(f"No docs added from {file.name}")
                    
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")
                logger.error(f"Error processing {file.name}: {e}")

if user_query:
    user_msg = HumanMessage(content=user_query)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append(user_msg)

    # ─── Determine Query Type ─────────────────
    pending_file_query = st.session_state.get("pending_file_query", False)
    has_uploaded_files = len(st.session_state.uploaded_files_info) > 0

    if pending_file_query:
        is_file_upload = True
        logger.info("Pending file query detected; routing to file_upload_qa")
    else:
        is_file_upload = False
        logger.info("No pending file query; using normal routing")
    
    logger.info(f"Query: {user_query}")
    logger.info(f"Has uploaded files: {has_uploaded_files}")
    logger.info(f"Final is_file_upload flag: {is_file_upload}")

    # ─── Run Graph Agent ────────────────────
    try:
        logger.info("Invoking graph agent")
        result = graph_agent.invoke(
            {"messages": [user_msg]},
            config={"configurable": {
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id,
                "is_file_upload": is_file_upload,
            }},
        )
        logger.info("Graph agent invocation successful")

        if pending_file_query:
            st.session_state.pending_file_query = False
            logger.info("Pending file query flag cleared")

        ai_msg = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None
        )
        if ai_msg:
            with st.chat_message("assistant"):
                st.markdown(ai_msg.content)
            st.session_state.messages.append(ai_msg)
            logger.info(f"AI response generated successfully")
        else:
            with st.chat_message("assistant"):
                st.markdown("⚠️ No response generated.")
            logger.warning("No response generated by the AI.")
            
    except Exception as e:
        logger.error(f"Error invoking graph agent: {e}")
        with st.chat_message("assistant"):
            st.markdown(f"❌ Error: `{e}`")
