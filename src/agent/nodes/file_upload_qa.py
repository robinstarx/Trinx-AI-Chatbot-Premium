import logging
from langchain_core.messages import HumanMessage
from src.core.state import AgentState
from langgraph.types import RunnableConfig
from src.rag.rag_store import rag_search_tool

logger = logging.getLogger(__name__)

def file_upload_qa_node(state: AgentState, config: RunnableConfig) -> AgentState:
    try:
        logger.info("Entering file_upload_qa_node")

        query = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )
        logger.info(f"File upload QA query: {query}")

        user_id = state.get("user_id")

        rag_result = rag_search_tool.invoke(
            {
                "query": query,
                "source": "upload",
                "user_id": user_id,
                "session_id": config["configurable"].get("thread_id"),
            },
        )
        logger.info(f"RAG result length: {len(rag_result)} chars")

        return {
            **state,
            "file_upload_cxt": rag_result,
            "route": "answer",
            "previous_route": "file_upload_qa",
        }
    except Exception as e:
        logger.error(f"Error in file_upload_qa_node: {e}")
        raise
