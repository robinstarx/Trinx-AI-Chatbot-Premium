
import logging

from langchain_core.messages import HumanMessage

from src.core.state import     AgentState


from src.tools.serper_web_search import web_search_tool


logger = logging.getLogger(__name__)


def web_search_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering web_node")
        query = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        logger.info(f"Web search query: {query}")

        result = web_search_tool.invoke({"query": query})
        logger.info(f"Web search returned content length: {len(result)}")

        logger.info("Exiting web_node")
        return {
            **state,
            "web_results": result,
            "route": "answer",
            "previous_route": "web",
        }
    except Exception as e:
        logger.error(f"Error in web_node: {e}")
        raise
