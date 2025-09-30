
import logging

from langchain_core.messages import HumanMessage
from src.core.state import AgentState
from src.rag.rag_store import rag_search_tool




logger = logging.getLogger(__name__)



def get_trinity_details_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering trinity_coin_deatils_node")
        query = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        logger.info(f"Trinity coin details query: {query}")

        trinity_answer = rag_search_tool.invoke({"query": query, "source": "trinity"})
        logger.info("Generated answer.")
        logger.info(f"Answer: {trinity_answer}")
        logger.info("Exiting trinity_coin_deatils_node")
        return {
            **state,
            "trinity_info": trinity_answer,
            "route": "answer",
            "previous_route": "trinity_coin_details",
        }
    except Exception as e:
        logger.error(f"Error in trinity_coin_deatils_node: {e}")
        raise