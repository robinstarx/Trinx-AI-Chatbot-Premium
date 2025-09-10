

import logging

from langchain_core.messages import HumanMessage
from src.core.llm_routes import trinity_coin_details_llm
from src.core.state import AgentState

from src.prompts.config.prompt_import import trinity_info_prompt
from src.core.state import TRINITY_KNOWLEDGE_BASE




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

        trinity_answer = trinity_coin_details_llm.invoke(
            [
                (
                    "system",
                    trinity_info_prompt["trinity_info_node_v1"]["system"].format(
                        TRINITY_KNOWLEDGE_BASE=TRINITY_KNOWLEDGE_BASE
                    ),
                ),
                ("user", query),
            ]
        )
        logger.info("Generated answer.")
        logger.info(f"Answer: {trinity_answer.content}")
        logger.info("Exiting trinity_coin_deatils_node")
        return {
            **state,
            "trinity_info": trinity_answer.content,
            "route": "answer",
            "previous_route": "trinity_coin_details",
        }
    except Exception as e:
        logger.error(f"Error in trinity_coin_deatils_node: {e}")
        raise