
import logging

from langchain_core.messages import HumanMessage
from src.core.llm_routes import query_router_llm
from src.core.state import AgentState, QueryDecision
    

from src.core.state_manager import clear_context_fields
from src.prompts.config.prompt_import import router_prompt



logger = logging.getLogger(__name__)

def query_router_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering route_intent_node")
        user_msgs = [
            m for m in state.get("messages", []) if isinstance(m, HumanMessage)
        ]
        query = user_msgs[-1].content if user_msgs else ""
        logger.info(f"Router query: {query}")

        cleared_state = clear_context_fields(state)

        decision: QueryDecision = query_router_llm.invoke(
            [("system", router_prompt["router_v1"]["system"]), ("user", query)]
        )
        logger.info(f"Router decision: {decision.route}")

        new_msgs = cleared_state.get("messages", []) + [HumanMessage(content=query)]
        response: AgentState = {
            **cleared_state,
            "messages": new_msgs,
            "route": decision.route,
        }
        logger.info("Exiting router_node")
        return response
    except Exception as e:
        logger.error(f"Error in router_node: {e}")
        raise