import logging
from langgraph.types import RunnableConfig
from langchain_core.messages import HumanMessage
from src.core.llm_routes import query_router_llm
from src.core.state import AgentState, QueryDecision
from src.core.state_manager import clear_context_fields
from src.prompts.config.prompt_import import router_prompt

logger = logging.getLogger(__name__)

def query_router_node(state: AgentState, config: RunnableConfig) -> AgentState:
    try:
        logger.info("Entering query_router_node")
        user_msgs = [
            m for m in state.get("messages", []) if isinstance(m, HumanMessage)
        ]
        query = user_msgs[-1].content if user_msgs else ""
        logger.info(f"Router query: {query}")
        state["is_file_upload"] = config["configurable"].get("is_file_upload", False)
        cleared_state = clear_context_fields(state)
        new_msgs = cleared_state.get("messages", []) + [HumanMessage(content=query)]

        # ---- File Priority Routing Logic ----
        # If is_file_upload is True, directly route to file_upload_qa
        if state.get("is_file_upload"):
            logger.info("File priority routing: Directly routing to file_upload_qa")
            response: AgentState = {
                **cleared_state,
                "messages": new_msgs,
                "route": "file_upload_qa",
                "user_id": config["configurable"].get("user_id"),
            }
            return response

        # Otherwise, use LLM routing for normal queries
        decision: QueryDecision = query_router_llm.invoke(
            [("system", router_prompt["router_v1"]["system"]), ("user", query)]
        )
        logger.info(f"Router decision: {decision.route}")

        response: AgentState = {
            **cleared_state,
            "messages": new_msgs,
            "route": decision.route,
        }
        logger.info("Exiting query_router_node")
        return response

    except Exception as e:
        logger.error(f"Error in query_router_node: {e}")
        raise
