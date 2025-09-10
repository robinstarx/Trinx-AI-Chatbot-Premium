

import logging

from langchain_core.messages import HumanMessage, AIMessage
from src.core.llm_routes import answer_compose_llm
from src.core.state import AgentState


from src.prompts.config.prompt_import import answer_prompt




logger = logging.getLogger(__name__)



def compose_answer_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering answer_node")
        user_q = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        logger.info(f"Answer node query: {user_q}")

        chat_history = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in state["messages"]
        )
        logger.info(f"Chat history: {chat_history}")

        # ✨ ROUTE-SPECIFIC CONTEXT INCLUSION
        # Only include context that's relevant to the current route
        last_route = state.get("previous_route", "")
        parts = []

        if last_route == "fetch_price" and state.get("price"):
            coin_price_ctx = str(state.get("price", "")).strip()
            if coin_price_ctx and coin_price_ctx != "0.0":
                parts.append("Coin Price:\n" + coin_price_ctx)

        elif last_route == "web" and state.get("web_results"):
            web_ctx = state.get("web_results", "").strip()
            if web_ctx:
                parts.append("Web Search:\n" + web_ctx)

        elif last_route == "trinity_coin_details" and state.get("trinity_info"):
            trinity_ctx = state.get("trinity_info", "").strip()
            if trinity_ctx:
                parts.append("Trinity Coin AI:\n" + trinity_ctx)

        context = "\n\n".join(parts) if parts else "No external context available."
        logger.info(f"Answer node context prepared for route: {last_route}")

        answer = answer_compose_llm.invoke(
            [
                HumanMessage(
                    content=answer_prompt["answer_node_v1"]["system"].format(
                        context=context, user_q=user_q
                    )
                )
            ]
        ).content

        logger.info("Generated answer.")
        logger.info("Exiting answer_node")

        return {**state, "messages": state["messages"] + [AIMessage(content=answer)]}
    except Exception as e:
        logger.error(f"Error in answer_node: {e}")
        raise
