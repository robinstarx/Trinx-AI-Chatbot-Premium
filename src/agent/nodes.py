import logging

from langchain_core.messages import HumanMessage, AIMessage
from src.core.llm_routes import (
    query_router_llm,
    coin_symbol_converter_llm,
    trinity_coin_details_llm,
    answer_compose_llm,
)
from src.core.state import (
    AgentState,
    QueryDecision,
    FetchCoinPriceDecision,
    TRINITY_KNOWLEDGE_BASE,
)

# from src.tools.web_search import web_search_tool # not used
from src.tools.fetch_coin_price import fetch_coin_price_tool
from src.tools.serper_web_search import web_search_tool
from src.core.state_manager import clear_context_fields
from src.prompts.prompt_import import (
    router_prompt,
    fetch_price_prompt,
    trinity_info_prompt,
    answer_prompt,
)


logger = logging.getLogger(__name__)


def query_intent_router(state: AgentState) -> AgentState:
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

        if decision.route == "end":
            if decision.reply:
                response["messages"] = new_msgs + [AIMessage(content=decision.reply)]
                logger.info("Exiting router_node")
                return response
            user_msg = query.lower().strip()

            # Predefined response variations
            smalltalk_responses = {
                "what is trinx ai": [
                    "TrinX AI is your intelligent assistant built inside the Trinity Coin ecosystem. I’m here to guide you through crypto, blockchain, coding, math, and even the latest news.",
                    "TrinX AI is an all-in-one assistant designed to help you explore Trinity Coin, understand crypto and blockchain, solve coding and math problems, and stay updated with news.",
                ],
                "who are you": [
                    "I’m TrinX AI, your personal assistant for Trinity Coin, crypto, blockchain, coding, math, and news.",
                    "I’m TrinX AI, an AI assistant created to help you with everything from crypto and blockchain to math, coding, and current events.",
                ],
                "who you": [
                    "I’m TrinX AI, your assistant for Trinity Coin and much more.",
                    "I’m TrinX AI, here to help you with crypto, blockchain, coding, math, and the latest news.",
                ],
                "hi": [
                    "Hello and welcome! I’m TrinX AI, your all-in-one assistant. Whether you’re curious about Trinity Coin, exploring the world of crypto and blockchain, brushing up on coding or math, or staying updated with the latest news, I’ve got you covered. How can I assist you today?"
                ],
                "how are you": [
                    "I’m doing great, thanks for asking! I’m always ready to help you with Trinity Coin, crypto, coding, math, or the latest news. How are you doing today?",
                    "I’m running smoothly and ready to assist. How are you feeling today?",
                ],
                "what are you up to": [
                    "I’m here, focused on helping you with anything you need—Trinity Coin, crypto, blockchain, coding, math, or news. What’s on your mind?",
                    "Right now, I’m waiting to help you explore crypto, answer math or coding questions, or share the latest updates. What would you like to start with?",
                ],
                "what's up": [
                    "Not much, just here and ready to assist with anything from Trinity Coin to coding and news. What’s up with you?",
                    "I’m all set to help you out with crypto, blockchain, coding, math, or the latest updates. What’s up on your end?",
                ],
            }

            # Match a response set (fallback = generic intro)
            replies = None
            for key, variations in smalltalk_responses.items():
                if key in user_msg:
                    replies = variations
                    break

            if not replies:
                replies = [
                    "Hello! I’m TrinX AI, your all-in-one assistant for Trinity Coin, crypto, blockchain, coding, math, and news. How can I help you today?"
                ]

            # Attach multiple possible AI responses
            response["messages"] = new_msgs + [
                AIMessage(content=msg) for msg in replies
            ]

        logger.info("Exiting router_node")
        return response
    except Exception as e:
        logger.error(f"Error in router_node: {e}")
        raise


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


def fetch_spot_price(state: AgentState) -> AgentState:
    try:
        logger.info("Entering price_node")
        query = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, HumanMessage)
            ),
            "",
        )
        logger.info(f"Price query: {query}")

        coin_decision: FetchCoinPriceDecision = coin_symbol_converter_llm.invoke(
            [
                ("system", fetch_price_prompt["fetch_price_node_v1"]["system"]),
                ("user", query),
            ]
        )
        symbol = (coin_decision.symbol or "").upper()
        if not symbol or symbol == "UNKNOWN":
            price_str = "Could not determine a valid trading symbol."
        else:
            price_result = fetch_coin_price_tool.invoke(
                {"symbol": symbol, "vs": "USDT"}
            )
            if (
                isinstance(price_result, dict)
                and "price" in price_result
                and "symbol" in price_result
            ):
                price_str = f"{price_result['symbol']}: {price_result['price']}"
            else:
                price_str = f"Price lookup failed: {price_result}"

        logger.info("Exiting price_node")
        return {
            **state,
            "price": price_str,
            "route": "answer",
            "previous_route": "fetch_price",
        }
    except Exception as e:
        logger.error(f"Error in price_node: {e}")
        raise


def trinity_coin_deatils_node(state: AgentState) -> AgentState:
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
