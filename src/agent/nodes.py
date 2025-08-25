import logging

from langchain_core.messages import HumanMessage, AIMessage
from src.core.llm_routes import query_router_llm, coin_symbol_converter_llm, trinity_coin_details_llm, answer_compose_llm
from src.core.state import AgentState, QueryDecision, FetchCoinPriceDecision, TRINITY_KNOWLEDGE_BASE
# from src.tools.web_search import web_search_tool
from src.tools.fetch_coin_price import fetch_coin_price_tool
from src.tools.serper_web_search import web_search_tool

import yaml

with open("src/prompts/router_prompt.yaml", "r", encoding="utf-8") as f:
    router_prompt = yaml.safe_load(f)

with open("src/prompts/fetch_price_node_prompt.yaml", "r", encoding="utf-8") as f:
    fetch_price_prompt = yaml.safe_load(f)

with open("src/prompts/trinity_info_node.yaml", "r", encoding="utf-8") as f:
    trinity_info_prompt = yaml.safe_load(f)

with open("src/prompts/answer_node_prompt.yaml", "r", encoding="utf-8") as f:
    answer_prompt = yaml.safe_load(f)

# ── LOGGING ────
logger = logging.getLogger(__name__)

def query_intent_router(state: AgentState) -> AgentState:
    try:
        logger.info("Entering route_intent_node")
        user_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        query = user_msgs[-1].content if user_msgs else ""
        logger.info(f"Router query: {query}")
        
        # system_prompt = (
        #         "You are the intent router for a crypto assistant. Choose exactly one route from: 'web', 'fetch_price', 'trinity_coin_details', 'answer', 'end'.\n"
        #         "\n"
        #         "- 'end': Use for greetings/small talk/thanks/compliments or empty/noise (e.g., 'hi', 'hello', 'thanks', 'how are you?'). Include a short friendly reply in 'reply'.\n"
        #         "- 'fetch_price': Use for requests that ask for a coin/token price or live quote (e.g., 'btc price', 'price of SOL in usd', 'ETH price now').\n"
        #         "- 'trinity_coin_details': Use for questions about Trinity Coin AI, its features, roadmap, or any other information related to the project.\n"
        #         "- 'web': Use for news/updates/headlines/announcements/regulatory items, 'what's happening' about coins/projects/exchanges/DeFi/NFTs/airdrops.\n"
        #         "- 'answer': Use for conceptual/explanatory questions that do not require real‑time lookup.\n"
        #         "\n"
        #         "Ambiguity rule: If a coin/ticker appears with words like 'price', 'quote', 'now', 'today', 'latest', 'update', prefer 'fetch_price'. Otherwise choose 'answer'.\n"
        #         "\n"
        #         "Return only the structured fields: route (one of the four) and reply (only when route == 'end').\n"
        # )

        decision: QueryDecision = query_router_llm.invoke([
            ("system", router_prompt["router_v1"]["system"]),
            ("user", query)
        ])
        logger.info(f"Router decision: {decision.route}")

        new_msgs = state.get("messages", []) + [HumanMessage(content=query)]
        response: AgentState = {
            **state,
            "messages": new_msgs,
            "route": decision.route
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
                    "TrinX AI is an all-in-one assistant designed to help you explore Trinity Coin, understand crypto and blockchain, solve coding and math problems, and stay updated with news."
                ],
                "who are you": [
                    "I’m TrinX AI, your personal assistant for Trinity Coin, crypto, blockchain, coding, math, and news.",
                    "I’m TrinX AI, an AI assistant created to help you with everything from crypto and blockchain to math, coding, and current events."
                ],
                "who you": [
                    "I’m TrinX AI, your assistant for Trinity Coin and much more.",
                    "I’m TrinX AI, here to help you with crypto, blockchain, coding, math, and the latest news."
                ],
                "hi": [
                    "Hello and welcome! I’m TrinX AI, your all-in-one assistant. Whether you’re curious about Trinity Coin, exploring the world of crypto and blockchain, brushing up on coding or math, or staying updated with the latest news, I’ve got you covered. How can I assist you today?"
                ],
                "how are you": [
                    "I’m doing great, thanks for asking! I’m always ready to help you with Trinity Coin, crypto, coding, math, or the latest news. How are you doing today?",
                    "I’m running smoothly and ready to assist. How are you feeling today?"
                ],
                "what are you up to": [
                    "I’m here, focused on helping you with anything you need—Trinity Coin, crypto, blockchain, coding, math, or news. What’s on your mind?",
                    "Right now, I’m waiting to help you explore crypto, answer math or coding questions, or share the latest updates. What would you like to start with?"
                ],
                "what's up": [
                    "Not much, just here and ready to assist with anything from Trinity Coin to coding and news. What’s up with you?",
                    "I’m all set to help you out with crypto, blockchain, coding, math, or the latest updates. What’s up on your end?"
                ]
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
            response["messages"] = new_msgs + [AIMessage(content=msg) for msg in replies]

        logger.info("Exiting router_node")
        return response
    except Exception as e:
        logger.error(f"Error in router_node: {e}")
        raise

def web_search_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering web_node")
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")
        logger.info(f"Web search query: {query}")

        result = web_search_tool.invoke({
            "query": query})
        logger.info(f"Web search returned content length: {len(result)}")

        logger.info("Exiting web_node")
        return {**state, "web_results": result, "route": "answer"}
    except Exception as e:
        logger.error(f"Error in web_node: {e}")
        raise

def fetch_spot_price(state: AgentState) -> AgentState:
    try:
        logger.info("Entering price_node")
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")
        logger.info(f"Price query: {query}")
        # system_prompt = (
        #    """You are a Crypto Query Converter.
        #         Your task is to take any user query related to checking the price of a cryptocurrency and return only the correct Binance trading symbol of that coin.
        #         Instructions:
        #         - Always output the exact Binance-accepted symbol (e.g., Bitcoin → BTC, Ethereum → ETH, Cardano → ADA).
        #         - Do not include extra text, explanations, or formatting.

        #         - If the query contains multiple coins, output only the primary coin explicitly mentioned.

        #         - If the coin name is ambiguous or not listed on Binance, return UNKNOWN.

        #         - Ignore unrelated text in the query and focus only on identifying the coin symbol.

        #         Examples:

        #         Input: “What’s the price of Bitcoin right now?” → Output: BTC

        #         Input: “Show me Ethereum live price.” → Output: ETH

        #         Input: “Tell me Cardano latest update.” → Output: ADA

        #         Input: “How much is Dogecoin trading for?” → Output: DOGE

        #         Input: “What’s the price of Tesla stock?” → Output: UNKNOWN"""
        # )
        coin_decision: FetchCoinPriceDecision = coin_symbol_converter_llm.invoke([
            ("system", fetch_price_prompt["fetch_price_node_v1"]["system"]),
            ("user", query)
        ])
        symbol = (coin_decision.symbol or "").upper()
        if not symbol or symbol == "UNKNOWN":
            price_str = "Could not determine a valid trading symbol."
        else:
            price_result = fetch_coin_price_tool.invoke({"symbol": symbol, "vs": "USDT"})
            if isinstance(price_result, dict) and "price" in price_result and "symbol" in price_result:
                price_str = f"{price_result['symbol']}: {price_result['price']}"
            else:
                price_str = f"Price lookup failed: {price_result}"

        logger.info("Exiting price_node")
        return {**state, "price": price_str, "route": "answer"}
    except Exception as e:
        logger.error(f"Error in price_node: {e}")
        raise


def trinity_coin_deatils_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering trinity_coin_deatils_node")
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")
        logger.info(f"Trinity coin details query: {query}")
        # system_prompt = (
        #     f"""You are a helpful assistant. Use the information below to answer the user's question. 
        #     {TRINITY_KNOWLEDGE_BASE}
        #     - Now answer clearly and concisely
        #     - Do not give everything in the knowledge base, just the relevant information
        #     """
        # )
        trinity_answer = trinity_coin_details_llm.invoke([
    ("system", trinity_info_prompt["trinity_info_node_v1"]["system"].format(TRINITY_KNOWLEDGE_BASE=TRINITY_KNOWLEDGE_BASE)),
    ("user", query)
])
        logger.info("Generated answer.")
        logger.info(f"Answer: {trinity_answer.content}")
        logger.info("Exiting trinity_coin_deatils_node")
        return {**state, "trinity_info": trinity_answer.content, "route": "answer"}
    except Exception as e:
        logger.error(f"Error in trinity_coin_deatils_node: {e}")
        raise



# ── Node 4: final answer ─────────────────────────────────────────────
def compose_answer_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering answer_node")
        user_q = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)),
            ""
        )
        logger.info(f"Answer node query: {user_q}")

        chat_history = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in state["messages"]
        )
        
        web_ctx = state.get("web_results", "").strip()
        coin_price_ctx = state.get("price","").strip()

        parts = []
        if coin_price_ctx:
            parts.append("Coin Price:\n" + coin_price_ctx)
        if web_ctx:
            parts.append("Web Search:\n" + web_ctx)
        if state.get("trinity_info", ""):
            parts.append("Trinity Coin AI:\n" + state.get("trinity_info", ""))
        context = "\n\n".join(parts) if parts else "No external context available."
        logger.info("Answer node context prepared")

        # prompt = f"""
        #             You are a helpful assistant. Use the information below to answer the user's latest question.

        #             Conversation so far:
        #             {chat_history}

        #             Context:
        #             {context}

        #             Now answer clearly and concisely:
        #             Question: {user_q}"""

        answer = answer_compose_llm.invoke([HumanMessage(content=answer_prompt["answer_node_v1"]["system"].format(context=context, user_q=user_q))]).content
        logger.info("Generated answer.")

        logger.info("Exiting answer_node")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=answer)]
        }
    except Exception as e:
        logger.error(f"Error in answer_node: {e}")
        raise