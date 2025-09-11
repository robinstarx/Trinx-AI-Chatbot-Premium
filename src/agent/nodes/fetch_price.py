

import logging

from langchain_core.messages import HumanMessage
from src.core.llm_routes import coin_symbol_converter_llm
from src.core.state import AgentState, FetchCoinPriceDecision

from src.tools.fetch_coin_price import fetch_coin_price_tool

from src.prompts.config.prompt_import import fetch_price_prompt


logger = logging.getLogger(__name__)


def fetch_price_node(state: AgentState) -> AgentState:
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