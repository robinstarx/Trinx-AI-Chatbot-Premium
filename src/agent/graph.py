import logging
from langgraph.graph import StateGraph, END
from src.agent.nodes import (
    query_intent_router,
    web_search_node,
    fetch_spot_price,
    compose_answer_node,
    trinity_coin_deatils_node,
)
from src.core.state import AgentState
from src.agent.routes import route_intent
from langgraph.checkpoint.memory import MemorySaver

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Build graph ─────────────────────────────────────────────────────
logger.info("Building agent graph...")

workflow = StateGraph(AgentState)

workflow.add_node("route.intent", query_intent_router)
workflow.add_node("search.news", web_search_node)
workflow.add_node("fetch.price", fetch_spot_price)
workflow.add_node("trinity_coin_details", trinity_coin_deatils_node)
workflow.add_node("answer.compose", compose_answer_node)

workflow.set_entry_point("route.intent")

workflow.add_conditional_edges(
    "route.intent",
    route_intent,
    {
        "fetch_price": "fetch.price",
        "web": "search.news",
        "answer": "answer.compose",
        "trinity_coin_details": "trinity_coin_details",
        "end": END,
    },
)

workflow.add_edge("fetch.price", "answer.compose")
workflow.add_edge("search.news", "answer.compose")
workflow.add_edge("trinity_coin_details", "answer.compose")

workflow.add_edge("answer.compose", END)

logger.info("Compiling agent graph...")
graph_agent = workflow.compile(checkpointer=MemorySaver())
logger.info("Agent graph compiled successfully.")
