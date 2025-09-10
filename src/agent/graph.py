import logging
from langgraph.graph import StateGraph, END
from src.agent.nodes.query_router import query_router_node
from src.agent.nodes.web_search import web_search_node
from src.agent.nodes.fetch_price import fetch_price_node
from src.agent.nodes.trinity_details import get_trinity_details_node
from src.agent.nodes.answer_compose import compose_answer_node
from src.core.state import AgentState
from src.agent.routes import route_intent
from langgraph.checkpoint.memory import MemorySaver

# ── LOGGING ────────
logger = logging.getLogger(__name__)

# ── Build graph ───────
logger.info("Building agent graph...")

workflow = StateGraph(AgentState)

workflow.add_node("route.intent", query_router_node)
workflow.add_node("search.news", web_search_node)
workflow.add_node("fetch.price", fetch_price_node)
workflow.add_node("trinity_coin_details", get_trinity_details_node)
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
    },
)

workflow.add_edge("fetch.price", "answer.compose")
workflow.add_edge("search.news", "answer.compose")
workflow.add_edge("trinity_coin_details", "answer.compose")

workflow.add_edge("answer.compose", END)

logger.info("Compiling agent graph...")
graph_agent = workflow.compile(checkpointer=MemorySaver())
logger.info("Agent graph compiled successfully")