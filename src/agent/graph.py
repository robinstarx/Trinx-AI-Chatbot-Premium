import logging
from langgraph.graph import StateGraph, END
from src.agent.nodes.query_router import query_router_node
from src.agent.nodes.web_search import web_search_node
from src.agent.nodes.fetch_price import fetch_price_node
from src.agent.nodes.trinity_details import get_trinity_details_node
from src.agent.nodes.answer_compose import compose_answer_node
from src.agent.nodes.file_upload_qa import file_upload_qa_node
from src.core.state import AgentState
from src.agent.routes import route_intent
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.redis import RedisSaver

# checkpointer = RedisSaver(redis_url="redis://localhost:6379")
checkpointer = MemorySaver()

# ── LOGGING ────────
logger = logging.getLogger(__name__)

# ── Build graph ───────
logger.info("Building agent graph...")

workflow = StateGraph(AgentState)

workflow.add_node("route.intent", query_router_node)
workflow.add_node("search.news", web_search_node)
workflow.add_node("fetch.price", fetch_price_node)
workflow.add_node("file_upload_qa", file_upload_qa_node)
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
        "file_upload_qa":"file_upload_qa",
        "trinity_coin_details": "trinity_coin_details",
    },
)

workflow.add_edge("fetch.price", "answer.compose")
workflow.add_edge("file_upload_qa", "answer.compose")
workflow.add_edge("search.news", "answer.compose")
workflow.add_edge("trinity_coin_details", "answer.compose")

workflow.add_edge("answer.compose", END)

logger.info("Compiling agent graph...")
graph_agent = workflow.compile(checkpointer=checkpointer)
logger.info("Agent graph compiled successfully")