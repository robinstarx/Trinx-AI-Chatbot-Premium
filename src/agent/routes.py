import logging
from typing import Literal
from src.core.state import AgentState

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Routing helpers ─────────────────────────────────────────────────
def route_intent(
    st: AgentState,
) -> Literal["fetch_price", "web", "answer", "trinity_coin_details", "file_upload_qa"]:
    route = st["route"]
    logger.info(f"Routing from router: {route}")
    return route
