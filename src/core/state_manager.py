from src.core.state import AgentState
import logging

logger = logging.getLogger(__name__)


def clear_context_fields(state: AgentState) -> AgentState:
    """Clear context fields to prevent contamination between queries"""
    cleared_state = {
        **state,
        "web_results": "",
        "price": 0.0,
        "trinity_info": "",
        "previous_route": "",
    }
    logger.info("Context fields cleared for new query")
    return cleared_state


def get_relevant_context(state: AgentState) -> str:
    """Get only the context relevant to the current route"""
    current_route = state.get("route", "")
    parts = []

    context_mappings = {
        "fetch_price": ("price", "Coin Price"),
        "web": ("web_results", "Web Search"),
        "trinity_coin_details": ("trinity_info", "Trinity Coin AI"),
        "answer": ("previous_route", "Previous Route"),
    }

    if current_route in context_mappings:
        field_name, label = context_mappings[current_route]
        field_value = state.get(field_name)

        if field_value and str(field_value).strip() and str(field_value) != "0.0":
            parts.append(f"{label}:\n{field_value}")

    return "\n\n".join(parts) if parts else "No external context available."
