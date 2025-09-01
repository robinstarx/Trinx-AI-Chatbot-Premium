import logging
import requests

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def fetch_coin_price_tool(symbol: str, vs: str = "USDT") -> dict:
    """Fetch real-time coin price from Binance API."""
    return get_coin_price(symbol, vs)


def get_coin_price(symbol: str, vs: str = "USDT") -> dict:
    """
    Fetch the latest price of a cryptocurrency using Binance API.

    Args:
        symbol (str): The symbol of the cryptocurrency (e.g., "BTC", "ETH").
        vs (str): The quote asset (default "USDT").

    Returns:
        dict: { "symbol": "BTCUSDT", "price": "29500.23" }
    """
    try:
        pair = f"{symbol.upper()}{vs.upper()}"
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={pair}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {"symbol": data["symbol"], "price": data["price"]}
    except requests.exceptions.RequestException as e:
        logger.error(f"Binance request error: {e}")
        return {"error": str(e)}
