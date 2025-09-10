import logging
import requests

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Common ticker → CoinGecko ID mapping
COINGECKO_SYMBOL_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "USDT": "tether",
    "USDC": "usd-coin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "SOL": "solana",
    "DOT": "polkadot",
    "MATIC": "polygon",
}

# Map quote assets for CoinGecko
COINGECKO_VS_MAP = {
    "USDT": "usd",
    "USDC": "usd",
    "BUSD": "usd",
    "BTC": "btc",
    "ETH": "eth",
    "INR": "inr",
    "EUR": "eur",
    "USD": "usd",
}

@tool
def fetch_coin_price_tool(symbol: str, vs: str = "usd") -> dict:
    """Fetch real-time coin price. Try Binance first, fallback to CoinGecko if Binance fails."""
    return get_coin_price(symbol, vs)


def get_coin_price(symbol: str, vs: str = "usd") -> dict:
    
  
    try:
        coin_id = COINGECKO_SYMBOL_MAP.get(symbol.upper())
        vs_id = COINGECKO_VS_MAP.get(vs.upper(), vs.lower())  # map if known

        if not coin_id:
            return {"error": f"No CoinGecko mapping found for symbol {symbol}"}

        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin_id, "vs_currencies": vs_id}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        if coin_id in data and vs_id in data[coin_id]:
            return {
                "symbol": f"{symbol.upper()}{vs.upper()}",
                "price": str(data[coin_id][vs_id]),
                "source": "coingecko",
            }
        else:
            return {"error": "CoinGecko: Symbol or currency not found"}
    except requests.exceptions.RequestException as e:
        logger.error(f"CoinGecko request error: {e}")
        return {"error": str(e)}
