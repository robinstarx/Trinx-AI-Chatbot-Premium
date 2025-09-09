from pydantic import BaseModel, Field
from typing import List, Literal, TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Trinity Coin AI Knowledge Base
TRINITY_KNOWLEDGE_BASE = """
Trinity Coin AI Overview

Trinity Coin AI is a community-owned cryptocurrency project that combines artificial intelligence with blockchain 
technology on the Trinity Blockchain. It is designed to transform decentralized finance (DeFi) by providing 
AI-powered tools for users, including predictive modeling, automated trading strategies, and market analysis. 
The project supports multiple networks, such as ETH (ERC-20) and BNB (BEP-20), and incorporates a wide ecosystem 
of features like AI integration, smart contracts, wallets, exchanges, farming, referral systems, and educational 
platforms. At its core, Trinity Coin AI seeks to deliver scalable, secure, and intelligent infrastructure for
decentralized applications.

Vision and Goals

The vision of Trinity Coin AI is to revolutionize the cryptocurrency landscape through the seamless integration 
of AI within blockchain systems. Its goals include developing AI models that allow users to input data and generate
predictions on investment strategies or market trends, enhancing decentralized application interfaces with predictive
analytics, and deploying AI for smart contract validation and IoT integration. The ecosystem emphasizes security and
efficiency, offering automated transaction processing and AI-enhanced validation. To drive adoption, 
Trinity Coin AI also focuses on community engagement through rewards, airdrops, education, and partnerships, 
while advancing continuous innovation with AI-optimized projects, debit cards, merchant mapping, and intelligent 
chatbots.

Tokonomics and Presale

The native token, TIX, has a total supply of 400 billion, distributed across presale allocations, public rewards,
liquidity pools, development, marketing, burns, and community airdrops. The presale is divided into four rounds,
with 200 billion TIX reserved for community allocation. Users can participate in the presale using ETH or BNB,
following a simple process of selecting a payment method and entering the purchase amount. A referral program 
further incentivizes participation by rewarding users who connect wallets and share referral links. At present, 
Stage 1 of the presale has allocated 30% of its tokens, with over 31 billion TIX already sold.

Roadmap, Features, and Tools

The project roadmap includes phases ranging from blockchain foundation development and token distribution to the
launch of AI-driven exchanges, Web3 wallets, farming systems, debit cards, and launchpads for AI-powered projects.
Advanced features such as Trinity Virtual Machine (TVM), merchant mapping tools, educational academies, and the 
Mind AI Chatbot expand the ecosystem's utility. Core features include AI-enhanced validation for smart contracts,
market analysis, and automated transactions, supported by the Trinity AI Assistant (Trinix AI), which will deliver
insights, investment advice, and earning opportunities through interactive learning. Together, these initiatives
position Trinity Coin AI as an ambitious platform that combines AI innovation with blockchain infrastructure to 
build a secure, intelligent, and community-driven future in DeFi.
"""


class QueryDecision(BaseModel):
    route: Literal["fetch_price", "web", "answer", "trinity_coin_details"]
    reply: str | None = Field(None, description="")


class FetchCoinPriceDecision(BaseModel):
    symbol: str
    vs: str | None = Field(None, description="The quote asset (default 'USDT')")


# ── Shared state type ────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]  
    route: Literal["web", "fetch_price", "answer", "trinity_coin_details"]
    previous_route: Literal[
        "web", "fetch_price", "answer", "trinity_coin_details"
    ]
    web_results: str
    price: float
    trinity_info: str
    knowledge_base: str