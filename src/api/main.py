# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyngrok import ngrok
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
import logging
import uuid
from src.core.config import Settings

from src.agent.graph import graph_agent


logger = logging.getLogger(__name__)
settings = Settings()


app = FastAPI(
    title="Trinity Chat Premium API",
    version="1.0.0",
    description="Premium Chatbot API powered by LangGraph",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str | None = None
    query: str


class ChatResponse(BaseModel):
    session_id: str
    response: str

@app.get("/check-health", tags=["health"])
async def health_check():
    """
    Health check endpoint.

    This endpoint can be used to verify that the API is running and responsive.
    It returns a simple JSON response indicating the status.
    """
    return {"status": "i'm alive and healthy"}

@app.post(settings.API_URL, response_model=ChatResponse)
async def chat_premium(request: ChatRequest):
    """
    Chat endpoint that takes session_id & query,
    and returns the chatbot's markdown answer.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        query = request.query.strip()

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        logger.info(f"[Session: {session_id}] User Query: {query}")

        # Call LangGraph Agent
        res = graph_agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": session_id}},
        )
        answer = next(
            (m for m in reversed(res["messages"]) if isinstance(m, AIMessage)), None
        )

        if not answer:
            raise HTTPException(status_code=500, detail="No AI response generated")

        logger.info(f"[Session: {session_id}] Answer: {answer.content[:100]}...")

        return ChatResponse(
            session_id=session_id,
            response=answer.content,
        )

    except Exception as e:
        logger.error(f"Error in chat-premium endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    # Open tunnel on port 8000
    # public_url = ngrok.connect(8000)
    # print(f"✅ Public URL for API: {public_url}")

    # Run FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8080)
