# app/main.py
from __future__ import annotations

import base64
import binascii
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException

import uvicorn
from langchain_core.messages import AIMessage, HumanMessage
import logging
import uuid
from src.core.config import Settings

from src.agent.graph import graph_agent
from src.rag.rag_store import add_docs, vector_store
from src.api.middleware import register_middlewares
from src.api.scheme import ChatRequest, ChatResponse, UploadFileRequest

logger = logging.getLogger(__name__)
settings = Settings()


app = FastAPI(
    title="Trinity Chat Premium API",
    version="1.0.0",
    description="Premium Chatbot API powered by LangGraph",
)


register_middlewares(app)


@app.get("/check-health", tags=["health"])
async def health_check():
    """
    Health check endpoint.

    This endpoint can be used to verify that the API is running and responsive.
    It returns a simple JSON response indicating the status.
    """
    return {"status": "I'm alive and healthy"}


@app.post(settings.API_URL, response_model=ChatResponse, tags=["chat"])
async def chat_premium(request: ChatRequest):
    """
    Chat endpoint that takes session_id & query,
    and returns the chatbot's markdown answer.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        query = request.prompt.strip()
        user_id = request.user_id

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        logger.info(f"[Session: {session_id}] User Query: {query}")

        # Call LangGraph Agent
        res = graph_agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "configurable": {
                    "thread_id": session_id,
                    "user_id": user_id,
                    "is_file_upload": request.is_file_upload,
                }
            },
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


@app.post(f"{settings.API_URL}/upload-file", response_model=ChatResponse, tags=["upload-file"])
async def upload_file(request: UploadFileRequest):
    """
    Upload file endpoint that takes session_id & file,
    and returns the chatbot's markdown answer.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        file_base64 = request.file_base64
        user_id = request.user_id
        filename = request.filename

        if not file_base64 or not user_id:
            raise HTTPException(status_code=400, detail="file_base64 and user_id are required")

        try:
            binary_data = base64.b64decode(file_base64, validate=True)
        except (binascii.Error, ValueError) as decode_error:
            raise HTTPException(status_code=400, detail="Invalid base64 payload") from decode_error

        suffix = Path(filename).suffix or ""
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(binary_data)
                tmp.flush()
                tmp_path = Path(tmp.name)

            added = add_docs(
                file_path=str(tmp_path),
                vector_store=vector_store,
                metadata={"source": "upload", "filename": filename, "user_id": user_id},
            )

            if not added:
                raise HTTPException(status_code=422, detail="Unable to extract contents from file")
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        return ChatResponse(
            session_id=session_id,
            response="File uploaded successfully",
        )
    except Exception as e:
        logger.error(f"Error in upload-file endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    # Open tunnel on port 8000
    # public_url = ngrok.connect(8000)
    # print(f"✅ Public URL for API: {public_url}")

    # Run FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8008)
