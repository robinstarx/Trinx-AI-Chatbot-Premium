"""Pydantic request/response schemas for the API layer."""

from __future__ import annotations

import base64
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Inbound payload for chat completions."""

    session_id: str | None = Field(
        default=None,
        description="Existing session identifier. Generates a new one when omitted.",
    )
    prompt: str = Field(..., min_length=1, description="User query text.")
    user_id: str = Field(..., description="User performing the upload.")
    is_file_upload: bool = Field(
        default=False,
        description="Flag that forces the router to answer using uploaded files.",
    )


class ChatResponse(BaseModel):
    """Unified response envelope for chat API endpoints."""

    session_id: str
    response: str


class UploadFileRequest(BaseModel):
    """Base64-backed document upload payload."""

    session_id: str | None = Field(
        default=None,
        description="Session identifier associated with the upload.",
    )
    user_id: str = Field(..., description="User performing the upload.")
    filename: str = Field(..., description="Original filename, including extension.")
    file_base64: str = Field(..., description="File contents encoded as base64 string.")
    content_type: str | None = Field(
        default=None,
        description="Optional MIME type supplied by the client.",
    )

    @field_validator("file_base64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        """Ensure the incoming string is valid base64."""

        try:
            base64.b64decode(value, validate=True)
        except (base64.binascii.Error, ValueError) as exc:  # pragma: no cover - guard clause
            raise ValueError("file_base64 must be a valid base64 string") from exc
        return value


