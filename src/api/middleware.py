"""Reusable FastAPI middleware configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def register_cors(app: FastAPI) -> None:
    """Attach permissive CORS middleware suitable for local development."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def register_middlewares(app: FastAPI) -> None:
    """Register all middleware components."""

    register_cors(app)



