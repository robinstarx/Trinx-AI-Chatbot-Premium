

import json
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API settings
    API_PREFIX: str = "/api"
    # CORS_ORIGINS: list[str] = Field(s
    #     default_factory=list,
    #     env="CORS_ORIGINS",
    # )

    # @field_validator("CORS_ORIGINS", mode="before")
    # def parse_cors_origins(cls, v):
    #     if isinstance(v, str):
    #         try:
    #             # First try parsing as JSON
    #             return json.loads(v)
    #         except json.JSONDecodeError:
    #             # If that fails, try splitting by comma
    #             return [origin.strip() for origin in v.split(",")]
    #     return v
    # LLM settings
    
    
    class Config:
        env_file = ".env"
        model_config = {"extra": "ignore"}

settings = Settings()