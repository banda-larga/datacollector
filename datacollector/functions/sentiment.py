"""Sentiment Analysis function implementation to use in data_collector.
"""
from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field, validator
from collector import Task


class SentimentFunctions(BaseModel):
    """Sentiment function implementation to use in data_collector."""

    name: str = Field(
        description="Name of the function.",
        example="print_sentiment",
    )
    description: str = Field(
        description="Description of the function.",
        example="A function that prints the given sentiment.",
    )
    labels: List[str] = Field(
        description="List of labels.",
        example=[
            "positive",
            "negative",
            "neutral",
        ],
    )

    @validator("name")
    def name_must_be_print_sentiment(cls, v):
        if v != "print_sentiment":
            raise ValueError("name must be print_sentiment")
        return v

    @classmethod
    def from_task_config(cls, config: Task) -> SentimentFunctions:
        """Create SentimentFunctions from a config."""
        return cls(
            name=config.get("name", "print_sentiment"),
            description=config.get(
                "description", "A function that prints the given sentiment."
            ),
            sentiment_description=config.get(
                "sentiment_description",
                "The sentiment.",
            ),
            labels=config.get(
                "labels",
                [
                    "positive",
                    "negative",
                    "neutral",
                ],
            ),
        )

    @classmethod
    def get_schema(cls, config: SentimentFunctions) -> List[Dict[str, Any]]:
        """Get schema of the function."""
        return [
            {
                "name": "print_sentiment",
                "description": config.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "enum": config.labels,
                            "description": config.sentiment_description,
                        },
                    },
                    "required": ["sentiment"],
                },
            }
        ]
