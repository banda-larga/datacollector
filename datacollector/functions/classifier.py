"""Classifier function implementation to use in data_collector.
"""
from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class Classifier(BaseModel):
    """Classifier function implementation to use in data_collector."""

    name: str = Field(
        description="Name of the function.",
        example="print_sentiment",
    )

    description: str = Field(
        description="Description of the function.",
        example="A function that prints the given sentiment.",
    )

    output: str = Field(
        description="Output of the function.",
        example="sentiment",
    )

    output_description: str = Field(
        description="Description of the output.",
        example="The sentiment.",
    )

    labels: List[str] = Field(
        description="List of labels.",
        example=[
            "positive",
            "negative",
            "neutral",
        ],
    )

    @classmethod
    def get_schema(cls, config: Classifier) -> List[Dict[str, Any]]:
        """Get schema of the function."""
        return [
            {
                "name": config.name,
                "description": config.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        config.output: {
                            "type": "string",
                            "enum": config.labels,
                            "description": config.output_description,
                        },
                    },
                    "required": [config.output],
                },
            }
        ]
