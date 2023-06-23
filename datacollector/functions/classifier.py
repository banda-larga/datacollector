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

    def get_schema(self) -> List[Dict[str, Any]]:
        """Get schema of the function."""
        return [
            {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        self.output: {
                            "type": "string",
                            "enum": self.labels,
                            "description": self.output_description,
                        },
                    },
                    "required": [self.output],
                },
            }
        ]
