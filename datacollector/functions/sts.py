"""Sentence similarity function implementation to use in data_collector.
"""
from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field, validator


class STS(BaseModel):
    """Sentence Similarity implementation to use in data_collector."""

    name: str = Field(
        description="Name of the function.",
        example="print_semantic_similarity",
    )
    description: str = Field(
        description="Description of the function.",
        example="A function that prints the semantic similarity.",
    )
    labels: List[str] = Field(
        description="List of values.",
        example=[0, 1, 2, 3, 4],
    )

    @validator("name")
    def name_must_be_print_similarity(cls, v):
        if v != "print_similarity":
            raise ValueError("name must be print_similarity")
        return v

    # @classmethod
    # def from_task_config(cls, config: Task) -> STSFunctions:
    #     """Create STSFunctions from a config."""
    #     return cls(
    #         name=config.get("name", "print_similarity"),
    #         description=config.get("description", "Prints the semantic similarity."),
    #         sentiment_description=config.get(
    #             "sentiment_description",
    #             "The semantic similarity.",
    #         ),
    #         labels=config.get(
    #             "values",
    #             [0, 1, 2, 3, 4],
    #         ),
    #     )

    @classmethod
    def get_schema(cls, config: STS) -> List[Dict[str, Any]]:
        """Get schema of the function."""
        return [
            {
                "name": "print_similarity",
                "description": config.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "similarity": {
                            "type": "integer",
                            "enum": config.labels,
                            "description": config.similarity_description,
                        },
                    },
                    "required": ["similarity"],
                },
            }
        ]
