"""Ner function implementation to use in data_collector.
"""
from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field, validator


class Ner(BaseModel):
    """Ner function implementation to use in data_collector."""

    name: str = Field(
        description="Name of the function.",
        example="named_entity_recognition",
        default="named_entity_recognition",
    )
    description: str = Field(
        description="Description of the function.",
        example="Named entity recognition function.",
    )
    named_entities_description: str = Field(
        description="Description of the named entities.",
        example="List of named entities.",
    )
    labels: List[str] = Field(
        description="List of labels.",
        example=[
            "PERSON",
            "NORP",
            "FAC",
        ],
    )

    @validator("name")
    def name_must_be_named_entity_recognition(cls, v):
        if v != "named_entity_recognition":
            raise ValueError("name must be named_entity_recognition")
        return v

    @classmethod
    def get_schema(cls, config: Ner) -> List[Dict[str, Any]]:
        """Get schema of the function."""
        return [
            {
                "name": config.name,
                "description": config.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "named_entities": {
                            "type": "array",
                            "description": config.named_entities_description,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "label": {
                                        "type": "string",
                                        "enum": config.labels,
                                    },
                                },
                                "required": ["text", "label"],
                            },
                        },
                    },
                    "required": ["named_entities"],
                },
            }
        ]
