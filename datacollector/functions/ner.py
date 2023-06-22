"""Ner function implementation to use in data_collector.
"""
from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field, validator
from collector import Task
from functions import Function


class Ner(BaseModel, Function):
    """Ner function implementation to use in data_collector."""

    name: str = Field(
        description="Name of the function.",
        example="named_entity_recognition",
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
    def from_task_config(cls, config: Task) -> NerFunctions:
        """Create NerFunctions from a config."""
        return cls(
            name=config.get("name", "named_entity_recognition"),
            description=config.get(
                "description", "A named entity recognition function."
            ),
            named_entities_description=config.get(
                "named_entities_description",
                "List of named entities.",
            ),
            labels=config.get(
                "labels",
                [
                    "PERSON",
                    "NORP",
                    "FAC",
                    "ORG",
                    "GPE",
                    "LOC",
                    "PRODUCT",
                    "EVENT",
                    "WORK_OF_ART",
                    "LAW",
                    "LANGUAGE",
                    "DATE",
                    "TIME",
                    "PERCENT",
                    "MONEY",
                    "QUANTITY",
                    "ORDINAL",
                    "CARDINAL",
                ],
            ),
        )

    @classmethod
    def get_schema(cls, config: NerFunctions) -> List[Dict[str, Any]]:
        """Get schema of the function."""
        return [
            {
                "name": "named_entity_recognition",
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
