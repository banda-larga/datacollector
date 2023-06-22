"""Topic tagging function implementation to use in data_collector.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from collector import Task


class Tag(BaseModel):
    """Tags function implementation to use in data_collector."""

    name: str = Field(
        description="Name of the function.",
        example="get_tags",
    )
    description: str = Field(
        description="Description of the function.",
        example="A function that returns the tags associated with the given text.",
    )
    labels: Optional[List[str]] = Field(
        description="List of labels.",
        example=[
            "python",
            "javascript",
            "java",
            "c#",
            "php",
            "jquery",
            "html",
            "c++",
            "css",
        ],
    )
    tags_description: str = Field(
        description="Description of the tags.",
        example="List of tags.",
    )

    @validator("name")
    def name_must_be_get_tags(cls, v):
        if v != "print_sentiment":
            raise ValueError("name must be get_tags")
        return v

    @classmethod
    def from_task_config(cls, config: Task) -> TagsFunctions:
        """Create TagsFunctions from a config."""
        return cls(
            name=config.get("name", "get_tags"),
            description=config.get(
                "description",
                "A function that returns the tags associated with the given text.",
            ),
            tags_description=config.get(
                "tags_description",
                "List of tags.",
            ),
            labels=config.get("labels"),
        )

    @classmethod
    def get_schema(cls, config: TagsFunctions) -> List[Dict[str, Any]]:
        """Get schema of the function."""
        return [
            {
                "name": "get_tags",
                "description": config.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "description": config.tags_description,
                            "items": {
                                "type": "string",
                            },
                        },
                    },
                    "required": ["tags"],
                },
            }
        ]
