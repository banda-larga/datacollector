from typing import List, Optional

from out_parser.parser import OutputParser
from pydantic import BaseModel, Field, validator
from functions.functions import Function


class Task(BaseModel):
    """A task definition. It contains all the information needed to run a task."""

    system: str = Field(None, description="Task role.", example="You are a {role}.")
    language: str = Field(None, description="Task language.", example="en")
    prompt: str = Field(
        "{text}",
        description="Task description.",
        example="Summarize the following text:\n{text}",
    )
    inputs: List[str] = Field(
        ["text"], description="Input columns.", example=["text", "role"]
    )
    outputs: List[str] = Field(
        ["output"], description="Task output columns.", example=["summary"]
    )
    function: Function = Field(
        None,
        description="Task function.",
        example={
            "name": "get_tags",
            "description": "A function that returns the tags associated with the given text.",
            "tags_description": "List of tags.",
            "labels": [
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
        },
    )

    class Config:
        arbitrary_types_allowed = True

    def load_functions(self):
        """Load functions for a task"""
        return self.function.get_schema()
