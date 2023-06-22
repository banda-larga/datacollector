from typing import List, Optional

from models import OutputParser
from pydantic import BaseModel, Field, validator
from functions import Function


class Task(BaseModel):
    """A task definition. It contains all the information needed to run a task."""

    _default_tasks = {
        "generation",
        "ner",
        "sentiment",
        "sts",
        "tagging",
        "classification",
    }

    type: str = Field(..., description="Task type.", example="generation")

    system: str = Field(None, description="Task role.", example="You are a {role}.")
    language: str = Field(None, description="Task language.", example="en")

    prompt: str = Field(
        "{text}",
        description="Task description.",
        example="Summarize the following text:\n{text}",
    )

    inputs: List[str] = Field(
        ..., description="Input columns.", example=["text", "role"]
    )
    outputs: List[str] = Field(
        ..., description="Task output columns.", example=["summary"]
    )
    output_parser: OutputParser = Field(
        ..., description="Output parser.", example="summary"
    )

    fn: Optional[Function] = Field(
        None, description="Function to apply to the input.", example=Function()
    )

    special_tokens: Optional[List[str]] = Field(
        None, description="Special tokens.", example=["<|endoftext|>"]
    )

    @validator("type")
    def type_must_be_supported(cls, v):
        if v not in cls._default_tasks:
            raise ValueError(
                "Task type must be one of: generation, ner, sentiment, sts, tagging, classification"
            )
        return v

    @validator("role")
    def role_only_for_generation(cls, v, values):
        if values.get("type") != "generation":
            raise ValueError("Task role is only supported for generation tasks")
        return v

    def require_functions(self) -> bool:
        return self.type != "generation"

    def load_functions(self):
        """Load functions for a task"""
        return self.get_schemas(self.fn)
