from typing import List
from pydantic import BaseModel, Field


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
    function: BaseModel = Field(
        None,
        description="Task function.",
    )

    class Config:
        arbitrary_types_allowed = True

    def load_functions(self):
        """Load functions for a task"""
        return self.function.get_schema()
