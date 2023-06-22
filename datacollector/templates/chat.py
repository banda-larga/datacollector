from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from string import Template

_LANGUAGE_TO_TEMPLATE = {
    "en": "English",
    "it": "Italian",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "ar": "Arabic",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "el": "Greek",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
}


class Message(BaseModel):
    role: str = Field(description="Role of the message.", example="system")
    content: str = Field(description="Content of the message.", example="Hello!")


class ChatPromptTemplate(BaseModel):
    system_message: str = Field(
        description="System message template.",
        example="You are a helpful assistant that translates {input_language} to {output_language}.",
    )
    user_message: str = Field(
        description="Human message template.",
        example="{text}",
    )
    language: Optional[str] = Field(
        description="Language of the prompt.",
        example="en",
    )

    @validator("language")
    def validate_language(cls, language: str) -> str:
        if language not in _LANGUAGE_TO_TEMPLATE:
            raise ValueError(
                "language must be one of: " + ", ".join(_LANGUAGE_TO_TEMPLATE.keys())
            )
        return language

    def get_messages(
        self,
        **kwargs,
    ) -> List[Message]:
        messages = []

        if self.system_message is not None:
            system_template = Template(self.system_message)
            if self.language is not None:
                system_message += f" You should ALWAYS write in {_LANGUAGE_TO_TEMPLATE[self.language]}."
            messages.append(
                Message(role="system", content=system_template.substitute(**kwargs))
            )

        user_template = Template(self.user_message)
        messages.append(
            Message(role="user", content=user_template.substitute(**kwargs))
        )
        messages = [message.dict() for message in messages]
        return messages
