from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, validator

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

    @classmethod
    def get_messages(
        cls,
        system_message: Optional[str] = None,
        user_message: str = "{text}",
        language: Optional[str] = None,
        **kwargs,
    ) -> List[Message]:
        messages = []

        if system_message is not None:
            if language is not None:
                system_message += (
                    f" You should ALWAYS write in {_LANGUAGE_TO_TEMPLATE[language]}."
                )
            messages.append(
                Message(role="system", content=system_message.format(**kwargs))
            )

        messages.append(Message(role="user", content=user_message.format(**kwargs)))

        return messages
