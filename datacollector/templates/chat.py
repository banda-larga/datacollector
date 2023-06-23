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

    @validator("language", pre=True, always=True)
    def validate_language(cls, language: Optional[str]) -> Optional[str]:
        if language is not None and language not in _LANGUAGE_TO_TEMPLATE:
            raise ValueError(f"Language {language} not supported.")
        return language

    def get_messages(
        self,
        **kwargs,
    ) -> List[Message]:
        messages = []

        if self.system_message is not None:
            if self.language is not None:
                self.system_message += f" You should ALWAYS write in {_LANGUAGE_TO_TEMPLATE[self.language]}."
            messages.append(
                Message(role="system", content=self.system_message.format(**kwargs))
            )

        messages.append(
            Message(role="user", content=self.user_message.format(**kwargs))
        )
        messages = [message.dict() for message in messages]
        return messages


if __name__ == "__main__":
    chat_prompt_template = ChatPromptTemplate(
        system_message="You are a helpful assistant that translates {input_language} to {output_language}.",
        user_message="{text}",
        language="en",
    )

    messages = chat_prompt_template.get_messages(
        input_language="English", output_language="French", text="Hello!"
    )
    print(messages)
