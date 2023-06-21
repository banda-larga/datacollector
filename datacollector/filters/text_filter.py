from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from datasets import Dataset
from transformers import AutoTokenizer
from langdetect import detect


class BaseTextFilter(ABC):
    @abstractmethod
    def __call__(self, text: str, **kwargs: Any) -> str:
        """Filters the text."""
        pass


class BaseListFilter(ABC):
    @abstractmethod
    def __call__(self, text: List[str], **kwargs: Any) -> List[str]:
        """Filters the list."""
        pass


class TextFilter(BaseTextFilter, ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create a TextFilter from a config."""
        pass


class MinTextLengthFilter(BaseTextFilter):
    def __init__(self, min_length: int):
        self.min_length = min_length

    def __call__(self, text: str) -> str:
        if len(text) >= self.min_length:
            return text
        return ""


class MaxTextLengthFilter(BaseTextFilter):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, text: str) -> str:
        return text[: self.max_length]


class TokenizedTextFilter(BaseTextFilter):
    def __init__(
        self,
        tokenizer: str,
        min_length: int = None,
        max_length: int = None,
        truncate: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.min_length = min_length
        self.max_length = max_length
        self.truncate = truncate

    def __call__(self, text: str) -> str:
        input_ids = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncate,
            padding=False,
            return_tensors="pt",
        )["input_ids"]

        if self.min_length is not None and input_ids.shape[1] < self.min_length:
            return ""

        return self.tokenizer.decode(
            input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )


class LanguageTextFilter(BaseTextFilter):
    def __init__(self, language: str):
        self.language = language

    def __call__(self, text: str) -> str:
        if detect(text) == self.language:
            return text
        return ""


class TextFilterFromConfig(TextFilter):
    """

    # start_with_ar = dataset.filter(lambda example: example["sentence1"].startswith("Ar"))

    Example yaml config:
    filters:
        input:
            tokenized: true
            min_token_length: 100
            max_token_length: 500
            truncate: true
            language: it

        output:
            min_num_entities: 3
            max_num_entities: 10
    """

    def __init__(self, config: Dict[str, Any], dataset: Optional[Dataset] = None):
        self.config = config
        self.dataset = dataset
        self.input_filters: List[BaseTextFilter] = []

        self._init_filters()

    def _init_filters(self):
        if "input" in self.config:
            self._init_input_filters()

        if "output" in self.config:
            self._init_output_filters()

    def _init_input_filters(self):
        input_config = self.config["input"]

        if input_config.get("tokenized"):
            self.input_filters.append(
                TokenizedTextFilter(
                    tokenizer=input_config["tokenizer"],
                    min_length=input_config.get("min_token_length"),
                    max_length=input_config.get("max_token_length"),
                    truncate=input_config.get("truncate", True),
                )
            )
        else:
            if input_config.get("min_length"):
                self.input_filters.append(
                    MinTextLengthFilter(input_config["min_length"])
                )

            if input_config.get("max_length"):
                self.input_filters.append(
                    MaxTextLengthFilter(input_config["max_length"])
                )

        if input_config.get("language"):
            self.input_filters.append(LanguageTextFilter(input_config["language"]))

    def _init_output_filters(self):
        output_config = self.config.get("output")

        if "min_num_entities" in output_config:
            self.filters.append(MinTextLengthFilter(output_config["min_num_entities"]))

        if "max_num_entities" in output_config:
            self.filters.append(MaxTextLengthFilter(output_config["max_num_entities"]))
