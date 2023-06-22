from abc import ABC, abstractmethod
from typing import Dict
from enum import Enum


class OutputParser(ABC):
    @abstractmethod
    def parse(self, output: Dict) -> Dict:
        pass


class Task(str, Enum):
    ner = "ner"
    sentiment = "sentiment"
    sts = "sts"
    paraphrase = "paraphrase"
    translation = "translation"
    generation = "generation"
