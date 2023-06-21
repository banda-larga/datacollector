from abc import ABC, abstractmethod
from typing import Dict
from enum import Enum


class OutputParser(ABC):
    @abstractmethod
    def parse(self, output: Dict) -> Dict:
        pass


class TaskType(str, Enum):
    ner = "ner"
    sentiment = "sentiment"
    summarization = "summarization"
    sts = "sts"
    neutralization = "neutralization"
