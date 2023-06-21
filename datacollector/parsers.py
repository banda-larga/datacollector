from typing import Dict, List
from abc import ABC, abstractmethod


class OutputParser(ABC):
    @abstractmethod
    def parse(self, output: Dict) -> Dict:
        pass

    @abstractmethod
    def parse_batch(self, output: List[Dict]) -> Dict:
        pass


class NERParser(OutputParser):
    def parse(self, output: Dict) -> Dict:
        """Parse the single output of a NER model."""
        return {
            "entities": output["entities"],
        }

    def parse_batch(self, output: List[Dict]) -> Dict:
        """Parse the batch output of a NER model."""

        return {
            "entities": [o["entities"] for o in output],
        }


class SentimentParser(OutputParser):
    def parse(self, output: Dict) -> Dict:
        return {
            "sentiment": output["sentiment"],
        }

    def parse_batch(self, output: List[Dict]) -> Dict:
        return {
            "sentiment": [o["sentiment"] for o in output],
        }


class STSParser(OutputParser):
    def parse(self, output: Dict) -> Dict:
        return {
            "score": int(output["score"]),
        }

    def parse_batch(self, output: List[Dict]) -> Dict:
        return {
            "score": [int(o["score"]) for o in output],
        }


class TagsParser(OutputParser):
    def parse(self, output: Dict) -> Dict:
        return {
            "tags": output["tags"],
        }

    def parse_batch(self, output: List[Dict]) -> Dict:
        return {
            "tags": [o["tags"] for o in output],
        }
