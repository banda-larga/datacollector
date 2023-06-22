from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable


class Function(ABC):
    @classmethod
    @abstractmethod
    def from_task_config(cls, config: Callable) -> function:
        pass

    @classmethod
    @abstractmethod
    def get_schema(cls, config: function) -> List[Dict[str, Any]]:
        pass


def load_functions(task: Task):
    type = task.type
    if type == "sentiment":
        from functions.sentiment import Sentiment

        sentiment_functions = Sentiment.from_task_config(task)
        schema = Sentiment.get_schema(sentiment_functions)

    elif type == "ner":
        from functions.ner import NER

        ner_functions = NER.from_task_config(task)
        schema = NER.get_schema(ner_functions)

    elif type == "tagging":
        from functions.topic_tagging import Tag

        tags_functions = Tag.from_task_config(task)
        schema = Tag.get_schema(tags_functions)

    elif type == "sts":
        from functions.sts import STS

        sts_functions = STS.from_task_config(task)
        schema = STS.get_schema(sts_functions)

    else:
        raise ValueError(f"Unsupported task: {type}")

    return schema
