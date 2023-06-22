from data_collector.models import Task
from omegaconf import OmegaConf


def load_functions(task: Task):
    type = task.type
    if type == "sentiment":
        from functions.sentiment import SentimentFunctions

        sentiment_functions = SentimentFunctions.from_task_config(task)
        schema = SentimentFunctions.get_schema(sentiment_functions)

    elif type == "ner":
        from functions.ner import NERFunctions

        ner_functions = NERFunctions.from_task_config(task)
        schema = NERFunctions.get_schema(ner_functions)

    elif type == "tagging":
        from functions.topic_tagging import TagsFunctions

        tags_functions = TagsFunctions.from_task_config(task)
        schema = TagsFunctions.get_schema(tags_functions)

    elif type == "sts":
        from functions.sts import STSFunctions

        sts_functions = STSFunctions.from_task_config(task)
        schema = STSFunctions.get_schema(sts_functions)

    else:
        raise ValueError(f"Unsupported task: {type}")

    return schema
