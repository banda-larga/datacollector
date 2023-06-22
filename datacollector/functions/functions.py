from data_collector.models import Task
from omegaconf import OmegaConf


def load_functions(task: Task, config: OmegaConf):
    if task == "sentiment":
        from functions.sentiment import SentimentFunctions

        sentiment_functions = SentimentFunctions.from_config(config)
        schema = SentimentFunctions.get_schema(sentiment_functions)

    elif task == "ner":
        from functions.ner import NERFunctions

        ner_functions = NERFunctions.from_config(config)
        schema = NERFunctions.get_schema(ner_functions)

    elif task == "tagging":
        from functions.topic_tagging import TagsFunctions

        tags_functions = TagsFunctions.from_config(config)
        schema = TagsFunctions.get_schema(tags_functions)

    elif task == "sts":
        from functions.sts import STSFunctions

        sts_functions = STSFunctions.from_config(config)
        schema = STSFunctions.get_schema(sts_functions)

    else:
        raise ValueError(f"Unsupported task: {task}")

    return schema
