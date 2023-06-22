# from functions.functions import load_function
from functions.ner import Ner
from functions.classifier import Classifier
from functions.sts import STS
from functions.topic_tagging import Tag
from functions.functions import Function

__all__ = [
    "Function",
    "Ner",
    "Classifier",
    "STS",
    "Tag",
]
