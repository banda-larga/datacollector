"""Library use case."""

import datacollector.crawler as crawler
from datacollector import Task, CollectorArgs, Collector

urls = [
    "https:/www.example.com",
    "https://www.example.com",
    "https://www.example.com",
]

crawler.crawl(
    urls,
    push_to_hub=True,
    username="username",
    repo="repo",
    max_items=1000,
)

config = CollectorArgs(
    task=Task.SUMMARIZATION,
    dataset="xsum",
    language="it",
    max_items=1000,
    batch_size=10,
    num_proc=4,
    push_to_hub=True,
    username="username",
    repo="repo",
)

collector = Collector(config)
collector.build()
