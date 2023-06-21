# DataCollector: create datasets from unstructured sources using LLMs

`Main Idea: A Langchain for data collection.`

DataCollector is a Python library for collecting data from unstructured sources using language models. It provides a simple API for collecting data for specific tasks, such as summarization, and crawling web pages. DataCollector is built on top of [HuggingFace Datasets](https://huggingface.co/docs/datasets/).

It's currently in the early stages of development, so it's a work in progress. If you have any ideas or suggestions, please open an issue to discuss them.

## Usage (WIP)

### Collector

The `Collector` is designed to collect data for a specific task, such as summarization. The following example demonstrates how to use the Collector for collecting cross-language summaries from the XSum dataset in Italian:

```python

from datacollector import Task, CollectorArgs, Collector

config = CollectorArgs(
    task=Task.SUMMARIZATION,
    dataset="xsum",
    model="gpt-3.5-turbo-0613"
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
collector.push_to_hub()
```

### Crawler (WIP)

The Crawler is designed to crawl web pages and collect data based on specified criteria. The following example demonstrates how to use the Crawler to crawl a website with a maximum depth of 2:

```python
from datacollector import Crawler, CrawlerArgs

crawler_args = CrawlerArgs(
    start_urls=["https://www.example.com"],
    max_depth=2,
    output_path="output",
    delay=0.5,
)

url_filter = UrlFilter()
url_filter.add_domain('example.com')

crawler = Crawler(args=crawler_args)
crawler.run(url_filter=url_filter)
crawler.push_to_hub()
```

## Features

- Collect data from various sources for specific tasks
- Crawl web pages and apply custom filters
- Easy-to-use API for data collection and processing

## Contributing

Contributions are welcome! If you have any ideas or suggestions, please open an issue to discuss them. If you'd like to contribute code, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.