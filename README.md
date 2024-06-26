<div align="center">

# DataCollector: Create Datasets from Unstructured Sources using LLMs (WIP) 🚧

**Main Idea: A Langchain for data collection**

</div>

DataCollector is a Python library designed to collect data from unstructured sources using language models. The goal is to provide a simple API for collecting data for specific tasks (summarization, custom ner, ...), and crawling web pages.

⚠️ Please note that this project is currently in the early stages of development and is a work in progress. If you have any ideas or suggestions, feel free to open an issue to discuss them. The ultimate goal is to have a simple API for collecting data using GPT-4 and other LLMs to generate synthetic data for custom tasks.

In the future, we plan to integrate a tool like `Argilla` to supervise the data generation process.

## Usage (WIP) 🚧

### Collector

The `Collector` is designed to collect data for a specific task, such as sentiment analysis. Here's an example demonstrating how to use the Collector for collecting sentiments from the XSum dataset:

```python

from datacollector.functions import Classifier
from datacollector import Collector, CollectorArgs, Task

function = Classifier(
    name="print_sentiment",
    description="A function that prints the given sentiment.",
    output="sentiment",
    output_description="The sentiment.",
    labels=[
        "positive",
        "negative",
        "neutral",
    ],
)

task = Task(
    function = function,
    inputs = ["text", "role"],
    outputs = ["sentiment"],
    system = "You are a {role}.",
    prompt = "Classify the following text:\n{text}",
    language = "en",
)

args = CollectorArgs(
    task=task,
    dataset="xsum",
    model="gpt-3.5-turbo-0613",
    max_items=1000,
    batch_size=10,
    output_dir="output",
    save_every=100,
    push_to_hub=True,
)

collector = Collector(args)
collector.build()
collector.push_to_hub()
```

### Crawler (WIP) 🚧

The `Crawler` is designed to crawl web pages and collect data based on specified criteria. The following example demonstrates how to use the Crawler to crawl a single domain and collect data from the pages:

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

## Features ✨

- Collect data from various sources for specific tasks
- Crawl web pages and apply custom filters
- Easy-to-use API for data collection and processing

## Contributing 👋

We welcome contributions! If you have any ideas or suggestions, please open an issue to discuss them. If you'd like to contribute code, fork the repository, make your changes, and submit a pull request.

## License 📄

This project is licensed under the MIT License.
