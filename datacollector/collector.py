from abc import ABC, abstractmethod
from typing import Any, Union, List, Callable
from pathlib import Path

from models import OutputParser
from datasets import load_dataset
from utils import generate_batch, generate
import os


class DatasetLoader:
    @classmethod
    def load(cls, dataset: Union[str, Path]) -> Any:
        if dataset is None or not dataset.exists():
            raise ValueError("No dataset provided")

        if dataset.suffix == ".jsonl":
            return load_dataset("jsonl", data_files=str(dataset), field="data")
        elif dataset.suffix == ".csv":
            return load_dataset("csv", data_files=str(dataset))
        elif dataset.suffix == ".json":
            return load_dataset("json", data_files=str(dataset))
        else:
            raise ValueError("Dataset must be either a JSON or CSV file")


class AbstractDataPipeline(ABC):
    @abstractmethod
    def build(self, **kwargs: Any) -> None:
        pass


class DataPipeline(AbstractDataPipeline):
    def __init__(
        self,
        dataset: Union[str, Path],
        function: Callable,
        preprocessing_function: Callable,
        parser: OutputParser,
        input_columns: List[str] = ["text"],
        output_columns: List[str] = ["output"],
        push_to_hub: bool = False,
    ):
        self.dataset = DatasetLoader.load(dataset)

        self.preprocessing_function = preprocessing_function
        self.function = function
        self.parser = parser

        if not all(col in self.dataset.column_names for col in input_columns):
            raise ValueError(
                f"Input columns {input_columns} not in dataset columns {self.dataset.column_names}"
            )

        self.input_columns = input_columns
        self.output_columns = output_columns

        if push_to_hub and not os.environ.get("HUGGINGFACE_TOKEN"):
            raise ValueError(
                "You need to set the `HUGGINGFACE_TOKEN` environment variable to push to the Hub"
            )

        self.push_to_hub = push_to_hub

    def build(
        self,
        output_path: str = None,
        n_items: int = 100,
        batch_size: int = 10,
        num_proc: int = 4,
        username: str = None,
        repo: str = None,
        **kwargs,
    ):
        if n_items == 0:
            raise ValueError("No items to map")

        if self.preprocessing_function:
            self.dataset = self.dataset.map(
                self.preprocessing_function,
                batched=False,
                num_proc=num_proc,
                fn_kwargs=kwargs,
                input_columns=self.input_columns,
            )

        batched = batch_size > 1
        batch_size = min(batch_size, n_items) if batched else None

        if batched:
            self.dataset = self.dataset.map(
                self._map_batched,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                fn_kwargs=kwargs,
            )
        else:
            self.dataset = self.dataset.map(
                self._map,
                batched=False,
                num_proc=num_proc,
                fn_kwargs=kwargs,
            )

        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.dataset.save_to_disk(output_path, keep_in_memory=False)

        if self.push_to_hub:
            if not username or not repo:
                raise ValueError(
                    "You need to provide a username and repo to push to the Hub"
                )
            self.dataset.push_to_hub(
                f"{username}/{repo}",
                use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
                private=False,
            )

    def _map(self, x, **kwargs):
        text = x[self.column_name]
        result = generate(text, self.function, **kwargs)

        output = self.parser.parse(result)

        x[self.output_column_name] = output
        return x

    def _map_batched(self, x, **kwargs):
        texts = x[self.column_name]
        batch = generate_batch(texts, self.function, **kwargs)

        output = self.parser.parse_batch(batch)

        x[self.output_column_name] = output
        return x
