from abc import ABC, abstractmethod
from typing import Any, Union, List, Callable, Dict
from pathlib import Path

from models import OutputParser
from datasets import load_dataset
from utils import generate_batch, generate
from pydantic import BaseModel, Field, validator
import os


class DatasetLoader:
    """Loads a dataset from a path or a Hugging Face dataset name

    Args:
        dataset (Union[str, Path]): Path to the dataset or Hugging Face dataset name
        split (str, optional): Split to load. Defaults to "train".
            It can also be: "train+test", "train[10:20]", "train[:10%]", "train[:10%]+train[-80%:]"
        streaming (bool, optional): Whether to load the dataset in streaming mode. Defaults to False.

    Returns:
        datasets.Dataset: Loaded dataset
    """

    @classmethod
    def load(
        cls,
        dataset: Union[str, Path],
        split: str = "train",
        streaming: bool = False,
        **kwargs,
    ):
        if dataset is None:
            raise ValueError("No dataset provided")

        # Load Hugging Face dataset
        if isinstance(dataset, str) and not os.path.exists(dataset):
            return load_dataset(dataset, split=split, streaming=streaming, **kwargs)

        dataset_path = Path(dataset)
        if not dataset_path.exists():
            raise ValueError("Dataset path does not exist")

        if dataset_path.is_dir():
            # Load all files in the folder
            files = {
                file.stem: str(file)
                for file in dataset_path.glob("*")
                if file.is_file()
            }
            return load_dataset("json", data_files=files)
        else:
            # Load single file
            if dataset_path.suffix == ".jsonl":
                return load_dataset("jsonl", data_files=str(dataset_path))
            elif dataset_path.suffix == ".csv":
                return load_dataset("csv", data_files=str(dataset_path))
            elif dataset_path.suffix == ".json":
                return load_dataset("json", data_files=str(dataset_path))
            else:
                raise ValueError("Dataset must be either a JSON, JSONL, or CSV file")


class AbstractDataPipeline(ABC):
    @abstractmethod
    def build(self, **kwargs: Any) -> None:
        pass


class Collector(AbstractDataPipeline):
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


class CollectorArgs(BaseModel):
    task: Task = Field(..., description="Task type.", example=Task.SUMMARIZATION)
    dataset: str = Field(..., description="Dataset name.", example="xsum")
    model: str = Field(..., description="Model name.", example="gpt-3.5-turbo-0613")
    language: str = Field(..., description="Language code.", example="it")
    max_items: int = Field(1000, description="Maximum number of items.", example=1000)
    batch_size: int = Field(10, description="Batch size.", example=10)
    num_proc: int = Field(4, description="Number of processes.", example=4)
    output_dir: str = Field(
        "output", description="Output directory path.", example="output"
    )
    save_every: int = Field(100, description="Save frequency.", example=100)
    push_to_hub: bool = Field(
        False, description="Push results to Hugging Face Hub.", example=True
    )

    @validator("batch_size")
    def batch_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("batch_size must be greater than 0")
        return v
