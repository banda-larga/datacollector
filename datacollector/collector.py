from abc import ABC, abstractmethod
from typing import Any, Union, List
from pathlib import Path

from task import Task
from datasets import load_dataset
from utils import generate
from pydantic import BaseModel, Field, validator
from datasets import Dataset
from templates.chat import ChatPromptTemplate
import os


class CollectorArgs(BaseModel):
    task: Task = Field(None, description="Task arguments.", example=Task())
    dataset: str = Field(None, description="Dataset name.", example="xsum")
    model: str = Field(None, description="Model name.", example="gpt-3.5-turbo-0613")
    language: str = Field(None, description="Language code.", example="it")
    max_items: int = Field(1000, description="Maximum number of items.", example=1000)
    batch_size: int = Field(10, description="Batch size.", example=10)
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
        **kwargs,
    ):
        if dataset is None:
            raise ValueError("No dataset provided")

        if isinstance(dataset, str) and not os.path.exists(dataset):
            return load_dataset(dataset, split=split, **kwargs)

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
            file_extension_map = {
                ".jsonl": "jsonl",
                ".csv": "csv",
                ".json": "json",
            }

            if dataset_path.suffix in file_extension_map:
                return load_dataset(
                    file_extension_map[dataset_path.suffix],
                    data_files=str(dataset_path),
                )
            else:
                raise ValueError("Dataset must be either a JSON, JSONL, or CSV file")

    @classmethod
    def prepare(
        cls,
        dataset: Dataset,
        args: CollectorArgs,
    ) -> Dataset:

        language = None if args.task.function else args.task.language

        template = ChatPromptTemplate(
            prompt=args.task.prompt,
            system=args.task.system,
            language=language,
        )

        return dataset.map(
            lambda x: template.format(**x),
            batched=True,
            batch_size=args.batch_size,
            input_columns=args.task.inputs,
        )


class Collector(BaseModel):
    def __init__(
        self,
        dataset: Union[str, Path],
        output_columns: List[str] = ["output"],
        args: CollectorArgs = CollectorArgs(),
    ):
        self.args = args

        if self.args.task.function:
            self.function = self.args.task.load_functions()
        else:
            self.function = None

        dataset = DatasetLoader.load(dataset)
        self.dataset = DatasetLoader.prepare(dataset, self.args)

        input_columns = self.args.task.inputs
        output_columns = self.args.task.outputs

        if not all(col in self.dataset.column_names for col in input_columns):
            raise ValueError(
                f"Input columns {input_columns} not in dataset columns {self.dataset.column_names}"
            )

        self.input_columns = input_columns
        self.output_columns = output_columns

        if self.args.push_to_hub and not os.environ.get("HUGGINGFACE_TOKEN"):
            raise ValueError(
                "You need to set the `HUGGINGFACE_TOKEN` environment variable to push to the Hub"
            )

        self.push_to_hub = self.args.push_to_hub

    def build(
        self,
        **kwargs,
    ):
        MAX_ITEMS = self.args.max_items
        BATCH_SIZE = self.args.batch_size
        OUTPUT_DIR = self.args.output_dir

        if MAX_ITEMS == 0:
            raise ValueError("No items to map")

        batched = BATCH_SIZE > 1
        batch_size = (
            min(
                BATCH_SIZE,
                MAX_ITEMS,
            )
            if batched
            else None
        )

        self.dataset = self.dataset.select(range(MAX_ITEMS))

        self.dataset = self.dataset.map(
            self.generate,
            batched=batched,
            batch_size=batch_size,
            fn_kwargs={"batched": batched, **kwargs},
        )

        if OUTPUT_DIR:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self.dataset.save_to_disk(OUTPUT_DIR, keep_in_memory=False)

        if self.push_to_hub:
            if not os.environ.get("HUGGINGFACE_TOKEN"):
                raise ValueError(
                    "You need to provide a Hugging Face API token to push to the Hub"
                )
            self.dataset.push_to_hub(
                f"{self.args.task.name}-generated",
                use_auth_token=os.environ.get("HUGGINGFACE_TOKEN"),
                private=kwargs.get("private", False),
            )

    def generate(self, examples, indices=None, **kwargs):
        results = generate(examples, self.function, **kwargs)

        if self.args.save_every and indices:
            if (indices[-1] + 1) % self.args.save_every == 0:
                self.dataset.save_to_disk(self.args.output_dir, keep_in_memory=False)

        return results
