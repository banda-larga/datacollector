import click
from rich import Console

from data_collector.collector import DataPipeline

from data_collector.parsers import load_parser
from data_collector.functions import load_function

from data_collector.models import TaskType
from omegaconf import OmegaConf

console = Console()


@click.command()
@click.option("--file", type=str, help="The dataset file to use", required=True)
@click.option("--config", type=str, help="The config file to use", required=True)
@click.option(
    "--column-name",
    type=str,
    default="text",
    help="The name of the input column to use",
)
@click.option(
    "--output-column-name",
    type=str,
    default="output",
    help="The name of the output column to use",
)
@click.option(
    "--push-to-hub", is_flag=True, help="Whether to push the dataset to the Hub"
)
@click.option("--username", type=str, help="The username to use to push to the Hub")
@click.option("--repo", type=str, help="The repo to use to push to the Hub")
@click.option("--n-items", type=int, default=100, help="The number of items to process")
@click.option("--batch-size", type=int, default=10, help="The batch size to use")
@click.option("--num-proc", type=int, default=4, help="The number of processes to use")
def main(
    file,
    config,
    column_name,
    output_column_name,
    push_to_hub,
    username,
    repo,
    n_items,
    batch_size,
    num_proc,
):
    """
    Run the data collector pipeline.
    """

    console.print("[bold]Starting the data collector pipeline...[/bold]")

    console.print(f"Using config file: {config}")
    config = OmegaConf.load(config)

    console.print(f"Task defined: {config.task}")
    task_type = TaskType(config.task)

    console.print(f"Loading function for task: {task_type}")
    function = load_function(
        task_type=task_type,
        config=config,
    )

    output_parser = load_parser(
        task_type=task_type,
        config=config,
    )

    console.print(f"[red]Building the pipeline...[/red]")
    pipeline = DataPipeline(
        dataset=file,
        function=function,
        parser=output_parser,
        column_name=column_name,
        output_column_name=output_column_name,
        push_to_hub=push_to_hub,
    )

    console.print(f"[bold]Running the pipeline...[/bold]")
    pipeline.build(
        output_path=config.output_path,
        n_items=n_items,
        batch_size=batch_size,
        num_proc=num_proc,
        username=username,
        repo=repo,
    )
