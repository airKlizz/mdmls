import typer
from datasets import load_dataset, load_metric

app = typer.Typer()


@app.command()
def rouge(
    data_file: str,
    predictions: str,
    references: str = "news",
    language: str = None,
):
    metric = load_metric("rouge")
    dataset = load_dataset("json", data_files={"split": data_file}, split="split")

    if language:

        def filter_by_language(example):
            return example["language"] == language

        dataset = dataset.filter(filter_by_language)

    score = metric.compute(
        predictions=dataset[predictions], references=dataset[references]
    )
    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_dict = {
        "precision": {
            (rn, round(score[rn].mid.precision * 100, 2)) for rn in rouge_names
        },
        "recall": {(rn, round(score[rn].mid.recall * 100, 2)) for rn in rouge_names},
        "fmeasure": {
            (rn, round(score[rn].mid.fmeasure * 100, 2)) for rn in rouge_names
        },
    }
    typer.echo(rouge_dict)
    typer.secho("Done", fg=typer.colors.GREEN, bold=True)