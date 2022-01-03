import typer
from datasets import load_dataset, load_metric, concatenate_datasets
from typing import List
from tqdm import tqdm
import pandas as pd
from io import StringIO

app = typer.Typer()


@app.command()
def rouge(
    data_file: str,
    prediction: str,
    reference: str = "news",
    language: str = None,
):
    metric = load_metric("rouge")
    dataset = load_dataset("json", data_files={"split": data_file}, split="split")

    if language:

        def filter_by_language(example):
            return example["language"] == language

        dataset = dataset.filter(filter_by_language)

    score = metric.compute(
        predictions=dataset[prediction], references=dataset[reference]
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


@app.command()
def combine_dataset(
    data_files: List[str],
    output_file: str = "combined_dataset.json",
    duplicated_columns: List[str] = [
        "language",
        "title",
        "news",
        "categories",
        "sources",
        "sources_text",
        "oracle_sources_text",
        "bert-base-multilingual-cased_extractive_summary",
    ],
):
    def remove_columns(dataset, columns):
        print(dataset)
        return dataset.remove_columns(
            list(set(columns).intersection(dataset.column_names))
        )

    dataset = concatenate_datasets(
        [
            load_dataset("json", data_files={"split": data_file}, split="split")
            if i == 0
            else remove_columns(
                load_dataset("json", data_files={"split": data_file}, split="split"),
                duplicated_columns,
            )
            for i, data_file in enumerate(data_files)
        ],
        axis=1,
    )
    dataset.to_json(output_file)


@app.command()
def rouge_multiple_methods(
    data_files: List[str],
    predictions: List[str] = typer.Option(None),
    reference: str = "news",
    language: str = None,
    duplicated_columns: List[str] = [
        "language",
        "title",
        "news",
        "categories",
        "sources",
        "sources_text",
        "oracle_sources_text",
    ],
):
    metric = load_metric("rouge")
    dataset = concatenate_datasets(
        [
            load_dataset("json", data_files={"split": data_file}, split="split")
            if i == 0
            else load_dataset(
                "json", data_files={"split": data_file}, split="split"
            ).remove_columns(duplicated_columns)
            for i, data_file in enumerate(data_files)
        ],
        axis=1,
    )

    if language:

        def filter_by_language(example):
            return example["language"] == language

        dataset = dataset.filter(filter_by_language)

    def score_to_dict(score):
        def per_rouge_name(s):
            return {
                "precision": round(s.mid.precision * 100, 2),
                "recall": round(s.mid.recall * 100, 2),
                "fmeasure": round(s.mid.fmeasure * 100, 2),
            }

        d = {
            "rouge1": per_rouge_name(score["rouge1"]),
            "rouge2": per_rouge_name(score["rouge2"]),
            "rougeL": per_rouge_name(score["rougeL"]),
            "rougeLsum": per_rouge_name(score["rougeLsum"]),
        }
        return {
            f"{rouge_type}_{k}": v
            for rouge_type, data in d.items()
            for k, v in data.items()
        }

    scores = {
        prediction: score_to_dict(
            metric.compute(
                predictions=dataset[prediction], references=dataset[reference]
            )
        )
        for prediction in tqdm(predictions, desc="Evaluation of each prediction...")
    }
    df = pd.DataFrame.from_dict(scores, orient="index")
    output = StringIO()
    df.to_csv(output)
    output.seek(0)
    typer.echo(output.read())
    typer.secho("Done", fg=typer.colors.GREEN, bold=True)
    return


@app.command()
def rouge_multiple_methods_multiple_languages(
    data_files: List[str],
    predictions: List[str] = typer.Option(None),
    reference: str = "news",
    languages: List[str] = ["all", "en", "de", "fr", "es", "pl", "pt", "it"],
    duplicated_columns: List[str] = [
        "language",
        "title",
        "news",
        "categories",
        "sources",
        "sources_text",
        "oracle_sources_text",
    ],
):
    metric = load_metric("rouge")
    dataset = concatenate_datasets(
        [
            load_dataset("json", data_files={"split": data_file}, split="split")
            if i == 0
            else load_dataset(
                "json", data_files={"split": data_file}, split="split"
            ).remove_columns(duplicated_columns)
            for i, data_file in enumerate(data_files)
        ],
        axis=1,
    )

    def score_to_dict(score):
        def per_rouge_name(s):
            return {
                "precision": round(s.mid.precision * 100, 2),
                "recall": round(s.mid.recall * 100, 2),
                "fmeasure": round(s.mid.fmeasure * 100, 2),
            }

        d = {
            "rouge1": per_rouge_name(score["rouge1"]),
            "rouge2": per_rouge_name(score["rouge2"]),
            "rougeL": per_rouge_name(score["rougeL"]),
            "rougeLsum": per_rouge_name(score["rougeLsum"]),
        }
        return {
            f"{rouge_type}_{k}": v
            for rouge_type, data in d.items()
            for k, v in data.items()
        }

    for language in languages:
        typer.echo(language)
        if language != "all":

            def filter_by_language(example):
                return example["language"] == language

            temp_dataset = dataset.filter(filter_by_language)
        else:
            temp_dataset = dataset

        scores = {
            prediction: score_to_dict(
                metric.compute(
                    predictions=temp_dataset[prediction],
                    references=temp_dataset[reference],
                )
            )
            for prediction in tqdm(predictions, desc="Evaluation of each prediction...")
        }
        df = pd.DataFrame.from_dict(scores, orient="index")
        output = StringIO()
        df.to_csv(output)
        output.seek(0)
        typer.echo(output.read())

    typer.secho("Done", fg=typer.colors.GREEN, bold=True)
    return