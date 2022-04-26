import typer
from .oracle import Oracle
from .extractive import Extractive, ExtractivePreAbstractive
from .abstractive import Abstractive
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

app = typer.Typer()


@app.command()
def extractive_oracle(data_file: str, output_file: str):
    dataset = load_dataset("json", data_files={"split": data_file}, split="split")
    oracle = Oracle()
    new_dataset = oracle.add_oracle_sources_text_to_dataset(dataset)
    new_dataset.to_json(output_file)
    typer.secho("Done", fg=typer.colors.GREEN, bold=True)


@app.command()
def extractive_bert(
    data_file: str,
    output_file: str,
    model_checkpoint: str = "distilbert-base-multilingual-cased",
    pre_abstractive: bool = False,
    abstractive_model_checkpoint: str = "google/mt5-small",
):
    dataset = load_dataset("json", data_files={"split": data_file}, split="split")
    ext = (
        Extractive(model_checkpoint=model_checkpoint)
        if not pre_abstractive
        else ExtractivePreAbstractive(
            extractive_model_checkpoint=model_checkpoint,
            abstractive_model_checkpoint=abstractive_model_checkpoint,
        )
    )
    typer.echo(f"Type of Extractive: {type(ext)}")
    new_dataset = ext.add_extractive_summary_to_dataset(dataset)
    new_dataset.to_json(output_file)
    typer.secho("Done", fg=typer.colors.GREEN, bold=True)


@app.command()
def abstractive_mt5(
    data_file: str,
    output_file: str,
    model_checkpoint: str,
    device: int = -1,
    sources_text: str = "oracle_sources_text",
    min_length: int = 200,
    max_length: int = 512,
    num_beams: int = 5,
    no_repeat_ngram_size: int = 3,
):
    dataset = load_dataset("json", data_files={"split": data_file}, split="split")
    abs = Abstractive(
        model_checkpoint=model_checkpoint,
        device=device,
        sources_text=sources_text,
        min_length=min_length,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    new_dataset = abs.add_abstractive_summary_to_dataset(dataset)
    new_dataset.to_json(output_file)
    typer.secho("Done", fg=typer.colors.GREEN, bold=True)


@app.command()
def tokenize(
    data_file: str,
    output_file: str,
    max_input_length: int = 512,
    max_target_length: int = 512,
    model_checkpoint: str = "google/mt5-small",
    source: str = "oracle_sources_text",
    target: str = "news",
):
    dataset = load_dataset("json", data_files={"split": data_file}, split="split")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples[source], max_length=max_input_length, truncation=True
        )
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[target], max_length=max_target_length, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.to_json(output_file)
    typer.secho("Done", fg=typer.colors.GREEN, bold=True)


@app.command()
def stats(
    data_file: str,
    output_file: str,
    model_checkpoint: str = "google/mt5-small",
    sources: str = "sources",
    sources_text: str = "sources_text",
    target: str = "news",
):
    dataset = load_dataset("json", data_files={"split": data_file}, split="split")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    rouge_score = load_metric("rouge")

    def stats_function(example):
        example_sources = example[sources]
        example_sources_text = example[sources_text]
        example_target = example[target]
        nb_of_sources = len(example_sources)
        nb_of_words_sources = len(word_tokenize(example_sources_text))
        sources_text_sent_tokenized = sent_tokenize(example_sources_text)
        nb_of_sentences_sources = len(sources_text_sent_tokenized)
        nb_of_tokens_sources = len(tokenizer(example_sources_text)["input_ids"])
        nb_of_words_target = len(word_tokenize(example_target))
        target_sent_tokenized = sent_tokenize(example_target)
        nb_of_sentences_target = len(target_sent_tokenized)
        nb_of_tokens_target = len(tokenizer(example_target)["input_ids"])

        rouge_score_sources = rouge_score.compute(
            references=["\n".join(target_sent_tokenized)],
            predictions=["\n".join(sources_text_sent_tokenized)],
        )

        if "stats" not in example.keys():
            example["stats"] = {}
        example["stats"][f"{sources_text}"] = {
            "nb": nb_of_sources,
            "nb_sentences": nb_of_sentences_sources,
            "nb_words": nb_of_words_sources,
            "nb_tokens": nb_of_tokens_sources,
            "rouge_score": rouge_score_sources,
        }
        example["stats"][f"{target}"] = {
            "nb_sentences": nb_of_sentences_target,
            "nb_words": nb_of_words_target,
            "nb_tokens": nb_of_tokens_target,
        }
        return example

    new_dataset = dataset.map(stats_function)
    new_dataset.to_json(output_file)
    typer.secho("Done", fg=typer.colors.GREEN, bold=True)


@app.command()
def print_stats(data_file: str, language: str = None):
    dataset = load_dataset("json", data_files={"split": data_file}, split="split")

    if language:

        def filter_by_language(example):
            return example["language"] == language

        dataset = dataset.filter(filter_by_language)

    stats = dataset["stats"]
    results = {}
    for stat in stats:
        for key, value in stat.items():
            if key not in results.keys():
                results[key] = {}
                for k, v in value.items():
                    if k == "rouge_score":
                        results[key]["R1_P"] = [round(v["rouge1"][1][0] * 100, 2)]
                        results[key]["R1_R"] = [round(v["rouge1"][1][1] * 100, 2)]
                        results[key]["R1_F"] = [round(v["rouge1"][1][2] * 100, 2)]
                        results[key]["R2_P"] = [round(v["rouge2"][1][0] * 100, 2)]
                        results[key]["R2_R"] = [round(v["rouge2"][1][1] * 100, 2)]
                        results[key]["R2_F"] = [round(v["rouge2"][1][2] * 100, 2)]
                        results[key]["RL_P"] = [round(v["rougeL"][1][0] * 100, 2)]
                        results[key]["RL_R"] = [round(v["rougeL"][1][1] * 100, 2)]
                        results[key]["RL_F"] = [round(v["rougeL"][1][2] * 100, 2)]
                        results[key]["RLsum_P"] = [round(v["rougeLsum"][1][0] * 100, 2)]
                        results[key]["RLsum_R"] = [round(v["rougeLsum"][1][1] * 100, 2)]
                        results[key]["RLsum_F"] = [round(v["rougeLsum"][1][2] * 100, 2)]
                    else:
                        results[key][k] = [v]
            else:
                for k, v in value.items():
                    if k == "rouge_score":
                        results[key]["R1_P"].append(round(v["rouge1"][1][0] * 100, 2))
                        results[key]["R1_R"].append(round(v["rouge1"][1][1] * 100, 2))
                        results[key]["R1_F"].append(round(v["rouge1"][1][2] * 100, 2))
                        results[key]["R2_P"].append(round(v["rouge2"][1][0] * 100, 2))
                        results[key]["R2_R"].append(round(v["rouge2"][1][1] * 100, 2))
                        results[key]["R2_F"].append(round(v["rouge2"][1][2] * 100, 2))
                        results[key]["RL_P"].append(round(v["rougeL"][1][0] * 100, 2))
                        results[key]["RL_R"].append(round(v["rougeL"][1][1] * 100, 2))
                        results[key]["RL_F"].append(round(v["rougeL"][1][2] * 100, 2))
                        results[key]["RLsum_P"].append(
                            round(v["rougeLsum"][1][0] * 100, 2)
                        )
                        results[key]["RLsum_R"].append(
                            round(v["rougeLsum"][1][1] * 100, 2)
                        )
                        results[key]["RLsum_F"].append(
                            round(v["rougeLsum"][1][2] * 100, 2)
                        )
                    else:
                        results[key][k].append(v)
    for key, value in results.items():
        print(f"===== {key.upper()} =====\n")
        for k, v in value.items():
            print(f"+++ {k} +++")
            print(f"Mean:             {round(np.mean(v), 2)}")
            print(f"99th percentile:  {round(np.percentile(v, 99), 2)}")
            print(f"95th percentile:  {round(np.percentile(v, 95), 2)}")
            print(f"75th percentile:  {round(np.percentile(v, 75), 2)}")
            print(f"50th percentile:  {round(np.percentile(v, 50), 2)}")
            print(f"25th percentile:  {round(np.percentile(v, 25), 2)}")
            print(f"5th percentile:   {round(np.percentile(v, 5), 2)}")
            print(f"1th percentile:   {round(np.percentile(v, 1), 2)}")
            print()
        print("\n\n")


if __name__ == "__main__":
    app()