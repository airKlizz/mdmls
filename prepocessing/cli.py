import typer
from .oracle import Oracle
from datasets import load_dataset
from transformers import AutoTokenizer

app = typer.Typer()


@app.command()
def extractive_oracle(data_file: str, output_file: str):
    dataset = load_dataset("json", data_files={"split": data_file}, split="split")
    oracle = Oracle()
    new_dataset = oracle.add_oracle_sources_text_to_dataset(dataset)
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


if __name__ == "__main__":
    app()