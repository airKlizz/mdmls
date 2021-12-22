import typer
from datasets import load_dataset, load_metric
from typing import List, Dict
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)
import numpy as np
from nltk.tokenize import sent_tokenize

app = typer.Typer()


@app.command()
def run(
    train_data_files: List[str] = typer.Option(None),
    validation_data_files: List[str] = typer.Option(None),
    training_scenario: str = None,
    model_checkpoint: str = "google/mt5-small",
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    num_train_epochs: int = 8,
    learning_rate: float = 5.6e-5,
    weight_decay: float = 0.01,
    save_total_limit: int = 3,
    push_to_hub: bool = True,
):
    tokenized_datasets = load_dataset(
        "json",
        data_files={
            "train": list(train_data_files),
            "validation": list(validation_data_files),
        },
    )
    metric = load_metric("rouge")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Show the training loss with every epoch
    logging_steps = len(tokenized_datasets["train"]) // batch_size
    model_name = model_checkpoint.split("/")[-1]

    args = Seq2SeqTrainingArguments(
        output_dir=f"{model_name}-wikinewssum-{training_scenario}",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        push_to_hub=push_to_hub,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = [
            "\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(sent_tokenize(label.strip())) for label in decoded_labels
        ]
        # Compute ROUGE scores
        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    typer.secho("Start the training", fg=typer.colors.GREEN, bold=True)
    trainer.train()
    typer.secho("Evaluate the best model", fg=typer.colors.GREEN, bold=True)
    trainer.evaluate()
    typer.secho("Push the models to the hub", fg=typer.colors.GREEN, bold=True)
    trainer.push_to_hub(commit_message="Training complete", tags="summarization")
    typer.secho("Done", fg=typer.colors.GREEN, bold=True)
