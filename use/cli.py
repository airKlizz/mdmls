import typer
from .summarizer import Summarizer

app = typer.Typer()

@app.command()
def summarize(
    text: str,
    extractive_model: str = "distilbert-base-multilingual-cased",
    abstractive_model: str = "airKlizz/mt5-base-wikinewssum-all-languages",
    device: int = -1,
    extractive_num_sentences: int = 13,
    extractive_min_lenght: int = 60,
    extractive_only: bool = False,
    min_length: int = 200,
    max_length: int = 512,
    num_beams: int = 5,
    no_repeat_ngram_size: int = 3,
) -> str:
    summarizer = Summarizer(extractive_model, abstractive_model, device)
    return summarizer(
        text,
        extractive_num_sentences,
        abstractive_model,
        device,
        extractive_num_sentences,
        extractive_min_lenght,
        extractive_only,
        min_length,
        max_length,
        num_beams,
        no_repeat_ngram_size,
    )

if __name__ == "__main__":
    app()