from nltk.tokenize import sent_tokenize
from transformers import pipeline

from transformers.pipelines.base import KeyDataset
import tqdm


class Abstractive:
    def __init__(
        self,
        model_checkpoint,
        device: int = -1,
        sources_text: str = "oracle_sources_text",
        min_length: int = 200,
        max_length: int = 512,
        num_beams: int = 5,
        no_repeat_ngram_size: int = 3,
    ):
        self.model_name = model_checkpoint.split("/")[-1]
        self.summarizer = pipeline(
            "summarization",
            model=model_checkpoint,
            tokenizer=model_checkpoint,
            device=device,
        )
        self.sources_text = sources_text
        self.min_length = min_length
        self.max_length = max_length
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def abstractive_summary(self, example):
        example[f"{self.model_name}_abstractive_summary"] = "\n".join(
            sent_tokenize(
                self.summarizer(
                    example[self.sources_text],
                    min_length=self.min_length,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    early_stopping=True,
                )[0]["summary_text"]
            )
        )
        return example

    def add_abstractive_summary_to_dataset(self, dataset):
        return dataset.map(self.abstractive_summary)
