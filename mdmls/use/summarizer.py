from nltk.tokenize import sent_tokenize
from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModel
from summarizer import Summarizer as ExtSummarizer

class Summarizer:
    def __init__(
        self,
        extractive_model: str = "distilbert-base-multilingual-cased",
        abstractive_model: str =  "airKlizz/mt5-base-wikinewssum-all-languages",
        device: int = -1,
        
    ):
        custom_config = AutoConfig.from_pretrained(extractive_model)
        custom_config.output_hidden_states = True
        custom_tokenizer = AutoTokenizer.from_pretrained(extractive_model)
        custom_model = AutoModel.from_pretrained(extractive_model, config=custom_config)
        self.extractive_summarizer = ExtSummarizer(
            custom_model=custom_model, custom_tokenizer=custom_tokenizer
        )
        self.abstractive_summarizer = pipeline(
            "summarization",
            model=abstractive_model,
            tokenizer=abstractive_model,
            device=device,
        )

    def __call__(
        self,
        text: str,
        extractive_num_sentences: int = 13,
        extractive_min_lenght: int = 60,
        extractive_only: bool = False,
        min_length: int = 200,
        max_length: int = 512,
        num_beams: int = 5,
        no_repeat_ngram_size: int = 3,
    ) -> str :
        extractive_summary = "\n".join(
            sent_tokenize(self.extractive_summarizer(text, num_sentences=extractive_num_sentences, min_length=extractive_min_lenght))
        )
        if extractive_only:
            return extractive_summary
        return "\n".join(
            sent_tokenize(
                self.abstractive_summarizer(
                    extractive_summary,
                    min_length=min_length,
                    max_length=max_length,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=True,
                )[0]["summary_text"]
            )
        )
