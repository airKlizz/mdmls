from nltk.tokenize import sent_tokenize
from transformers import AutoConfig, AutoTokenizer, AutoModel
from summarizer import Summarizer


def distilbert_base_multilingual_cased_summary(text):
    return


class Extractive:
    def __init__(self, model_checkpoint):
        self.model_name = model_checkpoint.split("/")[-1]
        custom_config = AutoConfig.from_pretrained(model_checkpoint)
        custom_config.output_hidden_states = True
        custom_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        custom_model = AutoModel.from_pretrained(model_checkpoint, config=custom_config)
        self.model = Summarizer(
            custom_model=custom_model, custom_tokenizer=custom_tokenizer
        )

    def extractive_summary(self, example):
        text = example["sources_text"]
        example[f"{self.model_name}_extractive_summary"] = "\n".join(
            sent_tokenize(self.model(text, num_sentences=11, min_length=60))
        )
        return example

    def add_extractive_summary_to_dataset(self, dataset):
        return dataset.map(self.extractive_summary)
