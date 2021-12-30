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


class ExtractivePreAbstractive:
    def __init__(self, extractive_model_checkpoint, abstractive_model_checkpoint):
        self.model_name = extractive_model_checkpoint.split("/")[-1]
        custom_config = AutoConfig.from_pretrained(extractive_model_checkpoint)
        custom_config.output_hidden_states = True
        custom_tokenizer = AutoTokenizer.from_pretrained(extractive_model_checkpoint)
        custom_model = AutoModel.from_pretrained(
            extractive_model_checkpoint, config=custom_config
        )
        self.model = Summarizer(
            custom_model=custom_model, custom_tokenizer=custom_tokenizer
        )

        self.abstractive_tokenizer = AutoTokenizer.from_pretrained(
            abstractive_model_checkpoint
        )

    def summary_tok_len(self, summary):
        return len(self.abstractive_tokenizer(summary)["input_ids"])

    def extractive_summary(self, example):
        text = example["sources_text"]
        num_sentences = 16
        last_num_sentences = num_sentences
        last_summary_len = -1
        last_summary = None
        summary = "\n".join(
            sent_tokenize(self.model(text, num_sentences=num_sentences, min_length=60))
        )
        summary_len = self.summary_tok_len(summary)
        next_num_sentences = (
            num_sentences + 1 if summary_len <= 512 else num_sentences - 1
        )
        while True:
            print(
                f"""\
---
Current summary len: {summary_len}
Prev summary len: {last_summary_len}
Nb of sentences: {num_sentences}
Next nb of sentences: {next_num_sentences}
Prev nb of sentences: {last_num_sentences}
---\
            """
            )
            last_num_sentences = num_sentences
            num_sentences = next_num_sentences
            last_summary_len = summary_len
            last_summary = summary
            summary = "\n".join(
                sent_tokenize(
                    self.model(text, num_sentences=num_sentences, min_length=60)
                )
            )
            summary_len = self.summary_tok_len(summary)
            next_num_sentences = (
                num_sentences + 1 if summary_len <= 542 else num_sentences - 1
            )
            if last_num_sentences == next_num_sentences or (
                last_summary_len == summary_len
                and (last_num_sentences - num_sentences) < 0
            ):
                break

        print(
            f"""\
---
Final summary len: {summary_len}
Prev summary len: {last_summary_len}
Final nb of sentences: {num_sentences}
Next nb of sentences: {next_num_sentences}
Prev nb of sentences: {last_num_sentences}
---\
            """
        )
        final_summary = (
            summary if (last_num_sentences - num_sentences) >= 0 else last_summary
        )
        print(f"FINAL SUMMARY LEN = {self.summary_tok_len(last_summary)}")
        example[f"{self.model_name}_extractive_summary"] = last_summary
        return example

    def add_extractive_summary_to_dataset(self, dataset):
        return dataset.map(self.extractive_summary)
