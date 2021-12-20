from nltk.tokenize import sent_tokenize
from datasets import load_metric
from transformers import AutoTokenizer


class Oracle:
    def __init__(self):
        self.rouge_score = load_metric("rouge")
        self.model_checkpoint = "google/mt5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

    def rouge2_precision(self, prediction, reference):
        return self.rouge_score.compute(
            predictions=["\n".join(prediction)], references=["\n".join(reference)]
        )["rouge2"].mid.precision

    def oracle_sources_text(self, example):
        text = example["sources_text"]
        reference = example["news"]
        sentences = list(set(sent_tokenize(text)))
        initial_scores = [self.rouge2_precision(sent, reference) for sent in sentences]
        sources_text = []
        sources_text_nb_tokens = 0
        for initial_score, sent in sorted(zip(initial_scores, sentences), reverse=True):
            with_sent_sources_text_nb_tokens = sources_text_nb_tokens + len(
                self.tokenizer(sent)["input_ids"]
            )
            if with_sent_sources_text_nb_tokens <= 512:
                sources_text += [sent]
                sources_text_nb_tokens = with_sent_sources_text_nb_tokens
        indexes = [sentences.index(sent) for sent in sources_text]
        sources_text = "\n".join(
            [sent for _, sent in sorted(zip(indexes, sources_text))]
        )
        example["oracle_sources_text"] = sources_text
        return example

    def add_oracle_sources_text_to_dataset(self, dataset):
        return dataset.map(self.oracle_sources_text)
