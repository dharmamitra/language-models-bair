from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["ChineseTranslationProcessor",
           "ChineseParagraphTranslationProcessor"]

    
class ChineseTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Chinese translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/mt-chn-eng-single-sentences", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset

class ChineseParagraphTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Chinese paragraph-level translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/mt-chn-eng-paragraphs", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset
