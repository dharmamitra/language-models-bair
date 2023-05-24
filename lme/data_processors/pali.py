from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["PaliTranslationProcessor",
           "PaliTranslationDevanagariProcessor"]

    
class PaliTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Pali Translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/pali-english", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation_500"],
            "test": dataset["test_500"],
        })
        return dataset

class PaliTranslationDevanagariProcessor(AbstractDataProcessor):
    """
    Data processor for Pali Translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/pali-english-devanagari", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation_500"],
            "test": dataset["test_500"],
        })
        return dataset
