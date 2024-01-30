from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["SanskritParagraphTranslationProcessor",
           "SanskritItihasaProcessor",
           "SanskritItihasaTaggedProcessor"]

    
class SanskritParagraphTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Sanskrit Translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/skt-en-paragraph", use_auth_token=True, 
        verification_mode="no_checks")
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset

class SanskritItihasaProcessor(AbstractDataProcessor):
    """
    Data processor for Sanskrit Translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/skt-en-itihasa", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset
    
class SanskritItihasaTaggedProcessor(AbstractDataProcessor):
    """
    Data processor for Sanskrit Translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/skt-en-itihasa-tagged", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset
    