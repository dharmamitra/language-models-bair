from datasets import load_dataset, DatasetDict

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["SandhiDataProcessor",
           "SandhiLongDataProcessor",
           "SandhiSentenceDataProcessor",
           "SandhiHackathonDataProcessor",
           "SandhiSighumDataProcessor",
           "SanskritDPDataProcessor",
           "SanskritLemmatizerProcessor",
           "SanskritLemmatizerLongProcessor"]



class SandhiDataProcessor(AbstractDataProcessor):
    """
    Sandhi data processor for sentence-level data

    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("chronbmm/sandhi-split", use_auth_token=True)

        dataset = dataset.rename_columns({
            "input_text": "sentence",
            "target_text": "unsandhied",
        })

        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })

        return dataset

class SandhiLongDataProcessor(AbstractDataProcessor):
    """
    Sandhi data processor for concatenated sentence-level data

    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("chronbmm/sandhi-split-long-pali", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation_500"],
            "test": dataset["test_500"],
        })
        return dataset

class SandhiSentenceDataProcessor(AbstractDataProcessor):
    """
    Sandhi data processor for sentence-level data
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("chronbmm/sanskrit-stemming-sentences", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation_500"],
            "test": dataset["test_500"],
        })
        return dataset

class SandhiHackathonDataProcessor(AbstractDataProcessor):
    """
    Sandhi data processor for hackathon data
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("chronbmm/sanskrit-sandhi-split-hackathon", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation_500"],
            "test": dataset["test_500"],
        })
        return dataset

class SandhiSighumDataProcessor(AbstractDataProcessor):
    """
    Sandhi data processor for SIGHUM data
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("chronbmm/sanskrit-sandhi-split-sighum", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation_500"],
            "test": dataset["test_500"],
        })
        return dataset

class SanskritDPDataProcessor(AbstractDataProcessor):
    """
    Data processor for Sanskrit Deependent Parsing
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("chronbmm/vedic-dependency-parsing", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset

class SanskritLemmatizerProcessor(AbstractDataProcessor):
    """
    Data processor for Sanskrit Lemmatizing+Morphosyntax Tagging
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("chronbmm/sanskrit-stemming-tagging-pali", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset

class SanskritLemmatizerLongProcessor(AbstractDataProcessor):
    """
    Data processor for Sanskrit Lemmatizing+Morphosyntax Tagging
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("chronbmm/sanskrit-stemming-tagging-pali-long", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation_long_500"],
            "test": dataset["test_long_500"],
        })
        return dataset
    
