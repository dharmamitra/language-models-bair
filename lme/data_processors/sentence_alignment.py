from datasets import load_dataset, DatasetDict, DownloadMode
from lme.data_processors.abstract import AbstractDataProcessor

__all__ = ["SentenceAlignmentProcessor"]

class SentenceAlignmentProcessor(AbstractDataProcessor):
    """
    Data processor for sentence alignment post-correction

    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/sentence-alignment-merged-postcorrection", use_auth_token=True, download_mode=DownloadMode.FORCE_REDOWNLOAD, ignore_verifications=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })

        return dataset