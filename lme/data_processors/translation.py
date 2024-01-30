from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["TranslationDataProcessor",
           "TibetanParagraphTranslationProcessor", 
           "EnglishTibetanParagraphTranslationProcessor"]


class TranslationDataProcessor(AbstractDataProcessor):
    """
    The original loaded dataset dict is
    DatasetDict({
        train: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 448849
        })
        validation: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 5000
        })
        test: Dataset({
            features: ['input_text', 'target_text'],
            num_rows: 5000
        })
    })

    The output dataset dict is
    DatasetDict({
        train: Dataset({
            features: ['tibetan', 'english'],
            num_rows: 448849
        })
        val: Dataset({
            features: ['tibetan', 'english'],
            num_rows: 5000
        })
        test: Dataset({
            features: ['tibetan', 'english'],
            num_rows: 5000
        })
    })
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/tib_eng_bitext", use_auth_token=True)

        dataset = dataset.rename_columns({
            "input_text": "tibetan",
            "target_text": "english",
        })

        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })

        return dataset

class TibetanParagraphTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Tibetan paragraph-level translation
    """

    def load(self) -> DatasetDict:
        base_path = "/rscratch/nehrdich/data/tib-eng/splits/"

        def load_tsv_as_dataset(file_path: str) -> Dataset:
            df = pd.read_csv(file_path, delimiter="\t", on_bad_lines="skip", engine="python")
            return Dataset.from_pandas(df)

        dataset = DatasetDict({
            "train": load_tsv_as_dataset(base_path + "train.tsv"),
            "val": load_tsv_as_dataset(base_path + "val_short.tsv"),
            "test": load_tsv_as_dataset(base_path + "test_short.tsv"),
        })

        return dataset


class EnglishTibetanParagraphTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Tibetan paragraph-level translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/tib-en-paragraph", use_auth_token=True)
        dataset = dataset.map(lambda x: {"input_text": x["target_text"], "target_text": x["input_text"]})
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset    