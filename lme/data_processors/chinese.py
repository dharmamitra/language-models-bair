from datasets import load_dataset, DatasetDict
import re

from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["ChineseTranslationProcessor",
        "EnglishChineseTranslationProcessor",
           "ChineseParagraphTranslationProcessor",
           "ChineseLiteraryTranslationProcessor",
           "ChineseModernTranslationProcessor",
           "ChineseKoreanTranslationProcessor", 
           "ChinesePretrainingProcessor"]

    
class ChineseTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Chinese translation
    """

    def load(self) -> DatasetDict:
        #dataset = load_dataset("buddhist-nlp/mt-chn-eng-single-sentences", use_auth_token=True)
        dataset = load_dataset("buddhist-nlp/buddhist-zh-en-with-classical", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset


class EnglishChineseTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Chinese translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/buddhist-zh-en-with-gpt", use_auth_token=True)
        dataset = dataset.map(lambda x: {"input_text": x["target_text"], "target_text": x["input_text"]})
        def strip_alphabetic_chars(example):
            example["target_text"] = re.sub(r'[a-zA-Z ]', '', example["target_text"])
            return example

        dataset = dataset.map(strip_alphabetic_chars)
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
        dataset = load_dataset("buddhist-nlp/buddhist-zh-en", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset

class ChineseLiteraryTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Chinese novels paragraph-level translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/zh-en-literary", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset


class ChineseModernTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Chinese novels paragraph-level translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/chn-en-modern", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset


class ChineseKoreanTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for Chinese to Korean translation
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/zh-ko", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        })
        return dataset
    
class ChinesePretrainingProcessor(AbstractDataProcessor):
    """
    Data processor for Chinese pretraining
    """

    def load(self) -> DatasetDict:
        dataset = load_dataset("buddhist-nlp/daizhige-masked", use_auth_token=True)
        dataset = DatasetDict({
            "train": dataset["train"],
            "val": dataset["val"],
            "test": dataset["test"],
        })
        return dataset
