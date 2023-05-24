from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from transformers import ByT5Tokenizer, AutoModelForSeq2SeqLM


__all__ = [
    "ByT5Google",
    "ByT5Sanskrit",
    "ByT5GoogleLarge",
]


class ByT5:
    MODEL_NAME: Union[None, str] = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        model_name = self.MODEL_NAME

        tokenizer = ByT5Tokenizer.from_pretrained(model_name)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.config.max_length = max_input_length

        return model


class ByT5Google(ByT5):
    MODEL_NAME = "google/byt5-base"
    MAX_INPUT_LENGTH = 512

class ByT5GoogleLarge(ByT5):
    MODEL_NAME = "google/byt5-large"
    MAX_INPUT_LENGTH = 512


class ByT5Sanskrit(ByT5):
    #MODEL_NAME = "buddhist-nlp/byt5-sanskrit"
    MODEL_NAME = "results/PretrainingByT5SanskritExperiment/5e-04/checkpoint-41000/"
    MAX_INPUT_LENGTH = 512