from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


__all__ = [
    "WMT21ModelMixin"
]


class WMT21ModelMixinBase:
    MODEL_NAME: Union[None, str] = None
    MAX_INPUT_LENGTH: int = 256
    
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # We use a unidirectional tokenizer from Tibetan to English
        tokenizer.src_lang = "zh"
        tokenizer.tgt_lang = "en"

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        assert model_name, f"Must override `MODEL_NAME` attribute of {self.name}"

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        model.config.max_length = max_input_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["en"]
        model._keys_to_ignore_on_save = []

        return model


class WMT21ModelMixin(WMT21ModelMixinBase):
    MODEL_NAME = "facebook/wmt21-dense-24-wide-x-en"
    #MODEL_NAME = "results/TranslationChineseWMT21Experiment/checkpoint-synthetic"    


