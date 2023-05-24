from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
 

__all__ = [
    "MBARTManyToOneModelMixin",
    "MBARTModelMixin",
]


class MBARTModelMixinBase:
    MODEL_NAME: Union[None, str] = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # We use a unidirectional tokenizer from Tibetan to English
        tokenizer.src_lang = "zh_CN"
        tokenizer.tgt_lang = "en_XX"

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        assert model_name, f"Must override `MODEL_NAME` attribute of {self.name}"

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id['en_XX']
        model.config.max_length = max_input_length        
        model._keys_to_ignore_on_save = []

        return model


class MBARTManyToOneModelMixin(MBARTModelMixinBase):
    MODEL_NAME = "facebook/mbart-large-50-many-to-one-mmt"

class MBARTModelMixin(MBARTModelMixinBase):
    MODEL_NAME = "facebook/mbart-large-50"

