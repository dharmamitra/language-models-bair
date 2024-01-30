from typing import Union
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoTokenizer, MT5ForConditionalGeneration


__all__ = [
    "MT5300MModelMixin",
    "MT5600MModelMixin",
    "MT51BModelMixin",
    "MT53BModelMixin",
    "MT513BModelMixin",
    "MADLADMT3BModelMixin",
    "MADLADMT10BModelMixin",
    "MT51BModelLoraMixin",
]


class MT5ModelMixinBase:
    MODEL_NAME: Union[None, str] = None

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        model.config.max_length = max_input_length

        return model

class MT5ModelLoraMixinbase:
    MODEL_NAME: Union[None, str] = None
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
        )
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        model.config.max_length = max_input_length
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, self.lora_config)


        return model



class MT5300MModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-small"


class MT5600MModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-base"
    MAX_INPUT_LENGTH = 1024

class MT51BModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-large"
    MAX_INPUT_LENGTH = 1024

class MT53BModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-xl"

class MADLADMT3BModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/madlad400-3b-mt"    
    MODEL_NAME = "buddhist-nlp/mitra-madlad-3b"   
    #MODEL_NAME = "results/TranslationMitraMADLADMTExperiment/checkpoint-mt/" 
    MAX_INPUT_LENGTH = 256

class MADLADMT7BModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/madlad400-7b-mt"    


class MADLADMT10BModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/madlad400-10b-mt"    


class MT513BModelMixin(MT5ModelMixinBase):
    MODEL_NAME = "google/mt5-xxl"

class MT51BModelLoraMixin(MT5ModelLoraMixinbase):
    MODEL_NAME = "google/mt5-large"
    MAX_INPUT_LENGTH = 512

