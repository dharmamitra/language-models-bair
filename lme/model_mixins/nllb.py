from typing import Union
import torch
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


__all__ = [
    "NLLB600MModelMixin",
    "NLLB1BModelMixin",
    "NLLB3BModelMixin",
]

""" def randomize_model(model):
    for module_ in model.named_modules(): 
        if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model
 """

class NLLBModelMixinBase:
    MODEL_NAME: Union[None, str] = None
    MAX_INPUT_LENGTH: int = 256
    
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # We use a unidirectional tokenizer from Tibetan to English        
        tokenizer.tgt_lang = "eng_Latn"
        #tokenizer.tgt_lang = "bod_Tibt"
        #tokenizer.tgt_lang = "zho_Hans"
        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        #assert model_name, f"Must override `MODEL_NAME` attribute of {self.name}"

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        model.config.max_length = max_input_length
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
        #model.config.forced_bos_token_id = tokenizer.lang_code_to_id["zho_Hant"]
        #model.config.forced_bos_token_id = tokenizer.lang_code_to_id["bod_Tibt"]
        #model._keys_to_ignore_on_save = []

        #model = randomize_model(model)

        return model





class NLLB600MModelMixin(NLLBModelMixinBase):
    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    

class NLLB1BModelMixin(NLLBModelMixinBase):
    MODEL_NAME = "facebook/nllb-200-1.3B"  
    MODEL_NAME = "results/TranslationMitraNLLB1BExperiment/checkpoint/"  

class NLLB3BModelMixin(NLLBModelMixinBase):
    MODEL_NAME = "facebook/nllb-200-3.3B"
    #MODEL_NAME = "results/TranslationTibetanParagraphNLLB3BExperiment/checkpoint"