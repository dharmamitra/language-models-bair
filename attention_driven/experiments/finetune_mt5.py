from collections import OrderedDict

import torch

from datasets import DatasetDict

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, MT5ForConditionalGeneration

from attention_driven.experiments.baseline_v2 import BaselineV2Experiment
from attention_driven.data_processors import LDTibetanEnglishDataV2Processor
from attention_driven.data_processors.utils import convert_df_to_hf_dataset


__all__ = ["FinetuneMT5BaseExperiment", "FinetuneMT5LargeExperiment", "FinetuneMT5XLExperiment"]


class FinetuneMT5ExperimentBase(BaselineV2Experiment):
    MODEL_NAME = None

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """
        We don't train the tokenizer on Tibetan corpora at the moment, but this is probably something we want to do.

        https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#t5-like-span-masked-language-modeling
        """

        """
        tokenizer = AutoTokenizer.from_pretrained("buddhist-nlp/mt5-tibetan-tokenizer")
        """

        model_name = self.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return tokenizer

    def get_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        model_name = self.MODEL_NAME
        max_input_length = self.MAX_INPUT_LENGTH

        """
        # Load pretrained parameter weights
        base_model_parameter_dict = AutoModelForSeq2SeqLM.from_pretrained(model_name).state_dict()
        base_model_parameter_dict = OrderedDict(base_model_parameter_dict)  # Make `base_model_parameter_dict` modifiable

        keys_to_modify = ["shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
        pretrained_embedding_weights = {k: base_model_parameter_dict.pop(k) for k in keys_to_modify}

        # Create new model
        config = AutoConfig.from_pretrained(model_name, vocab_size=tokenizer.vocab_size + 2)
        model = MT5ForConditionalGeneration(config)

        # Load pretrained weights into new model with a slight change to embeddings
        # since we have a larger vocab size
        model.load_state_dict(base_model_parameter_dict, strict=False)
        model_parameter_dict = model.state_dict()
        with torch.no_grad():
            for weight_name, pretrained_embedding_weight in pretrained_embedding_weights.items():
                pretrained_vocab_size, hidden_dim = pretrained_embedding_weight.size()
                model_parameter_dict[weight_name][:pretrained_vocab_size, :hidden_dim].copy_(pretrained_embedding_weight)
        """

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.config.max_length = max_input_length

        return model

    # This is the exact same function as BaselineV2Experiment unless noted otherwise
    def load_data(self, tokenizer: PreTrainedTokenizer) -> DatasetDict:
        """
        This function assumes that https://github.com/Linguae-Dharmae/language-models
        has been cloned into the same root folder.
        """
        val_split_size = self.VAL_SPLIT_SIZE
        max_input_length = self.MAX_INPUT_LENGTH

        # Load our datasets from disk into HF Dataset's
        data_processor = LDTibetanEnglishDataV2Processor()

        train_dataset, test_dataset = convert_df_to_hf_dataset(data_processor())
        train_val_dataset = train_dataset.train_test_split(val_split_size, seed=42)

        dataset = DatasetDict(
            train=train_val_dataset["train"],
            val=train_val_dataset["test"],
            test=test_dataset,
        )
        print("Human readable dataset:", dataset)

        def tokenize_fn(examples):
            prefix = "translate to english: "
            tibetan_inputs = [prefix + t for t in examples["tibetan"]]
            model_inputs = tokenizer(tibetan_inputs, max_length=max_input_length, truncation=True)

            labels = tokenizer(text_target=examples["english"], max_length=max_input_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["tibetan", "english"])
        print("Model readable dataset:", tokenized_dataset)

        return tokenized_dataset


class FinetuneMT5BaseExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-base"


class FinetuneMT5LargeExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-large"


class FinetuneMT5XLExperiment(FinetuneMT5ExperimentBase):
    MODEL_NAME = "google/mt5-xl"
