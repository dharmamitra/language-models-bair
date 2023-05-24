from datasets import DatasetDict
from transformers.tokenization_utils import PreTrainedTokenizerBase

def tokenize_sandhi_split(translation_dataset: DatasetDict, max_input_length: int, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    print(translation_dataset)
    def tokenize_fn(examples):
        model_inputs = tokenizer(examples["sentence"], max_length=max_input_length, truncation=True)

        labels = tokenizer(text_target=examples["unsandhied"], max_length=max_input_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return translation_dataset.map(
        tokenize_fn, batched=True, remove_columns=["sentence", "unsandhied"], desc="Tokenizing sandhi split"
    )
