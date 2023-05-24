from datasets import DatasetDict
from transformers.tokenization_utils import PreTrainedTokenizerBase

def tokenize_pali(translation_dataset: DatasetDict, max_input_length: int, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    print(translation_dataset)
    def tokenize_fn(examples):
        model_inputs = tokenizer(examples["input_text"], max_length=max_input_length, truncation=True)

        labels = tokenizer(text_target=examples["target_text"], max_length=max_input_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return translation_dataset.map(
        tokenize_fn, batched=True, desc="Tokenizing Pali-English"
    )
