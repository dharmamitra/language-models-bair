from datasets import DatasetDict
from transformers.tokenization_utils import PreTrainedTokenizerBase

def tokenize_alignment(translation_dataset: DatasetDict, max_input_length: int, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    print(translation_dataset)
    def tokenize_fn(examples):
        model_inputs = examples["input"]
        labels = examples["output"]        
        model_inputs = [str(x) for x in model_inputs]
        labels = [str(x) for x in labels]
        model_inputs = tokenizer(model_inputs, max_length=max_input_length, truncation=True)

        labels = tokenizer(text_target=labels, max_length=max_input_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return translation_dataset.map(
        tokenize_fn, batched=True, desc="Tokenizing dataset", num_proc=32
    )
