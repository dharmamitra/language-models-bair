"""
A randomly selected set of pretrain span corruption examples.

DatasetDict({
    train: Dataset({
        features: ['id', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 20480000
    })
})

"""

from datasets import DatasetDict, load_dataset

from transformers import AutoTokenizer

from lme.training_dataset_utils.flores.utils import (
    tokenize_pretrain, mask_and_create_labels_for_pretrain
)


MAX_SEQ_LEN = 128
SEED = 42
DATASET_NAME = "chronbmm/sanskrit-monolingual-pretraining-corrupted"


def main():
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

    pretrain_dataset = load_dataset("chronbmm/sanskrit-monolingual-pretraining")["train"]
    print(f"Pretrain dataset\n{pretrain_dataset}\n{pretrain_dataset[0]}")

    tokenized_dataset = tokenize_pretrain(pretrain_dataset, tokenizer, MAX_SEQ_LEN)
    print(f"Tokenized dataset\n{tokenized_dataset}\n{tokenized_dataset[0]}")

    corrupted_dataset = mask_and_create_labels_for_pretrain(tokenized_dataset, tokenizer, seed=SEED)
    print(f"Corrupted dataset\n{corrupted_dataset}\n{corrupted_dataset[0]}")

    pretrain_dataset_val = load_dataset("chronbmm/sanskrit-monolingual-pretraining")["validation"]
    print(f"Pretrain dataset validation\n{pretrain_dataset_val}\n{pretrain_dataset_val[0]}")

    tokenized_dataset_val = tokenize_pretrain(pretrain_dataset_val, tokenizer, MAX_SEQ_LEN)
    print(f"Tokenized dataset validation\n{tokenized_dataset_val}\n{tokenized_dataset_val[0]}")

    corrupted_dataset_val = mask_and_create_labels_for_pretrain(tokenized_dataset_val, tokenizer, seed=SEED)
    print(f"Corrupted dataset validation\n{corrupted_dataset_val}\n{corrupted_dataset_val[0]}")

    pretrain_dataset_test = load_dataset("chronbmm/sanskrit-monolingual-pretraining")["test"]
    print(f"Pretrain dataset test\n{pretrain_dataset_test}\n{pretrain_dataset_test[0]}")

    tokenized_dataset_test = tokenize_pretrain(pretrain_dataset_test, tokenizer, MAX_SEQ_LEN)
    print(f"Tokenized dataset test\n{tokenized_dataset_test}\n{tokenized_dataset_test[0]}")

    corrupted_dataset_test = mask_and_create_labels_for_pretrain(tokenized_dataset_test, tokenizer, seed=SEED)
    print(f"Corrupted dataset test\n{corrupted_dataset_test}\n{corrupted_dataset_test[0]}")
    

    dataset_dict = DatasetDict({
        "train": corrupted_dataset,
        "val": corrupted_dataset_val,
        "test": corrupted_dataset_test,
    })

    dataset_dict.push_to_hub(DATASET_NAME)


if __name__ == "__main__":
    main()
