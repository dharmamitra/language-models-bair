from typing import Callable

from datasets import DatasetDict, load_dataset

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.compute_metrics_utils import get_sentence_alignment_score
from lme.data_processors import SentenceAlignmentProcessor

from lme.training_dataset_utils.alignment import tokenize_alignment


class AlignmentCorrection:
    MAX_INPUT_LENGTH = 1024  # this is guesswork really
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_sentence_alignment_score(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = SentenceAlignmentProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_alignment(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset

