from typing import Callable

from datasets import DatasetDict, load_dataset

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.compute_metrics_utils import get_translation_compute_metrics
from lme.data_processors import SanskritParagraphTranslationProcessor, SanskritItihasaProcessor, SanskritItihasaTaggedProcessor

from lme.training_dataset_utils.pali import tokenize_pali


class TranslationSanskritParagraph:
    MAX_INPUT_LENGTH = 512
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_translation_compute_metrics(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = SanskritParagraphTranslationProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_pali(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset

class TranslationSanskitItihasa:
    MAX_INPUT_LENGTH = 256
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_translation_compute_metrics(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = SanskritItihasaProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_pali(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset
    
class TranslationSanskritItihasaTagged:
    max_input_length = 256
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.max_input_length

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")
    
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_translation_compute_metrics(tokenizer)
    
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.max_input_length

        translation_dataset = SanskritItihasaTaggedProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_pali(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset