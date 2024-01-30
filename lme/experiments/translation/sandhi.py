from typing import Callable

from datasets import DatasetDict, load_dataset

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizerBase

from lme.compute_metrics_utils import get_levenshtein_compute_metrics, get_uas_las_metrics
from lme.data_processors import SandhiLongDataProcessor, SandhiSentenceDataProcessor, SandhiHackathonDataProcessor, SandhiSighumDataProcessor, SanskritDPDataProcessor, SanskritLemmatizerProcessor, SanskritLemmatizerLongProcessor

from lme.training_dataset_utils.sandhi import tokenize_sandhi_split


class TranslationSandhi:
    MAX_INPUT_LENGTH = 512  # ByT5 allows up to 1024
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")

    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_levenshtein_compute_metrics(tokenizer)

    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = SandhiLongDataProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_sandhi_split(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset

class TranslationSandhiSentence:
    MAX_INPUT_LENGTH = 256  # ByT5 allows up to 1024
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")
    
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_levenshtein_compute_metrics(tokenizer)
    
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = SandhiSentenceDataProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_sandhi_split(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset
    
class TranslationSandhiHackathon:
    MAX_INPUT_LENGTH = 256  # ByT5 allows up to 1024
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")
    
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_levenshtein_compute_metrics(tokenizer)
    
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = SandhiHackathonDataProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_sandhi_split(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset

class TranslationSandhiSighum:
    MAX_INPUT_LENGTH = 256  # ByT5 allows up to 1024
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")
    
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_levenshtein_compute_metrics(tokenizer)
    
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = SandhiSighumDataProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_sandhi_split(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset    
    
class TranslationSanskritDP:
    MAX_INPUT_LENGTH = 512  # ByT5 allows up to 1024
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")
    
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_uas_las_metrics(tokenizer)
    
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = SanskritDPDataProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_sandhi_split(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset
    

class PretrainingSanskrit:
    MAX_INPUT_LENGTH = 512  # ByT5 allows up to 1024
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH

        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")
    
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return None #get_levenshtein_compute_metrics(tokenizer)
    
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        
        return load_dataset("chronbmm/sanskrit-monolingual-pretraining-corrupted")
    
class LemmatizeSanskrit:
    MAX_INPUT_LENGTH = 512  
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH
        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")
    
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_levenshtein_compute_metrics(tokenizer)
    
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = SanskritLemmatizerProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_sandhi_split(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset    
    
class LemmatizeLongSanskrit:
    MAX_INPUT_LENGTH = 512  
    TRAINER_CLS = Seq2SeqTrainer

    def get_data_collator(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        max_input_length = self.MAX_INPUT_LENGTH
        return DataCollatorForSeq2Seq(tokenizer, max_length=max_input_length, padding="max_length")
    
    def get_compute_metrics(self, tokenizer: PreTrainedTokenizerBase) -> Callable:
        return get_levenshtein_compute_metrics(tokenizer)
    
    def get_tokenized_dataset(self, tokenizer: PreTrainedTokenizerBase, training_arguments: TrainingArguments) -> DatasetDict:
        max_input_length = self.MAX_INPUT_LENGTH

        translation_dataset = SanskritLemmatizerLongProcessor()(training_arguments)

        with training_arguments.main_process_first():
            tokenized_dataset = tokenize_sandhi_split(translation_dataset, max_input_length, tokenizer)

        return tokenized_dataset    


    