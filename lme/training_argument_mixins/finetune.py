import os

from transformers import TrainingArguments, Seq2SeqTrainingArguments

from lme.training_argument_mixins.utils import (
    get_deepspeed_args, calculate_batch_size_args, get_default_training_arguments
)


__all__ = [
    "MT5FinetuneArgsMixin",
    "MT513BFinetuneArgsMixin",
    "NLLBFinetuneArgsMixin",
    "MT5PretrainingArgsMixin",
    "MBARTFinetuneArgsMixin",
    "WMT21FinetuneArgsMixin",
]


class MT5FinetuneArgsMixin:
    """
    This results in 10k * 512 = 5mil examples.
    """
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )

        # !TODO: This is very poor engineering and programming practice. Sue me
        use_bf16 = os.environ.get("USE_BF16", "true")
        use_bf16 = use_bf16 == "true"

        target_total_batch_size_per_update = 2 ** 9  # 512
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=50000,
            eval_steps=400,
            save_steps=400,
            warmup_steps=0,
            weight_decay=0.01,
            gradient_accumulation_steps=gradient_accumulation_steps,
            #gradient_checkpointing=True,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size= int(2 *per_device_batch_size),
            fp16=False,
            bf16=use_bf16,
            metric_for_best_model="bleu_score",            
            #metric_for_best_model="perfect_matches",
            greater_is_better=True,
            deepspeed=get_deepspeed_args("WarmupLR"),
            **get_default_training_arguments(),
        )
    
class MT5PretrainingArgsMixin:
    """
    This results in 10k * 512 = 5mil examples.
    """
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )

        # !TODO: This is very poor engineering and programming practice. Sue me
        use_bf16 = os.environ.get("USE_BF16", "true")
        use_bf16 = use_bf16 == "true"

        target_total_batch_size_per_update = 2 ** 9  # 512
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        print("gradient_accumulation_steps", gradient_accumulation_steps)
        print("per_device_batch_size", per_device_batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=1000000,
            eval_steps=1000,
            save_steps=1000,
            warmup_steps=0,
            weight_decay=0.01,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=2 * per_device_batch_size,
            fp16=False,
            bf16=use_bf16,
            greater_is_better=False,
            deepspeed=get_deepspeed_args("WarmupLR"),
            **get_default_training_arguments(),
        )


class MT513BFinetuneArgsMixin:
    """
    This results in 10k * 512 = 5mil examples.
    """
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )

        target_total_batch_size_per_update = 2 ** 9  # 512
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=2000,
            eval_steps=100,
            save_steps=100,
            warmup_steps=0,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=2 * per_device_batch_size,
            gradient_checkpointing=True,
            fp16=False,
            bf16=True,
            metric_for_best_model="bleu_score",
            greater_is_better=True,
            deepspeed=get_deepspeed_args("WarmupLR"),
            **get_default_training_arguments(),
        )


class NLLBFinetuneArgsMixin:
    """
    This results in 10k * 512 = 5mil examples.
    """
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )

        target_total_batch_size_per_update = 2 ** 9  # 512
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=30000,
            eval_steps=400,
            save_steps=400,
            warmup_steps=1000,  # 10% of the total number of steps
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=2 * per_device_batch_size,
            fp16=True,
            metric_for_best_model="bleu_score",
            greater_is_better=True,
            deepspeed=get_deepspeed_args("WarmupDecayLR"),
            **get_default_training_arguments(),
        )

class MBARTFinetuneArgsMixin:
    """
    MBART finetuning args
    """
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )

        target_total_batch_size_per_update = 2 ** 9  # 512
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=2000,
            eval_steps=100,
            save_steps=100,
            weight_decay=0.01,
            warmup_steps=1000,  # 10% of the total number of steps
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=2 * per_device_batch_size,
            fp16=True,
            metric_for_best_model="bleu_score",
            greater_is_better=True,
            deepspeed=get_deepspeed_args("WarmupDecayLR"),
            **get_default_training_arguments(),
        )

class WMT21FinetuneArgsMixin:
    """
    This results in 10k * 512 = 5mil examples.
    """
    def get_training_arguments(self, batch_size: int, learning_rate: float) -> TrainingArguments:
        output_dir = os.path.join(
            self.experiment_class_output_dir, f"{learning_rate:.0e}"
        )

        target_total_batch_size_per_update = 2 ** 9  # 512
        gradient_accumulation_steps, per_device_batch_size = calculate_batch_size_args(target_total_batch_size_per_update, batch_size)
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=10000,
            eval_steps=100,
            save_steps=100,
            warmup_steps=1000,  # 10% of the total number of steps
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=2 * per_device_batch_size,
            fp16=True,
            metric_for_best_model="bleu_score",
            greater_is_better=True,
            deepspeed=get_deepspeed_args("WarmupDecayLR"),
            **get_default_training_arguments(),
        )
