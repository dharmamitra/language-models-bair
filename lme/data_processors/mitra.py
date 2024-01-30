from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from lme.data_processors.abstract import AbstractDataProcessor


__all__ = ["MitraTranslationProcessor"]

    
class MitraTranslationProcessor(AbstractDataProcessor):
    """
    Data processor for the Mitra X->Eng translation task
    """
    def load(self) -> DatasetDict:
        base_path = "/rscratch/nehrdich/data/mitra/splits/"

        def load_tsv_as_dataset(file_path: str) -> Dataset:
            df = pd.read_csv(file_path, delimiter="\t", names=['input_text', 'target_text'], on_bad_lines="skip", encoding='utf-8', encoding_errors='ignore')            
            return Dataset.from_pandas(df)

        dataset = DatasetDict({
            "train": load_tsv_as_dataset(base_path + "train.tsv"),
            "val": load_tsv_as_dataset(base_path + "val_short.tsv"),
            "test": load_tsv_as_dataset(base_path + "test_short.tsv"),
        })
        return dataset

