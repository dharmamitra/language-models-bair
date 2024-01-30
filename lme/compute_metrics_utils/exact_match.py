from typing import Callable
from Levenshtein import distance 

import evaluate

from transformers.tokenization_utils import PreTrainedTokenizerBase


__all__ = ["get_exact_match_compute_metrics", 
           "get_levenshtein_compute_metrics",
           "get_uas_las_metrics",
           "get_sentence_alignment_score"]


def get_exact_match_compute_metrics(tokenizer: PreTrainedTokenizerBase) -> Callable:
    exact_match = evaluate.load("exact_match")

    def compute_metrics(eval_preds):
        logits, label_ids = eval_preds
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)

        return exact_match.compute(predictions=predictions, references=references)


    return compute_metrics

def get_levenshtein_compute_metrics(tokenizer: PreTrainedTokenizerBase) -> Callable:

    def compute_metrics(eval_preds):
        logits, label_ids = eval_preds
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)
        distances = []
        for sref, spred in zip(references, predictions):
            #for sref, spred in zip(ref.split(" # "), pred.split(" # ")):
            if 1:
                sref = sref.strip()
                spred = spred.strip()
                if len(sref) >= 2:
                    current_distance = distance(sref, spred)
                    print(f"ref: {sref}\n pred:{spred}\n dist: {current_distance}\n")
                    distances.append(current_distance)
        average_distance = sum(distances) / len(distances)
        sum_of_perfect_matches = sum([1 for d in distances if d == 0])
        if sum_of_perfect_matches == 0:
            perfect_matches = 0
        else:
            perfect_matches = sum_of_perfect_matches / len(distances)
        print("average_distance", average_distance)
        print("perfect_matches", perfect_matches)

        return {
            "average_distance": average_distance,
            "perfect_matches": perfect_matches,
        }

    return compute_metrics

def get_sentence_alignment_score(tokenizer: PreTrainedTokenizerBase) -> Callable:

    def compute_metrics(eval_preds):
        logits, label_ids = eval_preds
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)
        perfect_matches = 0
        total_sentences = 0
        distances = []
        for sref, spred in zip(references, predictions):
            srefs = sref.split("$")
            spreds = spred.split("$")
            total_sentences += len(srefs)
            for pred in spreds:
                if pred in srefs:
                    perfect_matches += 1

            current_distance = distance(sref, spred)
            print(f"ref: {sref}\n pred:{spred}\n dist: {current_distance}\n")
            distances.append(current_distance)
        average_distance = sum(distances) / len(distances)
        sum_of_perfect_matches = perfect_matches / total_sentences
        print("average_distance", average_distance)
        print("perfect_matches", sum_of_perfect_matches)

        return {
            "average_distance": average_distance,
            "perfect_matches": perfect_matches,
        }

    return compute_metrics

def get_uas_las_metrics(tokenizer: PreTrainedTokenizerBase) -> Callable:

    def compute_metrics(eval_preds): 
        logits, label_ids = eval_preds
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)
        uas = []
        las = []
        for reference, trans in zip(references, predictions):            
            ref_split = reference.split(" # ")
            trans_split = trans.split(" # ")
            for c_ref, c_trans in zip(ref_split, trans_split):
                c_ref = c_ref.strip()
                c_trans = c_trans.strip()            
                cref_tokens = c_ref.split(" ")
                ctrans_tokens = c_trans.split(" ")
                for ref_token, trans_token in zip(cref_tokens, ctrans_tokens):
                    if len(ref_token.split("-")) == 2 and len(trans_token.split("-")) == 2:
                        ref_label, ref_arc = ref_token.split("-")
                        trans_label, trans_arc = trans_token.split("-")
                        if ref_arc == trans_arc:
                            uas.append(1)
                        else:
                            uas.append(0)

                        if ref_token == trans_token:
                            las.append(1)
                        else:
                            las.append(0)
        if sum(las) > 0:
            las = sum(las)/len(las)
            uas = sum(uas)/len(uas)
            print("UAS", uas)
            print("LAS", las)
        return {"LAS": las,
                "UAS": uas }
    return compute_metrics