import torch

from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DataCollatorWithPadding, BatchEncoding
import os
import random
from .arguments import DataArguments
import math 

def _unpack_doc_values(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doc_examples = []
    for f in features:
        keys = list(f.keys())
        lists_per_key = len(f[keys[0]])
        for idx in range(lists_per_key):
            doc_examples.append({k: f[k][idx] for k in keys})
    return doc_examples


@dataclass
class DistilEmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    args: DataArguments = None

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        try:
            query = [f['query'] for f in features]
            f = features[0]
            neg_key = "neg" if "neg" in f else "negatives"
            pos_key = "pos" if "pos" in f else "positives"
            passage_list = []
            for f in features:
                if pos_key=="pos":
                    if 'pos_scores' in f and f['pos_scores']!=None:
                        passage_list.append((f[pos_key][0], f['pos_scores'][0]) if isinstance(f[pos_key], list) else f[pos_key])
                    else:
                        passage_list.append((f[pos_key][0], 100.0) if isinstance(f[pos_key], list) else f[pos_key])
                else:
                    passage_list.append((f[pos_key]["docs"][0], f[pos_key]['score'][0]) if isinstance(f[pos_key]["docs"], list) else f[pos_key]["docs"])
                if neg_key in f:
                    if neg_key=="neg":
                        if f[neg_key]!=None:
                            negs = [x for x in f[neg_key]]
                            if 'neg_scores' in f and f['neg_scores']!=None:
                                neg_scores = f['neg_scores']
                            else:
                                neg_scores = [-100.0]*len(negs)
                        else:
                            negs = []
                        negs = [(x,y) for x,y in zip(negs,neg_scores)]
                    else:
                        negs = f[neg_key]["docs"]
                        neg_scores = f[neg_key]["score"]
                        negs = [(x,y) for x,y in zip(negs,neg_scores)]
                else:
                    negs = []
                #negs = random.sample(negs,k=min(8,len(negs))) #@meetdoshi changed k=8 to k=2
                if negs!=[]:
                    negs = random.choices(negs, k=self.args.train_group_size-1)
                    passage_list.extend(negs)
                # else:
                #     negs = [("", -100.0)]*(self.args.train_group_size-1)
                #     passage_list.extend(negs) 
            #passage_list = [random.choice(f['pos']),random.sample(f['neg'],k=8) for f in features]
            passage = [x[0] for x in passage_list]
            score_list = [x[1] for x in passage_list]
            if 'weight' in features[0]:
                weights = [f['weight'] for f in features]
            else:
                weights = []
            # negs = random.sample(neg_list, 7*len(query))
        except:
            query = [f[0] for f in features]
            passage = [f[1] for f in features]
            weights = [1.0 for f in features]
            

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])
        q_collated = self.tokenizer(
            query,
            padding="longest",
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding="longest",
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        score_list = [s if s is not None else 0.0 for s in score_list]
        weights = [w if w is not None else 1.0 for w in weights]
        # if weights!=[]:
        #     max_val = max(weights)
        #     power = math.ceil(-math.log10(max_val)) if max_val < 1 else 0
        #     power = min(5,power-1)
        #     scale = 10 ** power
        #     weights = [w * scale for w in weights]

        return {"query": q_collated, "passage": d_collated, "teacher_score": torch.tensor(score_list), "weight": torch.tensor(weights)}
    
    
@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    args: DataArguments = None

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        try:
            query = []
            # f = features[0]
            passage_list = []
            for f in features:
                if f is None:
                    continue
                neg_key = "neg" if "neg" in f else ("negatives" if "negatives" in f else None)
                pos_key = "pos" if "pos" in f else ("positives" if "positives" in f else None)
                if pos_key=="pos":
                    passage_list.append(f[pos_key][0] if isinstance(f[pos_key], list) else f[pos_key])
                    query.append(f['query'])
                elif pos_key=='positives' and f[pos_key]!=None:
                    passage_list.append(f[pos_key]["docs"][0] if isinstance(f[pos_key]["docs"], list) else f[pos_key]["docs"])
                    query.append(f['query'])
                else:
                    continue
                if neg_key in f:
                    if neg_key=="neg":
                        negs = f[neg_key]
                    elif neg_key=="negatives" and f[neg_key]!=None:
                        negs = f[neg_key]["docs"]
                    else:
                        negs = []
                else:
                    negs = []
                #negs = random.sample(negs,k=min(8,len(negs))) #@meetdoshi changed k=8 to k=2
                if negs!=[]:
                    negs = random.choices(negs, k=self.args.train_group_size-1)
                    passage_list.extend(negs) 
                # elif self.args.train_group_size-1>0:
                #     in_batch_negatives = passage_list[:-1] #exclude pos doc
                #     if len(in_batch_negatives)==0:
                #         in_batch_negatives = [""]*(self.args.train_group_size-1)
                #     negs = random.choices(in_batch_negatives, k=self.args.train_group_size-1)
                #     passage_list.extend(negs) 
                # else:
                #     negs = [""]*(self.args.train_group_size-1)
                #     passage_list.extend(negs) 
            #passage_list = [random.choice(f['pos']),random.sample(f['neg'],k=8) for f in features]
            passage = passage_list
            if 'weight' in features[0]:
                weights = [f['weight'] for f in features]
            else:
                weights = []
            # negs = random.sample(neg_list, 7*len(query))
        except Exception as e:
            print(f'Exception occured {e} {features}')
            query = [f[0] for f in features]
            passage = [f[1] for f in features]
            weights = [1.0 for f in features]
            

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])
        for i in range(len(passage)):
            if passage[i]==None:
                # print(f'Found None in passage list {query} {passage}')
                passage[i] = ""
        weights = [w if w is not None else 1.0 for w in weights]
        q_collated = self.tokenizer(
            query,
            padding="longest",
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding="longest",
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated, "weight": torch.tensor(weights)}
    
@dataclass
class InBatchEmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        try:
            query = [f['query'] for f in features]
            #passage = [f['pos'][0] for f in features]
            f = features[0]
            pos_key = "pos" if "pos" in f else "positives"
            passage_list = []
            for f in features:
                if pos_key=="pos":
                    passage_list.append(f[pos_key][0] if isinstance(f[pos_key], list) else f[pos_key])
                else:
                    passage_list.append(f[pos_key]["docs"][0] if isinstance(f[pos_key]["docs"], list) else f[pos_key]["docs"])
            passage = passage_list
        except:
            query = [f[0] for f in features]
            passage = [f[1] for f in features]
            

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])
        q_collated = self.tokenizer(
            query,
            padding="longest",
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding="longest",
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}