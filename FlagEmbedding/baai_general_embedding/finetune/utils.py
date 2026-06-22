
import json
import torch
import torch.distributed as dist
import os

from typing import List, Union, Optional, Tuple, Mapping, Dict

def full_contrastive_scores_and_labels(
        query: torch.Tensor,
        key: torch.Tensor,
        use_all_pairs: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])

    train_n_passages = key.shape[0] // query.shape[0]
    
    labels = torch.arange(0, query.shape[0], dtype=torch.long, device=query.device)
    labels = labels * train_n_passages

    # batch_size x (batch_size x n_psg)
    qk = torch.mm(query, key.t())

    if not use_all_pairs:
        return qk, labels

    # batch_size x dim
    sliced_key = key.index_select(dim=0, index=labels)
    assert query.shape[0] == sliced_key.shape[0]

    # batch_size x batch_size
    kq = torch.mm(sliced_key, query.t())
    kq.fill_diagonal_(float('-inf'))

    qq = torch.mm(query, query.t())
    qq.fill_diagonal_(float('-inf'))

    kk = torch.mm(sliced_key, sliced_key.t())
    kk.fill_diagonal_(float('-inf'))

    scores = torch.cat([qk, kq, qq, kk], dim=-1)

    return scores, labels