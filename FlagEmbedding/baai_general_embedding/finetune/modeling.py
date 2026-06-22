import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
from .infonce import InfoNCE
import pdb
import numpy as np
from .utils import full_contrastive_scores_and_labels
torch.use_deterministic_algorithms(True)
logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    ranking_reward: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 infonce: bool = False,
                 full_contrastive = False,
                 lora = False
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.loss_fun = InfoNCE(temperature=temperature)
        self.infonce = infonce
        self.full_contrastive = full_contrastive
        self.lora = lora
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self._keys_to_ignore_on_save = None
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self,**kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            # q_reps = bs, dim
            # p_reps = bs * grp_size, dim
            # out = bs, dim @ dim , bs * grp_size
            # out = bs, bs * grp_size
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None, weight: Tensor=None):
        batch_size = query['input_ids'].shape[0]
        group_size = passage['input_ids'].shape[0] // batch_size
        q_reps = self.encode(query)
        #p_reps = self.encode(passage)
        # sub batching for training with larger bs when a lot of negatives are present
        p_reps = []
        for i in range(group_size):
            sub_batch = {k:v[(i)*batch_size:(i+1)*batch_size] for k,v in passage.items()}
            p_reps.append(self.encode(sub_batch))
        p_reps = torch.cat(p_reps, dim=0)
        
        if self.training:
            #pdb.set_trace()
            if self.negatives_cross_device: 
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                if teacher_score != None:
                    teacher_score = self._dist_gather_tensor(teacher_score.squeeze())
                    assert len(teacher_score.shape) == 1
            if teacher_score!=None:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature
                scores = scores.view(q_reps.size(0), -1) # bs, bs*grp_size
                teacher_score = teacher_score.view(q_reps.size(0), -1).detach() # bs, grp_size
                teacher_targets = torch.nn.functional.softmax(teacher_score, dim=-1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                group_size = p_reps.size(0) // q_reps.size(0)
                target = target * group_size
                loss = 0
                # z loss term
                # z_loss = torch.logsumexp(scores,dim=-1) ** 2.0
                # scaler = 1e-4
                # loss += scaler * z_loss.mean()
                # bge m3 kd loss
                mask = torch.zeros_like(scores)
                for i in range(group_size):
                    temp_target = target + i # bs -> [0, grp_size, grp_size*2, .... grp_size*(bs-1)] + i
                    temp_scores = scores + mask # bs, bs*grp_size
                    temp_loss = torch.nn.functional.cross_entropy(temp_scores, temp_target, reduction="none")  # bs
                    if weight is not None and weight.numel()>0: #bge
                        weight = weight.view(-1,1)
                        weight = self._dist_gather_tensor(weight)
                        loss += torch.mean(teacher_targets[:, i] * temp_loss * weight)
                    else:
                        loss += torch.mean(teacher_targets[:, i] * temp_loss)
                    mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                        value=torch.finfo(scores.dtype).min)
            elif self.full_contrastive:
                scores,labels = full_contrastive_scores_and_labels(q_reps,p_reps)
                scores = scores / self.temperature
                loss = self.compute_loss(scores, labels) 

                loss *= self.world_size
            else:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature
                scores = scores.view(q_reps.size(0), -1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))
                    
                if weight is not None and weight.numel()>0:
                    weight = weight.view(-1,1)
                    weight = self._dist_gather_tensor(weight)
                    loss = torch.nn.functional.cross_entropy(scores, target, reduction='none')
                    loss = torch.mean(loss * weight.view(-1,1))
                else:
                    loss = self.compute_loss(scores, target)   
           
        else:
            if teacher_score!=None:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores / self.temperature
                scores = scores.view(q_reps.size(0), -1) # bs, bs*grp_size
                # normalize with row average
                scores /= scores.mean(dim=-1, keepdim=True)
                teacher_score = teacher_score.view(q_reps.size(0), -1).detach() # bs, grp_size
                teacher_targets = torch.nn.functional.softmax(teacher_score, dim=-1)
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                group_size = p_reps.size(0) // q_reps.size(0)
                target = target * group_size
                loss = 0
                # bge m3 kd loss
                mask = torch.zeros_like(scores)
                for i in range(group_size):
                    temp_target = target + i # bs -> [0, grp_size, grp_size*2, .... grp_size*(bs-1)] + i
                    temp_scores = scores + mask # bs, bs*grp_size
                    temp_loss = torch.nn.functional.cross_entropy(temp_scores, temp_target, reduction="none")  # bs
                    if weight is not None and weight.numel()>0: #bge
                        weight = weight.view(-1,1)
                        weight = self._dist_gather_tensor(weight)
                        loss += torch.mean(teacher_targets[:, i] * temp_loss * weight)
                    else:
                        loss += torch.mean(teacher_targets[:, i] * temp_loss)
                    mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                        value=torch.finfo(scores.dtype).min)
            elif self.full_contrastive:
                scores,labels = full_contrastive_scores_and_labels(q_reps,p_reps)
                scores = scores.view(q_reps.size(0), -1)
                
                # normalize with row average
                scores /= scores.mean(dim=-1, keepdim=True)

                scores = scores / self.temperature
                loss = self.compute_loss(scores, labels) 

                loss *= self.world_size
            else:
                scores = self.compute_similarity(q_reps, p_reps)
                scores = scores.view(q_reps.size(0), -1)
                
                # normalize with row average
                scores /= scores.mean(dim=-1, keepdim=True)

                scores = scores / self.temperature
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (p_reps.size(0) // q_reps.size(0))
                    
                if weight is not None and weight.numel()>0:
                    weight = weight.view(-1,1)
                    weight = self._dist_gather_tensor(weight)
                    loss = torch.nn.functional.cross_entropy(scores, target, reduction='none')
                    loss = torch.mean(loss * weight.view(-1,1))
                else:
                    loss = self.compute_loss(scores, target) 
            # scores = self.compute_similarity(q_reps, p_reps)
            # scores = scores / self.temperature
            # scores = scores.view(q_reps.size(0), -1)
            # target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            # target = target * (p_reps.size(0) // q_reps.size(0))
            # if weight is not None and weight.numel()>0:
            #     weight = weight.view(-1,1)
            #     weight = self._dist_gather_tensor(weight)
            #     loss = torch.nn.functional.cross_entropy(scores, target, reduction='none')
            #     loss = torch.mean(loss * weight.view(-1,1))
            # else:
            #     loss = self.compute_loss(scores, target)
            #print(loss)
        
        ranking_rewards = self.rank_metrics_from_scores(scores, target)['reciprocal_rank']

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
            ranking_reward=ranking_rewards
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def rank_metrics_from_scores(self, logits: torch.Tensor, # shape (B, N = grp_size * B)
                                pos_indices: torch.LongTensor  # shape (B,) — index of positive in each row
                                ):
        """
        Given logits (or scores) for a batch of queries vs candidates,
        and pos_indices giving which column in each row is the positive,
        compute ranking-based scores for each row.

        Returns a dict of:
        - ranks: 1-based rank of positive (1 = highest score)
        - reciprocal_ranks: 1/rank
        - normalized_rank_score: (N − rank) / (N − 1)  (so the best gets 1.0, worst gets 0.0)
        - margin_to_best_negative: (score_pos − max negative score)
        - “soft rank score”: using softmax-based scoring (see below)
        """
        B, N = logits.size()
        device = logits.device

        # Extract the positive scores (shape B,)
        pos_scores = logits[torch.arange(B, device=device), pos_indices]  # (B,)

        # Compute rank: how many candidates have score >= pos_score
        # (ties count as >=, so positive if highest gets rank = 1)
        # For each row i, we compare pos_scores[i] to logits[i, :]
        # Create a mask: which scores >= pos_score
        # We do broadcasting:
        mask_ge = logits >= pos_scores.unsqueeze(1)  # (B, N); True where candidate ≥ positive
        # Sum how many such (including itself)
        ranks = mask_ge.sum(dim=1)  # (B,), dtype = long (implicitly)
        # ranks[i] = number of candidates whose score >= positive, so rank of positive (1 = best)

        # Reciprocal rank
        reciprocal_ranks = 1.0 / ranks.float()

        # Normalized rank score in [0,1]
        # We interpret “best” = rank = 1 => normalized_score = 1.0
        # “worst” = rank = N => normalized_score = 0.0
        normalized_rank_score = (N - ranks).float() / (N - 1)

        # Margin to the top negative (how much better the positive is vs best negative)
        # First, set the positive position in logits to -inf (so it’s not considered) or mask it
        # Or simpler: compute max over negatives only
        # Mask for negatives (False at pos index)
        neg_mask = torch.ones_like(logits, dtype=torch.bool)
        neg_mask[torch.arange(B, device=device), pos_indices] = False
        # For each row, select max among logits where neg_mask is True
        neg_scores = logits.masked_fill(~neg_mask, float("-inf"))  # set positive slot to -inf
        best_neg_scores, _ = neg_scores.max(dim=1)  # (B,)
        margin_to_best_negative = pos_scores - best_neg_scores  # positive if margin > 0

        # Soft rank style: use softmax weight of pos among row
        # i.e. compute softmax over logits, then pick the probability assigned to the positive item
        probs = torch.softmax(logits, dim=1)
        soft_rank_score = probs[torch.arange(B, device=device), pos_indices]

        return {
            "rank": ranks,  # 1 = best, 2 = second, etc.
            "reciprocal_rank": reciprocal_ranks,
            "norm_rank_score": normalized_rank_score,
            "margin_to_best_neg": margin_to_best_negative,
            "soft_rank_score": soft_rank_score,
        }

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
    def load(self,checkpoint_path):
        return self.model.from_pretrained(checkpoint_path)
