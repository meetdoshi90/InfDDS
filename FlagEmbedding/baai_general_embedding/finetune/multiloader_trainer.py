import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models

import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
#dist.init_process_group(backend='nccl')

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer import (
    Trainer,
    DataLoader
)
from transformers.trainer import *
from transformers import TrainerCallback, DefaultFlowCallback
from transformers.tokenization_utils_base import BatchEncoding
from itertools import product
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.checkpoint.constants import BASE_OPTIMIZER_STATE
from deepspeed.runtime.zero.utils import is_zero_param
from deepspeed.utils import safe_get_full_optimizer_state, safe_set_full_optimizer_state, safe_set_full_fp32_param, safe_get_full_fp32_param

from datasets import load_dataset, DatasetDict, Dataset
import logging
import pdb
from tqdm import tqdm
from copy import deepcopy
import copy
from itertools import tee


import gc
from .modeling import BiEncoderModel
import wandb
import pandas as pd
from numpy.random import multinomial
wandb.init(project="multi-dds-reward-visualization", name="rewards")


logger = logging.getLogger(__name__)

def move_optimizer_state_to_cpu(opt):
    # Move any tensors in optimizer.state to CPU to avoid GPU-held buffers
    for k, v in list(opt.state.items()):
        # v is a dict (state for each param)
        if isinstance(v, dict):
            for sk, sv in list(v.items()):
                if torch.is_tensor(sv):
                    v[sk] = sv.cpu()
    # no return; opt.state modified in-place

def clear_optimizer_state(opt):
    try:
        opt.state.clear()   # remove references to state tensors
    except Exception:
        # fallback: pop keys
        for k in list(opt.state.keys()):
            opt.state.pop(k, None)

def sanity_check_param_mapping(model_clone, optimizer_clone):
    opt_params = set()
    for group in optimizer_clone.param_groups:
        opt_params.update(group['params'])

    for p in model_clone.parameters():
        assert p in opt_params, (
            f"Parameter {p.shape} not found in optimizer.param_groups!"
        )

    clone_params = set(model_clone.parameters())
    extra = opt_params - clone_params
    assert not extra, f"Optimizer has params not in model_clone: {extra}"

    print("✅ All cloned model parameters are correctly tracked by the optimizer.")


def deepspeed_sanity_check_param_mapping(model_clone, optimizer_clone):
    opt_params = set()
    for group in optimizer_clone.optimizer.bf16_groups:
        opt_params.update(group)

    for p in model_clone.parameters():
        assert p in opt_params, (
            f"Parameter {p.shape} not found in optimizer.param_groups!"
        )

    clone_params = set(model_clone.parameters())
    extra = opt_params - clone_params
    assert not extra, f"Optimizer has params not in model_clone: {extra}"

    print("✅ All cloned model parameters are correctly tracked by the optimizer.")

def check_model_sync(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_data = param.data.clone()
            checksum = torch.sum(param_data).detach()
            checksums = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
            dist.all_gather(checksums, checksum)
            for i, cs in enumerate(checksums):
                if not torch.allclose(checksum, cs, atol=1e-6):
                    raise ValueError(f"Parameter '{name}' differs between ranks. Rank 0 checksum: {checksum.item()}, Rank {i} checksum: {cs.item()}")
    if dist.get_rank() == 0:
        print("✅ Model parameters are synchronized across all GPUs.")


def cleanup_clone(model_clone, opt_clone, device_rank=None):
    try:
        if opt_clone is not None:
            # move optimizer state to CPU first (safer)
            move_optimizer_state_to_cpu(opt_clone)
            clear_optimizer_state(opt_clone)

            # if optimizer keeps any buffers as attributes, try to delete them
            try:
                opt_clone.param_groups = []
            except Exception:
                pass

        # move model to cpu (so CUDA memory for params is freed)
        if model_clone is not None:
            try:
                model_clone.to('cpu')
            except Exception:
                pass
    finally:
        # delete references and collect
        del opt_clone
        del model_clone
        gc.collect()
        # release CUDA cache (doesn't force free all allocations but helps)
        torch.cuda.empty_cache()
        if device_rank is not None:
            torch.cuda.reset_peak_memory_stats(device=torch.device(f'cuda:{device_rank}'))

def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls'):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)


def batch_to_device(batch, rank):
    """Move tensors and BatchEncoding to cuda:{rank}."""
    if rank=='cpu':
        device = torch.device(f"{rank}")
    else:
        device = torch.device(f"cuda:{rank}")
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, BatchEncoding):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, rank) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(v, rank) for v in batch)
    return batch


class MultiLoaderIterator:
    def __init__(self, output_dir, loaders, names, probs, psis, weights, datasets_num_instances, batch_size, is_train):
        self._loaders = loaders
        self._names = names
        self._probs = probs
        self._psis = psis 
        self._weights = weights
        self._avg_weights = [0.0 for _ in weights] 
        self._weight_updates = 0
        # self._iters = [l.__iter__() for l in self._loaders]
        self._iters = [None] * len(self._loaders)
        self.dataset = self._loaders[0].dataset  # trainer expects this member
        self._output_dir = output_dir
        self._count=0
        self._local_epoch = [0] * len(self._iters)
        self.datasets_num_instances = datasets_num_instances//batch_size
        self.eval_count = 0
        self.is_train = is_train
        self.base_seed=1235
    
    def get_iter(self, idx):
        if self._iters[idx] is None:
            self._iters[idx] = iter(self._loaders[idx])
        return self._iters[idx]
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.datasets_num_instances
    
    def __next__(self, key=None): #key
        #print('Next called')
        #print('Args',key)
        if key==None:
            choice = multinomial(1, self._probs).argmax() #@meetdoshi ToDo: Need to find a way to update this prob
        else:
            if key < len(self._probs):
                choice = key
            else:
                raise Exception(key, 'Key out of bound. Must be less than', len(self._probs))
        try:
            it = self.get_iter(choice)
            batch=next(it)
        except StopIteration as x:
            #print(f'Dataset {choice} has ran out of batches, reinitializing the dataset')
            ep=self._local_epoch[choice]+1
            # with open(self._output_dir + '/dataloader.'+str(os.getpid()), 'at') as pidOut:
            #     print('reshuffling: ', self._names[choice], 'local epoch: ', ep, file=pidOut)
            if hasattr(self._loaders[choice].dataset, 'set_epoch'): # For iterable datasets
                self._loaders[choice].dataset.set_epoch(ep)
            else: # For sharded datasets
                new_seed = self.base_seed + ep # Use the initial seed plus the epoch number
                self._loaders[choice].dataset = self._loaders[choice].dataset.shuffle(seed=new_seed)
            old_iter = self._iters[choice]
            if old_iter is not None:
                del old_iter
                self._iters[choice] = None
                gc.collect()
            self._iters[choice] = self._loaders[choice].__iter__()
            batch = next(self._iters[choice])
            self._local_epoch[choice] = ep
            
        if (not self.is_train) and key==None:
            self.eval_count+=1
        else:
            self.eval_count=0

        if self.eval_count>self.datasets_num_instances:
            #print('Max eval steps reached')
            self.eval_count = 0
            for c in range(len(self._iters)):
                old_iter = self._iters[c]
                if old_iter is not None:
                    del old_iter
                    self._iters[c] = None
                    gc.collect()
                self._iters[c]=self._loaders[c].__iter__()
            raise StopIteration
        # if self._count<10:
        #     with open(self._output_dir + '/dataloader.'+str(os.getpid()), 'at') as pidOut:
        #         print(self._count, ' pid: ',os.getpid(), 'choosing: ', self._names[choice], file=pidOut)
        #         print('query_ids:', batch['query'], file=pidOut)
        #     self._count = self._count+1
        return batch
    
    def reset_iter_after_dds(self):
        # Loop over all loaders/iterators and reset them.
        for i in range(len(self._iters)):
            # Increment the local epoch count for this loader.
            new_epoch = self._local_epoch[i] + 1
            # Assuming your dataset supports shuffling via set_epoch,
            # call set_epoch() to trigger a new shuffle.
            self._loaders[i].dataset.set_epoch(new_epoch)
            # Reset the iterator.
            old_iter = self._iters[i]
            if old_iter is not None:
                del old_iter
                self._iters[i] = None
                gc.collect()
            self._iters[i] = self._loaders[i].__iter__()
            # Update the local epoch.
            self._local_epoch[i] = new_epoch
        
    def custom_next(self, key=None): #key
        #print('Next called')
        #print('Args',key)
        if key==None:
            choice = multinomial(1, self._probs).argmax() #@meetdoshi ToDo: Need to find a way to update this prob
        else:
            if key < len(self._probs):
                choice = key
            else:
                raise Exception(key, 'Key out of bound. Must be less than', len(self._probs))
        try:
            it = self.get_iter(choice)
            batch=next(it)
        except StopIteration as x:
            # print(f'Dataset {choice} has ran out of batches, reinitializing the dataset')
            ep=self._local_epoch[choice]+1
            # with open(self._output_dir + '/dataloader.'+str(os.getpid()), 'at') as pidOut:
            #     print('reshuffling: ', self._names[choice], 'local epoch: ', ep, file=pidOut)
            if hasattr(self._loaders[choice].dataset, 'set_epoch'): # For iterable datasets
                self._loaders[choice].dataset.set_epoch(ep)
            else: # For sharded datasets
                new_seed = self.base_seed + ep # Use the initial seed plus the epoch number
                self._loaders[choice].dataset = self._loaders[choice].dataset.shuffle(seed=new_seed)
            old_iter = self._iters[choice]
            if old_iter is not None:
                del old_iter
                self._iters[choice] = None
                gc.collect()
            self._iters[choice] = self._loaders[choice].__iter__()
            batch = next(self._iters[choice])
            self._local_epoch[choice] = ep
            
        if (not self.is_train) and key==None:
            self.eval_count+=1
        else:
            self.eval_count=0

        if self.eval_count>=self.datasets_num_instances:
            print('Max eval steps reached')
            self.eval_count = 0
            raise StopIteration
        # if self._count<10:
        #     with open(self._output_dir + '/dataloader.'+str(os.getpid()), 'at') as pidOut:
        #         print(self._count, ' pid: ',os.getpid(), 'choosing: ', self._names[choice], file=pidOut)
        #         print('query_ids:', batch['query'], file=pidOut)
        #     self._count = self._count+1
        return batch

class AverageEvalLossCallback(TrainerCallback):
    def __init__(self):
        self.eval_losses = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        dataset_name = next((key for key in metrics if key.startswith('eval_') and key.endswith('_loss')), None)
        if dataset_name:
            self.eval_losses[dataset_name] = metrics[f'{dataset_name}']

        if len(self.eval_losses) == args.val_datasets_count:
            avg_eval_loss = sum(self.eval_losses.values()) / len(self.eval_losses)
            metrics['eval_loss'] = avg_eval_loss
            print(f"Average Evaluation Loss: {avg_eval_loss:.4f}")
            self.eval_losses.clear()

class MultiDDSUpateCallback(TrainerCallback):
    '''
    This call back function is created as part of Multi-DDS training. It is called on every step end but here
    we check whether the number of theta (model) update steps have occurred or not to perform one dds update step.
    The get_batch routine is written to extract batch from MultiDatasetHandler for train and from Accelerate data dispatcher for val set

    The DDS is calculated in a distributed manner, rewards are calculated by calculating the similarity between dev and train gradients for each
    GPU and then the rewards are copied across all of them to allow same update of the MultiDatasetHandler across GPUs.
    Since a copy of this class is passed to each GPU we update it after every DDS update so that it stays synchronized.
    '''
    def __init__(self, reference_model_doremi):
        super().__init__()
        self.reference_model_doremi = reference_model_doremi


    def get_random_item(self, iterator, k):
        # Generate a random number in the range [0, k)
        random_index = random.randint(0, k - 1)
        #print('Random index',random_index)
        
        # Iterate through the iterator up to the random index
        for index, item in enumerate(iterator):
            if index == random_index:
                return item
            
        # This part is not expected to be reached if k is correct
        raise ValueError("Iterator exhausted before reaching the random index")
    
    def get_batch(self, train_dataloader,val_dataloader, train_datasets_count, val_datasets_count, dds_batch_size=None, dds_eval_batch_size=None):
        train_batches = []
        val_batches = []
        if dds_eval_batch_size==None:
            dds_eval_batch_size = dds_batch_size
        train_itr= iter(train_dataloader)
        #print('Train itr',train_itr)
        if val_datasets_count==1:
            val_itr = iter(val_dataloader) #val_dataloader._get_iterator() #need to check if number of iterating batches have decreased in dev set
        else:
            val_itr = iter(val_dataloader)
            #print('Val itr',val_itr)
        if isinstance(train_datasets_count, list):
            iterate = train_datasets_count
        else: 
            iterate = list(range(train_datasets_count))
        for i in iterate:
            cur_batch = train_itr.custom_next(i)
            #print('New train batch', cur_batch)
            # Reduce the train batch size to val batch size for faster dds updates
            bs = int(cur_batch['query']['input_ids'].shape[0])
            grp_size = int(cur_batch['passage']['input_ids'].shape[0] // bs)
            for j in cur_batch:
                if j=='query':
                    for k in cur_batch[j]:
                        cur_batch[j][k] = cur_batch[j][k][:dds_batch_size]
                elif j=='passage':
                    for k in cur_batch[j]:
                        cur_batch[j][k] = cur_batch[j][k][:dds_batch_size*grp_size]
                elif j=='teacher_score':
                    cur_batch[j] = cur_batch[j][:dds_batch_size*grp_size]
                elif j=='weight':
                    pass
                else:
                    raise Exception(f'Key {j} not expected')
            train_batches.append(cur_batch)
            #print('Train batch',train_batches[-1]['query']['input_ids'][0])
        #print('\n'*3)
        #print('Got train batches', len(train_batches))
        for i in range(val_datasets_count):
            cur_batch = val_itr.custom_next(i)
            #print('New val batch', cur_batch)
            bs = int(cur_batch['query']['input_ids'].shape[0])
            grp_size = int(cur_batch['passage']['input_ids'].shape[0] // bs)
            for j in cur_batch:
                if j=='query':
                    for k in cur_batch[j]:
                        cur_batch[j][k] = cur_batch[j][k][:dds_eval_batch_size]
                elif j=='passage':
                    for k in cur_batch[j]:
                        cur_batch[j][k] = cur_batch[j][k][:dds_eval_batch_size*grp_size]
                elif j=='teacher_score':
                    cur_batch[j] = cur_batch[j][:dds_eval_batch_size*grp_size]
                elif j=='weight':
                    pass
                else:
                    raise Exception(f'Key {j} not expected')
            val_batches.append(cur_batch) #if making val dataloader multi dev then add key. since currently multidataset iterator is not being used for dev set
        #print('Got batches', len(train_batches), len(val_batches))
        return train_batches, val_batches


    @staticmethod
    def clone_model_with_optimizer(model, optimizer):
        model_clone = copy.deepcopy(model)
        
        inner_optimizer = optimizer.optimizer
        inner_optimizer_class = inner_optimizer.__class__
        
        param_mapping = {}
        
        for old_param, new_param in zip(model.parameters(), model_clone.parameters()):
            param_mapping[old_param] = new_param
        
        new_param_groups = []
        for old_group in inner_optimizer.param_groups:
            new_group = {k: v for k, v in old_group.items() if k != 'params'}
            new_group['params'] = [param_mapping[p] for p in old_group['params']]
            new_param_groups.append(new_group)
        
        new_inner_optimizer = inner_optimizer_class(new_param_groups)
        
        for param, new_param in param_mapping.items():
            if param in inner_optimizer.state:
                new_inner_optimizer.state[new_param] = copy.deepcopy(inner_optimizer.state[param])
        from accelerate.optimizer import AcceleratedOptimizer
        
        new_accelerated_optimizer = AcceleratedOptimizer(
            new_inner_optimizer,
            device_placement=optimizer.device_placement,
            scaler=optimizer.scaler
        )
        
        new_accelerated_optimizer.accelerator_state = copy.deepcopy(optimizer.accelerator_state)
        new_accelerated_optimizer.gradient_state = copy.deepcopy(optimizer.gradient_state)
        
        # print(f"Original optimizer state has {len(inner_optimizer.state)} parameter entries")
        # print(f"New optimizer state has {len(new_inner_optimizer.state)} parameter entries")
       
        return model_clone, new_accelerated_optimizer


    @staticmethod
    def deepspeed_clone_model_with_optimizer_fp16(model, optimizer):
        model_clone = copy.deepcopy(model)
        # print(f'Using dtype: {next(model.parameters()).dtype}')
        # model_clone = model_clone.to(dtype=next(model.parameters()).dtype)
        model_clone = model_clone.float()
        orig_name_to_param = dict(model.named_parameters())
        inner_optimizer = optimizer.optimizer
        inner_optimizer_class = inner_optimizer.__class__
        
        param_mapping = {}
        
        for old_param, new_param in zip(model.parameters(), model_clone.parameters()):
            param_mapping[old_param] = new_param
        
        new_param_groups = []
        # new_param_names = []   # parallel list of lists of names
        # print(dir(optimizer))
        # for old_group in inner_optimizer.param_groups:
        #     for p in old_group['params']:
        #         logger.info(f'1 {p.shape}')
        for old_group in inner_optimizer.param_groups:
            # logger.info(f'{old_group.keys()}')
            new_group = {k: v for k, v in old_group.items() if k != 'params'}
            # name_list = []
            # for p in old_group['params']:
            #     for nm, op in orig_name_to_param.items():
            #         logger.info(f'2 {op.shape}, {nm}')
            #         if op.shape==p.shape and torch.allclose(op.data, p.data.to(op.dtype), atol=1e-6, rtol=1e-5):
            #             name_list.append(nm)
            #             break
            #     else:
            #         raise RuntimeError(f"Could not find name for parameter {p}")
            new_group['params'] = [param_mapping[p] for p in param_mapping]
            new_param_groups.append(new_group)
            # new_param_names.append(name_list)

        base_optimizer = torch.optim.AdamW(new_param_groups)
        # print(inner_optimizer_class, type(inner_optimizer_class))
        # base_optimizer = inner_optimizer_class(new_param_groups, new_param_names)
        for optim_state, (param, new_param) in zip(inner_optimizer.state.keys(), param_mapping.items()):
            state = inner_optimizer.state[optim_state]
            new_state = {}
            for k, v in state.items():
                if torch.is_tensor(v):
                    new_state[k] = v.detach().clone().to(torch.float32).to(v.device)
                else:
                    new_state[k] = copy.deepcopy(v)
            base_optimizer.state[new_param] = new_state
            #print(param.shape , new_param.shape, optim_state.shape, base_optimizer.state[new_param])
        # new_inner_optimizer = FP16_UnfusedOptimizer(base_optimizer)
        # for group in base_optimizer.param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #             p.grad = p.grad.to(torch.float32)
                
        return model_clone, base_optimizer

    @staticmethod
    def deepspeed_clone_model_with_optimizer(model: torch.nn.Module,
                                         optimizer_wrapper) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """
        Clone a PyTorch model and clone the wrapped optimizer (e.g., DeepSpeedOptimizerWrapper).
        Returns (model_clone, cloned_base_optimizer).

        Important:
        - optimizer_wrapper may be a wrapper (Accelerate/DeepSpeed). We try to get the inner optimizer
            via attribute 'optimizer'. If not present, we treat optimizer_wrapper as the optimizer itself.
        - This will only work reliably when the optimizer state is fully present on this process (i.e., not
            partitioned by ZeRO stage 2/3 across ranks). See caveats below.
        """
        model_clone = copy.deepcopy(model)

        # 2) get inner optimizer (accelerate/deepspeed wrappers often store .optimizer)
        inner_optim = getattr(optimizer_wrapper, "optimizer", optimizer_wrapper)
        torch_optim = getattr(inner_optim, "optimizer", inner_optim)
        # inner_optim = getattr(inner_optim, "optimizer", inner_optim)
        OptimClass = torch_optim.__class__
        defaults = getattr(inner_optim, "defaults", {})

        # 3) name maps for safe lookup
        old_named = dict(model.named_parameters())
        new_named = dict(model_clone.named_parameters())

        # map old param object id -> parameter name (so we can map optimizer param back to name)
        old_param_id_to_name = {id(p): name for name, p in old_named.items()}
        # optional: map data_ptr fallback (sometimes helpful)
        old_param_data_ptr_to_name = {p.data_ptr(): name for name, p in old_named.items()}

        # 4) rebuild param_groups from inner optimizer but with new params (preserve order)
        new_param_groups = []
        # print("Inner optim params",[p.shape for old_group in inner_optim.bf16_groups for p in old_group])
        for old_group_bf16, old_group in zip(inner_optim.bf16_groups, inner_optim.param_groups):
            new_group = {}
            # copy non-params entries (lr, weight_decay, etc.)
            for k, v in old_group.items():
                if k == "params":
                    continue
                new_group[k] = copy.deepcopy(v)

            # build new params list preserving the original order of old_group['params']
            new_params_for_group = []
            # print("Old group params",[(p.shape, p.requires_grad) for p in old_group_bf16])
            for p in old_group_bf16:
                # prefer identity mapping (id). p is a Parameter object from the original model
                name = old_param_id_to_name.get(id(p))
                if name is None:
                    # fallback to data_ptr (safer than elementwise equality)
                    name = old_param_data_ptr_to_name.get(getattr(p, "data_ptr", lambda: None)())
                # if name is None:
                #     # last resort: try to match by shape+device (best effort)
                #     # this is fragile but we attempt it to avoid crash
                #     match = None
                #     for nm, np_t in old_named.items():
                #         if np_t.shape == p.shape and np_t.device == p.device:
                #             match = nm
                #             break
                #     if match is None:
                #         raise RuntimeError(f"Could not find name for param (shape={p.shape}, device={p.device}).")
                #     name = match

                # now pick the corresponding param in the cloned model
                if name not in new_named:
                    raise KeyError(f"Parameter name {name} not in cloned model.")
                new_params_for_group.append(new_named[name])

            new_group["params"] = new_params_for_group
            new_param_groups.append(new_group)

        # 5) instantiate cloned optimizer with same class + defaults
        try:
            cloned_optimizer = OptimClass(new_param_groups, **defaults)
        except Exception:
            # some optimizers accept only param_groups; fallback to passing param_groups without defaults
            cloned_optimizer = OptimClass(new_param_groups)

        # 6) copy optimizer state. Prefer load_state_dict if param_groups/order matches.
        try:
            cloned_optimizer.load_state_dict(inner_optim.optimizer.state_dict())
        except Exception as e:
            # fallback: manual per-param copy (rarely needed)
            cloned_optimizer.state = {}
            for old_p, new_p in zip(list(model.parameters()), list(model_clone.parameters())):
                full_exp_avg = safe_get_full_optimizer_state(old_p, "exp_avg")
                full_exp_avg_sq = safe_get_full_optimizer_state(old_p, "exp_avg_sq")
                # print(full_exp_avg.shape, full_exp_avg.dtype, full_exp_avg_sq.shape, full_exp_avg_sq.dtype)
                # step = safe_get_full_optimizer_state(old_p, "step")
                old_state = {
                    'step': inner_optim.state[list(inner_optim.optimizer.state.keys())[0]]['step'],
                    'exp_avg': full_exp_avg.to(new_p.dtype),
                    'exp_avg_sq': full_exp_avg_sq.to(new_p.dtype)
                }
                if old_state is None:
                    print('Old state is none')
                    raise Exception
                    continue
                new_state = {}
                for k, v in old_state.items():
                    if torch.is_tensor(v):
                        new_state[k] = v.detach().clone().to(new_p.device)
                    else:
                        new_state[k] = copy.deepcopy(v)
                cloned_optimizer.state[new_p] = new_state
            # ensure param_groups set
            cloned_optimizer.param_groups = new_param_groups

        return model_clone, cloned_optimizer

    def merge_optim_state_safe(self,optim_A, optim_B, C, reset=False):
        if len(optim_A.param_groups) != len(optim_B.param_groups):
            raise ValueError("Parameter group counts differ.")

        # for i, (pg_A, pg_B) in enumerate(zip(optim_A.param_groups, optim_B.param_groups)):
        #     print(pg_A.keys(), pg_B.keys())
        #     for key in pg_A:
        #         if key != 'params' and key in pg_B: #and pg_A[key] != pg_B[key]:
        #             raise ValueError(f"Param group {i} mismatch on '{key}': {pg_A[key]} vs {pg_B[key]}")

        for (param_A, state_A), (param_B, state_B) in zip(optim_A.state.items(), optim_B.state.items()):
            # Identify corresponding param in A
            assert param_A.shape == param_B.shape, f'Param A has shape {param_A.shape} Param B has shape {param_B.shape}'
            state_A = optim_A.state[param_A]
            for buf_name, buf_B in state_B.items():
                buf_A = state_A.get(buf_name)
                if reset==True:
                    state_A[buf_name] *= 0.0
                    continue
                if buf_A is None:
                    state_A[buf_name] = C * buf_B.clone()
                else:
                    state_A[buf_name] = buf_A + C * buf_B
                # print(buf_name,state_A[buf_name].shape)
                if state_A[buf_name].dim() > 0 and torch.any(torch.isnan(state_A[buf_name])):
                    print(buf_name, state_A[buf_name])
                    raise FloatingPointError

    @staticmethod
    def average_optimizer_state(param, state_list, weights):
        aggregated_state = {}
        keys = state_list[0].keys() 
        for key in keys:
            if torch.is_tensor(state_list[0][key]):
                agg = None
                for i, state in enumerate(state_list):
                    if key != 'step':
                        val = state[key].to(param.device).type(param.dtype)
                    else:
                        val = state[key].to(param.device).float()
                    if agg is None:
                        agg = weights[i] * val
                    else:
                        agg += weights[i] * val
                if key == 'step':
                    aggregated_state[key] = torch.tensor(float(round(agg.item())))
                else:
                    aggregated_state[key] = agg.to(param.dtype)
            else:
                try:
                    agg = sum(weights[i] * state_list[i][key] for i in range(len(state_list)))
                    aggregated_state[key] = agg
                except Exception:
                    aggregated_state[key] = state_list[0][key]
        return aggregated_state

    @staticmethod
    def deepspeed_average_optimizer_state(param, state_list, weights):
        aggregated_state = {}
        keys = state_list[0].keys() 
        for key in keys:
            if torch.is_tensor(state_list[0][key]):
                agg = None
                for i, state in enumerate(state_list):
                    if key != 'step':
                        val = state[key].to(param.device).type(torch.float32)
                    else:
                        val = state[key].to(param.device).float()
                    if agg is None:
                        agg = weights[i] * val
                    else:
                        agg += weights[i] * val
                if key == 'step':
                    aggregated_state[key] = torch.tensor(float(round(agg.item())))
                else:
                    aggregated_state[key] = agg.to(torch.float32)
            else:
                try:
                    agg = sum(weights[i] * state_list[i][key] for i in range(len(state_list)))
                    aggregated_state[key] = agg
                except Exception:
                    aggregated_state[key] = state_list[0][key]
        return aggregated_state
     
    def normalize_optim_state(self, optim_A: torch.optim.Optimizer,optim_B: torch.optim.Optimizer, Cn: float) -> None:
        """
        Divides each optimizer state value in optim_A by the normalization constant Cn.

        This normalizes any accumulated additions (such as from earlier merge operations).
        The operation is performed in place.
        
        Args:
            optim_A: The optimizer whose state will be normalized.
            Cn:     The scalar normalization constant (> 0).
        """
        if Cn <= 1e-3:
            raise ValueError("Normalization constant Cn must be non-zero", Cn)
        # for param, state in optim_B.state.items():
        #     for key, buf in state.items():
        #         print('-',key,torch.max(state[key]), state[key].shape)
        # print('#'*20)
        for param, state in optim_A.state.items():
            for key, buf in state.items():
                # print('-',key,torch.max(state[key]), state[key].shape)
                if torch.is_tensor(buf):
                    state[key] = buf.div(Cn)
                else:
                    try:
                        # In case some states are numeric (floats, ints), handle those too
                        state[key] = buf / Cn
                    except Exception:
                        # Skip unsupported types
                        pass
    
    def aggregate_model_and_optimizer(self, old_model, old_optimizer, new_model_list, new_optimizers_list, weights, offloading=False, rank=-1):
        if offloading:
            self.move_to_cpu(old_model,old_optimizer)
            weights = weights.to('cpu')
        num_models = len(new_model_list)
        if not (len(new_optimizers_list) == num_models and len(weights) == num_models):
            raise ValueError("The lengths of new_model_list, new_optimizers_list, and weights must match.")

        new_model_state = {}
        base_state = new_model_list[0].state_dict()
        for key in base_state.keys():
            agg = None
            for i in range(num_models):
                state_val = new_model_list[i].state_dict()[key]
                if agg is None:
                    agg = weights[i] * state_val
                else:
                    agg += weights[i] * state_val
            new_model_state[key] = agg
        old_model.load_state_dict(new_model_state)

        old_named_params = dict(old_model.named_parameters())
        new_optim_state_by_name = {}
        for name, param in old_named_params.items():
            state_list = []
            for i in range(num_models):
                new_named_params = dict(new_model_list[i].named_parameters())
                new_param = new_named_params.get(name)
                if new_param is None:
                    raise ValueError(f"Parameter {name} not found in new model copy {i}")
                new_optimizer = new_optimizers_list[i]
                if new_param in new_optimizer.state and new_optimizer.state[new_param]:
                    state_list.append(new_optimizer.state[new_param])
                else:
                    state_list.append({
                        'step': 0.,
                        'exp_avg': torch.zeros_like(param),
                        'exp_avg_sq': torch.zeros_like(param)
                    })
            aggregated_state = self.average_optimizer_state(param, state_list, weights)
            new_optim_state_by_name[name] = aggregated_state

        new_state = {}
        for name, state in new_optim_state_by_name.items():
            param = old_named_params[name]
            new_state[param] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}
        old_optimizer.state = new_state

        for group in old_optimizer.param_groups:
            new_params = []
            for p in group['params']:
                name_list = [n for n, op in old_named_params.items() if op.shape == p.shape and torch.allclose(op, p)]
                if not name_list:
                    raise ValueError("Parameter in optimizer group not found in old_named_params")
                new_params.append(old_named_params[name_list[0]])
            group['params'] = new_params
        
        if offloading:
            self.move_to_gpu(old_model, old_optimizer, rank=rank)
        world_size = dist.get_world_size()

        # 1) All‑reduce model parameters (in-place)
        for param in old_model.parameters():
            dist.all_reduce(param.data, op=ReduceOp.SUM)
            param.data.div_(world_size)

        # 2) All‑reduce optimizer tensors (in-place)
        for param, buf in old_optimizer.state.items():
            for _, v in buf.items():
                if isinstance(v, torch.Tensor) and v.device.type != 'cpu':
                    dist.all_reduce(v, op=ReduceOp.SUM)
                    v.div_(world_size)

        dist.barrier()
        check_model_sync(old_model)

        return old_model, old_optimizer

    def deepspeed_aggregate_model_and_optimizer_fp16(self, old_model, old_optimizer, new_model_list, new_optimizers_list, weights, offloading=False, rank=-1):
        if offloading:
            self.move_to_cpu(old_model,old_optimizer)
            weights = weights.to('cpu')
        num_models = len(new_model_list)
        if not (len(new_optimizers_list) == num_models and len(weights) == num_models):
            raise ValueError("The lengths of new_model_list, new_optimizers_list, and weights must match.")

        new_model_state = {}
        base_state = new_model_list[0].state_dict()
        for key in base_state.keys():
            agg = None
            for i in range(num_models):
                state_val = new_model_list[i].state_dict()[key]
                if agg is None:
                    agg = weights[i] * state_val
                else:
                    agg += weights[i] * state_val
            new_model_state[key] = agg
        old_model.load_state_dict(new_model_state)

        old_named_params = dict(old_model.named_parameters())
        for k in old_named_params:
            old_named_params[k] = old_named_params[k].to(torch.float32)
        new_optim_state_by_name = {}
        for name, param in old_named_params.items():
            state_list = []
            for i in range(num_models):
                new_named_params = dict(new_model_list[i].named_parameters())
                new_param = new_named_params.get(name)
                if new_param is None:
                    raise ValueError(f"Parameter {name} not found in new model copy {i}")
                new_optimizer = new_optimizers_list[i]
                if new_param in new_optimizer.state and new_optimizer.state[new_param]:
                    state_list.append(new_optimizer.state[new_param])
                else:
                    print('State not found for param', new_param.shape)
                    state_list.append({
                        'step': 0.,
                        'exp_avg': torch.zeros_like(param, dtype=torch.float32),
                        'exp_avg_sq': torch.zeros_like(param, dtype=torch.float32)
                    })
            aggregated_state = self.deepspeed_average_optimizer_state(param, state_list, weights)
            new_optim_state_by_name[name] = aggregated_state

        new_state = {}
        for name, state in new_optim_state_by_name.items():
            param = old_named_params[name]
            new_state[param] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}
        #old_optimizer.state = new_state
        for old_opt_state_key, new_opt_state_key in zip(old_optimizer.state.keys(), new_state.keys()):
            assert old_opt_state_key.shape==new_opt_state_key.shape, (old_opt_state_key.shape,new_opt_state_key.shape,)
            old_optimizer.state[old_opt_state_key] = new_state[new_opt_state_key]
        

        op_items = iter([n for n in old_named_params])
        for group in old_optimizer.param_groups:
            new_params = []
            for p in group['params']:
                n = next(op_items)
                op = old_named_params[n]
                assert p.shape == op.shape
                name_list = [n]
                if not name_list:
                    raise ValueError("Parameter in optimizer group not found in old_named_params")
                new_params.append(old_named_params[name_list[0]])
            for p1, p2 in zip(group['params'],new_params):
                with torch.no_grad():
                    p1.zero_()
                    p1.add_(p2.to(p1.device))
            #group['params'] = new_params
        
        # for n,p in old_model.named_parameters():
        #     if 'position' in n:
        #         print('1.5', p.shape, p.sum(), p.device, p.dtype)
        # for group in old_optimizer.param_groups:
        #     p = group['params'][1]
        #     state = old_optimizer.state[p]
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             print(f"State key: {k}, dtype: {v.dtype}")
        #     print(p.shape, p.sum(), p.device, p.dtype)
        #     break

        if offloading:
            self.move_to_gpu(old_model, old_optimizer, rank=rank)
        world_size = dist.get_world_size()

        # 1) All‑reduce model parameters (in-place)
        for param in old_model.parameters():
            dist.all_reduce(param.data, op=ReduceOp.SUM)
            param.data.div_(world_size)

        # 2) All‑reduce optimizer tensors (in-place)
        for param, buf in old_optimizer.state.items():
            for _, v in buf.items():
                if isinstance(v, torch.Tensor) and v.device.type != 'cpu':
                    dist.all_reduce(v, op=ReduceOp.SUM)
                    v.div_(world_size)

        # old_model = copy.deepcopy(old_model.half())
        # old_optimizer.optimizer.fp16_groups = [list(old_model.parameters())]
        # old_optimizer.optimizer.refresh_fp32_params()
        # for n,p in old_model.named_parameters():
        #     if 'position' in n:
        #         print('1.5', p.shape, p.sum(), p.device, p.dtype)
        # for group in old_optimizer.param_groups:
        #     p = group['params'][1]
        #     state = old_optimizer.state[p]
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             print(f"State key: {k}, dtype: {v.dtype}")
        #     print(p.shape, p.sum(), p.device, p.dtype)
        #     break
        # deepspeed_sanity_check_param_mapping(old_model, old_optimizer)
        dist.barrier()
        check_model_sync(old_model)

        return old_model, old_optimizer
    
    def deepspeed_aggregate_model_and_optimizer(self, old_model, old_optimizer, new_model_list, new_optimizers_list, weights, offloading=False, rank=-1):
        if offloading:
            self.move_to_cpu(old_model,old_optimizer)
            weights = weights.to('cpu')
        num_models = len(new_model_list)
        if not (len(new_optimizers_list) == num_models and len(weights) == num_models):
            raise ValueError("The lengths of new_model_list, new_optimizers_list, and weights must match.")

        new_model_state = {}
        base_state = new_model_list[0].state_dict()
        for key in base_state.keys():
            agg = None
            for i in range(num_models):
                state_val = new_model_list[i].state_dict()[key]
                if agg is None:
                    agg = weights[i] * state_val
                else:
                    agg += weights[i] * state_val
            new_model_state[key] = agg
        # print(new_model_state.keys())
        old_model.load_state_dict(new_model_state)

        old_named_params = dict(old_model.named_parameters())
        # for k in old_named_params:
        #     old_named_params[k] = old_named_params[k].to(torch.float32)
        new_optim_state_by_name = {}
        for name, param in old_named_params.items():
            state_list = []
            for i in range(num_models):
                new_named_params = dict(new_model_list[i].named_parameters())
                new_param = new_named_params.get(name)
                if new_param is None:
                    raise ValueError(f"Parameter {name} not found in new model copy {i}")
                new_optimizer = new_optimizers_list[i]
                if new_param in new_optimizer.state and new_optimizer.state[new_param]:
                    state_list.append(new_optimizer.state[new_param])
                else:
                    print('State not found for param', new_param.shape)
                    state_list.append({
                        'step': 0.,
                        'exp_avg': torch.zeros_like(param, dtype=torch.float32),
                        'exp_avg_sq': torch.zeros_like(param, dtype=torch.float32)
                    })
            aggregated_state = self.deepspeed_average_optimizer_state(param, state_list, weights)
            new_optim_state_by_name[name] = aggregated_state

        new_state = {}
        for name, state in new_optim_state_by_name.items():
            param = old_named_params[name]
            new_state[param] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}
        #old_optimizer.state = new_state
        max_steps_count = 0
        for n,p in old_model.named_parameters():
            new_state_for_param = new_state[p]
            for k, v in new_state_for_param.items():
                if torch.is_tensor(v) and k!='step':
                    # print(p.shape, v.shape, k)
                    safe_set_full_optimizer_state(p, v, k)
                elif k=='step':
                    # print('---',k,v,n)
                    max_steps_count = max(max_steps_count, v.data)
                else:
                    print('###',k,v,n)
                    print(old_optimizer.state[list(old_optimizer.state.keys())[0]]['step'])
        
        # print(old_optimizer.state[list(old_optimizer.state.keys())[0]]['step'])

        old_optimizer.state[list(old_optimizer.state.keys())[0]]['step'].add_(max_steps_count-old_optimizer.state[list(old_optimizer.state.keys())[0]]['step'].data)
        
        # print(max_steps_count,old_optimizer.state[list(old_optimizer.state.keys())[0]]['step'])

        # old_named = dict(old_model.named_parameters())
        # new_named = dict(new_model.named_parameters())
        # op_items = iter([n for n in old_named_params])
        # for group in old_optimizer.param_groups:
        #     new_params = []
        #     for p in group['params']:
        #         n = next(op_items)
        #         op = old_named_params[n]
        #         assert p.shape == op.shape
        #         name_list = [n]
        #         if not name_list:
        #             raise ValueError("Parameter in optimizer group not found in old_named_params")
        #         new_params.append(old_named_params[name_list[0]])
        #     for p1, p2 in zip(group['params'],new_params):
        #         with torch.no_grad():
        #             p1.zero_()
        #             p1.add_(p2.to(p1.device))

        if offloading:
            self.move_to_gpu(old_model, old_optimizer, rank=rank)
        world_size = dist.get_world_size()

        # 1) All‑reduce model parameters (in-place)
        for param in old_model.parameters():
            dist.all_reduce(param.data, op=ReduceOp.SUM)
            param.data.div_(world_size)

        # 2) All‑reduce optimizer tensors (in-place)
        for param, buf in old_optimizer.state.items():
            for _, v in buf.items():
                if isinstance(v, torch.Tensor) and v.device.type != 'cpu':
                    dist.all_reduce(v, op=ReduceOp.SUM)
                    v.div_(world_size)

        # old_model = copy.deepcopy(old_model.half())
        # old_optimizer.optimizer.fp16_groups = [list(old_model.parameters())]
        # old_optimizer.optimizer.refresh_fp32_params()
        # for n,p in old_model.named_parameters():
        #     if 'position' in n:
        #         print('1.5', p.shape, p.sum(), p.device, p.dtype)
        # for group in old_optimizer.param_groups:
        #     p = group['params'][1]
        #     state = old_optimizer.state[p]
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             print(f"State key: {k}, dtype: {v.dtype}")
        #     print(p.shape, p.sum(), p.device, p.dtype)
        #     break
        deepspeed_sanity_check_param_mapping(old_model, old_optimizer)
        dist.barrier()
        check_model_sync(old_model)

        return old_model, old_optimizer
    
    def move_to_cpu(self, model, optimizer=None):
        model.to('cpu')

        if optimizer!=None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()


    def move_to_gpu(self, model, optimizer=None, rank=-1):
        device = torch.device(f'cuda:{rank}')
        model.to(device)
        if optimizer!=None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        if k == 'step':
                            state[k] = v.cpu()
                        else:
                            state[k] = v.to(device)

    def on_step_end(self, args, state, control, **kwargs):

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        #----------------------------------------------------------------------------------------------------------------#
        # Cooldown sampling https://arxiv.org/pdf/2410.04579
        if args.cooldown_sampling == True:
            train_dataloader = kwargs.get('train_dataloader', None)
            if train_dataloader == None:
                print('train_dataloader not loaded from trainer callback')
                return
            
            old_weights = train_dataloader._weights

            if state.global_step >= state.max_steps//2:
                t = 1.0 # for last 50%
                new_train_datasets_weight = [pow(x,1.0/t) for x in old_weights] 
                new_train_datasets_probs = [w/sum(new_train_datasets_weight) for w in new_train_datasets_weight]
                new_train_datasets_psis = [np.log(x) for x in new_train_datasets_probs]
                train_dataloader._probs = new_train_datasets_probs
                train_dataloader._psis = new_train_datasets_psis
            else:
                t = 5.0 # for first 50%
                new_train_datasets_weight = [pow(x,1.0/t) for x in old_weights] 
                new_train_datasets_probs = [w/sum(new_train_datasets_weight) for w in new_train_datasets_weight]
                new_train_datasets_psis = [np.log(x) for x in new_train_datasets_probs]
                train_dataloader._probs = new_train_datasets_probs
                train_dataloader._psis = new_train_datasets_psis

            if rank==0:
                if state.global_step%1000==1: # to reduce logging frequency
                    print()
                    print(state.global_step)
                    for i in range(args.train_datasets_count):
                        print( f'Sampling probability for {train_dataloader._names[i]}', new_train_datasets_probs[i])
                        wandb.log({
                            f'Sampling probability for {train_dataloader._names[i]}': new_train_datasets_probs[i]
                        })
            return

        #----------------------------------------------------------------------------------------------------------------#
        # DoReMi https://arxiv.org/pdf/2305.10429 
        if args.doremi_reweighing==True:
            if args.doremi_warmup_steps>state.global_step:
                return
            if state.global_step%args.doremi_update_step==0:
                optim = kwargs.get('optimizer', None)
                proxy_model = kwargs.get('model', None)
                reference_model = self.reference_model_doremi # TODO: Update this
                assert reference_model != None, "Reference model cannot be none during doremi training"
                reference_model.to(proxy_model.model.device)
                train_dataloader = kwargs.get('train_dataloader', None)
                eval_dataloader = kwargs.get('eval_dataloader', None)
                if optim == None:
                    print('Optimizer not loaded from trainer callback')
                    return
                if proxy_model == None:
                    print('model not loaded from trainer callback')
                    return
                if train_dataloader == None:
                    print('train_dataloader not loaded from trainer callback')
                    return
                if eval_dataloader == None:
                    print('eval_dataloader not loaded from trainer callback')
                    return
                train_dataloader = train_dataloader
                eval_dataloader = eval_dataloader

                old_weights = torch.tensor(train_dataloader._probs[:])
                lambdas = [0]*args.train_datasets_count
                doremi_batches = 1
                for doremi_step_idx in range(doremi_batches):
                    train_batches, _ = self.get_batch(
                        train_dataloader,
                        eval_dataloader, 
                        args.train_datasets_count, 
                        0, # dont need val batches for doremi
                        args.doremi_batch_size)

                    for train_data_idx, train_batch in tqdm(enumerate(train_batches)): #Do this for k batches for each train domain
                        train_batch = batch_to_device(train_batch, rank=rank)
                        
                        with torch.no_grad():
                            output = proxy_model(**train_batch)
                            loss_proxy = output.loss
                            output = reference_model(**train_batch)
                            loss_reference = output.loss
                            excess_loss = torch.max(loss_proxy - loss_reference, torch.tensor(0.0).to(loss_reference.device))
                            lambdas[train_data_idx] += excess_loss

                # Gather lambdas of excess loss from gpus
                lambdas = torch.tensor(lambdas).to(proxy_model.model.device)
                gathered_lambdas = [torch.zeros_like(lambdas) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_lambdas, lambdas)

                gathered_lambdas = torch.cat([x.view(1,-1) for x in gathered_lambdas], dim=0)
                gathered_lambdas = torch.mean(gathered_lambdas, dim=0) 
                
                log_new_train_domain_weights = torch.log(old_weights) + args.doremi_reweight_eta * gathered_lambdas.cpu()
                log_new_train_domain_weights = log_new_train_domain_weights - torch.logsumexp(log_new_train_domain_weights, dim=0)
                train_domain_weights = (1.0 - args.doremi_reweight_eps) * torch.exp(log_new_train_domain_weights) + args.doremi_reweight_eps / len(log_new_train_domain_weights)
            
                # Update domain weights
                for train_data_idx in range(args.train_datasets_count):
                    if rank==0:
                        print(f'{doremi_step_idx} Excess loss between proxy and reference model for dataset {train_dataloader._names[train_data_idx]} ', gathered_lambdas[train_data_idx].data.item())
                        wandb.log({
                            f'excess_loss_trainset_{train_dataloader._names[train_data_idx]}': gathered_lambdas[train_data_idx].data.item()
                        })

                new_train_datasets_weight = train_domain_weights[:].numpy()
                new_train_datasets_probs = train_domain_weights[:].numpy()
                train_dataloader._weights = new_train_datasets_weight
                train_dataloader._probs = new_train_datasets_probs
                train_dataloader._avg_weights = np.array(train_dataloader._avg_weights) + new_train_datasets_weight
                train_dataloader._weight_updates += 1

                if rank==0:
                    print()
                    print(state.global_step)
                    for i in range(args.train_datasets_count):
                        print( f'Sampling probability for {train_dataloader._names[i]}', new_train_datasets_probs[i])
                        wandb.log({
                            f'Sampling probability for {train_dataloader._names[i]}': new_train_datasets_probs[i]
                        })
                    for i in range(args.train_datasets_count):
                        print( f'Sampling weights for {train_dataloader._names[i]}', new_train_datasets_weight[i])
                        wandb.log({
                            f'Sampling weights for {train_dataloader._names[i]}': new_train_datasets_weight[i]
                        })
                    for i in range(args.train_datasets_count):
                        print( f'Sampling avg weights for {train_dataloader._names[i]}', train_dataloader._avg_weights[i]/train_dataloader._weight_updates)
                        wandb.log({
                            f'Sampling avg weights for {train_dataloader._names[i]}': train_dataloader._avg_weights[i]/train_dataloader._weight_updates
                        })
            return
        
        #----------------------------------------------------------------------------------------------------------------#
        # DoGE https://arxiv.org/pdf/2310.15393
        # DoGE implementation is quite similar to DDS so we reuse components from our MultiDDS implementation. Sorry if the notation overlaps :)
        if args.doge_reweighing==True:
            if args.doge_warmup_steps>state.global_step:
                return
            if state.global_step%args.doge_update_step==1:

                optim = kwargs.get('optimizer', None)
                proxy_model = kwargs.get('model', None)
                train_dataloader = kwargs.get('train_dataloader', None)
                eval_dataloader = kwargs.get('eval_dataloader', None)
                lr_scheduler = kwargs.get('lr_scheduler', None)

                if optim == None:
                    print('Optimizer not loaded from trainer callback')
                    return
                if proxy_model == None:
                    print('proxy_model not loaded from trainer callback')
                    return
                if train_dataloader == None:
                    print('train_dataloader not loaded from trainer callback')
                    return
                if eval_dataloader == None:
                    print('eval_dataloader not loaded from trainer callback')
                    return
                if lr_scheduler == None:
                    print('lr_scheduler not loaded from trainer callback')
                    return
                train_dataloader = train_dataloader
                eval_dataloader = eval_dataloader
                last_lr = lr_scheduler.get_last_lr()[0] if args.doge_lr==0.0 else args.doge_lr

                rewards = [0]*args.train_datasets_count

                old_probs = torch.tensor(train_dataloader._probs[:])

                # 1 update step 
                doge_batches = 1
                for dds_step_idx in range(doge_batches):
                    train_batches, val_batches = self.get_batch(
                        train_dataloader,
                        eval_dataloader, 
                        args.train_datasets_count, 
                        args.val_datasets_count,
                        args.dds_batch_size)

                    # for each of the n train set and each of the m val set store the reward index of the train set
                    train_idxs = [i for i in range(args.train_datasets_count) for _ in range(args.val_datasets_count)]
                    reward_idx = 0
                    proxy_model.train()
                    train_data_idx = 0
                    val_data_idx = 0
                    val_batch_cache = []
                    val_batch_norm_cache = []
                    for train_batch, val_batch in tqdm(product(train_batches, val_batches)): #Do this for k batches for each train-dev combo
                        train_batch = batch_to_device(train_batch, rank=rank)
                        optim.zero_grad()
                        #print('Train batch',train_batch)
                        output = proxy_model(**train_batch)
                        loss = output.loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(proxy_model.parameters(), args.max_grad_norm)
                        
                        # store train gradients
                        train_grads = []
                        for idx, param in enumerate(proxy_model.parameters()):
                            if param.grad!=None:
                                train_grads.append(param.grad.clone())

                        # norm of train grads
                        sq_train = torch.tensor(0.0,device=proxy_model.model.device)
                        sq_train_count = 0
                        for idx in range(len(train_grads)):
                            sq_train += torch.sum(train_grads[idx] * train_grads[idx])
                            sq_train_count += train_grads[idx].numel()

                        # flat avg norm of params -> we had to do this for doge since the norm is taken across flattened parameter tensors not per parameter
                        sq_train_flat_norm_sum = 0
                        for idx in range(len(train_grads)):
                            sq_train_flat_norm_sum += (train_grads[idx]/torch.sqrt(sq_train)).sum() #model level normalized gradient
                        sq_train_flat_norm_sum_avg = sq_train_flat_norm_sum / sq_train_count
                        
                        # reset grads to zero to not affect dev grads
                        optim.zero_grad()
                        #print('Val batch',val_batch)
                        if len(val_batch_cache)<args.val_datasets_count:
                            output = proxy_model(**val_batch)
                            loss = output.loss
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(proxy_model.parameters(), args.max_grad_norm)

                            # store val grads
                            val_grads = []
                            for idx, param in enumerate(proxy_model.parameters()):
                                if param.grad!=None:
                                    val_grads.append(param.grad.clone())
                            val_batch_cache.append(val_grads)
                            # norm of val grads
                            sq_val = torch.tensor(0.0,device=proxy_model.model.device)
                            for idx in range(len(val_grads)):
                                sq_val += torch.sum(val_grads[idx] * val_grads[idx])
                            val_batch_norm_cache.append(sq_val)
                        else:
                            #print(train_data_idx, val_data_idx, 'using cached grads')
                            val_grads = val_batch_cache[val_data_idx]
                            sq_val = val_batch_norm_cache[val_data_idx]
                            
                        # doge implementation uses flat param grads, but if we are not normalizing and summing the overall contributions it is equivalent
                        # we will take care of the train normalization of scores done by doge later on -> avg_norm = train_mat.norm(dim=-1).mean(); scores = scores/(avg_norm+1e-6)
                        dot_pdt = torch.tensor(0.0,device=proxy_model.model.device)
                        for i in range(len(train_grads)):
                            dot_pdt += torch.sum(train_grads[i]*val_grads[i])
                        reward = (last_lr * dot_pdt) / (sq_train_flat_norm_sum_avg + 1e-6) #flattened normalized gradient norm avg following doge :) -> scores = scores/(avg_norm+1e-6)
                        reward = torch.clip(reward, min=last_lr*args.doge_dw_min, max=last_lr*args.doge_dw_max) #from doge, we multiply by lr here -> scores = torch.clip(scores, min=lr_t*self.dw_min, max=lr_t*self.dw_max)
                        
                        if rank==0:
                            print(f'{dds_step_idx} score between {train_dataloader._names[train_data_idx]} and {eval_dataloader._names[val_data_idx]} = ', reward.data.item())
                            wandb.log({
                                f'score_trainset_{train_dataloader._names[train_data_idx]}_valset_{eval_dataloader._names[val_data_idx]}': reward.data.item()
                            })
                        # sum rewards across each dev data
                        # Similar to what is finally done in doge implementation for the scores matrix -> scores = lr_t * scores_mat.sum(dim=-1)
                        rewards[train_idxs[reward_idx]] += reward
                        reward_idx+=1
                        optim.zero_grad()
                        
                        val_data_idx += 1
                        if val_data_idx%args.val_datasets_count==0:
                            val_data_idx = 0
                            train_data_idx += 1

                    

                rewards = torch.tensor(rewards).to(proxy_model.model.device)
                gathered_rewards = [torch.zeros_like(rewards) for _ in range(dist.get_world_size())]
                
                # Gather rewards from all GPUs to all GPUs
                dist.all_gather(gathered_rewards, rewards)


                gathered_rewards = torch.cat([x.view(1,-1) for x in gathered_rewards], dim=0)
                gathered_rewards = torch.mean(gathered_rewards, dim=0).cpu()


                if rank==0:
                    for i in range(args.train_datasets_count):
                        wandb.log({
                            f'Total_score_trainset_{train_dataloader._names[i]}': gathered_rewards[i].data.item()
                        })

                # update logic for DoGE
                dw_prev = old_probs
                log_dw_new = torch.log(dw_prev) + gathered_rewards / args.doge_bregman_coeff
                dw_new = torch.nn.functional.softmax(log_dw_new, dim=-1)
                # we dont use reweight_eps since its set to 0.0 in doge config so dw_new = dw_new
                new_train_datasets_probs = dw_new[:].numpy()
                train_dataloader._avg_weights = np.array(train_dataloader._avg_weights) + new_train_datasets_probs
                train_dataloader._weight_updates += 1
                train_dataloader._probs = new_train_datasets_probs

                if rank==0:
                    print()
                    print(state.global_step)
                    for i in range(args.train_datasets_count):
                        print( f'Sampling weights for {train_dataloader._names[i]}', new_train_datasets_probs[i])
                        wandb.log({
                            f'Sampling weights for {train_dataloader._names[i]}': new_train_datasets_probs[i]
                        })
                    for i in range(args.train_datasets_count):
                        print( f'Sampling avg weights for {train_dataloader._names[i]}', train_dataloader._avg_weights[i]/train_dataloader._weight_updates)
                        wandb.log({
                            f'Sampling avg weights for {train_dataloader._names[i]}': train_dataloader._avg_weights[i]/train_dataloader._weight_updates
                        })
                #To do: save the optimizer states and restore them afterwards if dataloader is messed up
                eval_dataloader.reset_iter_after_dds()
                torch.cuda.empty_cache()
            return


        

        #----------------------------------------------------------------------------------------------------------------#    
        # InfDDS 
        if args.inf_dds_reweighing==True:
            if args.dds_warmup_steps>state.global_step:
                return
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            offloading = False 
            sub_sample_size = args.inf_dds_subsample_size if args.inf_dds_subsample_size!=-1 else None
            dds_batches = args.dds_repeat_batches
            reward_scaler = args.infdds_rewards_scaler
            if state.global_step%args.dds_update_step==1:
                optim = kwargs.get('optimizer', None)
                model = kwargs.get('model', None)
                train_dataloader = kwargs.get('train_dataloader', None)
                eval_dataloader = kwargs.get('eval_dataloader', None)
                lr_scheduler = kwargs.get('lr_scheduler', None)
                if optim == None:
                    print('Optimizer not loaded from trainer callback')
                    return
                if model == None:
                    print('model not loaded from trainer callback')
                    return
                if train_dataloader == None:
                    print('train_dataloader not loaded from trainer callback')
                    return
                if eval_dataloader == None:
                    print('eval_dataloader not loaded from trainer callback')
                    return
                if lr_scheduler == None:
                    print('lr_scheduler not loaded from trainer callback')
                    return
                train_dataloader = train_dataloader
                eval_dataloader = eval_dataloader
                rewards = [0]*args.train_datasets_count
                old_probs = train_dataloader._probs
                old_psis = train_dataloader._psis 
                old_weights = torch.tensor(train_dataloader._probs[:])

                if sub_sample_size!=None:
                    max_dataset_count = min(args.train_datasets_count, sub_sample_size)
                else:
                    max_dataset_count = args.train_datasets_count
                if max_dataset_count==args.train_datasets_count:
                    print(f'No subsampling of rewards done for rank {rank}')
                    sampled_datasets = list(range(max_dataset_count))
                else:
                    sampled_datasets = random.sample(list(range(args.train_datasets_count)),k=max_dataset_count)
                    print(f'Sampled datasets indexes for rank {rank}: {sampled_datasets}')

                # initial model parameters theta_t and optimizer at timestep t optim_t
                if args.deepspeed:
                    model_cache, optim_cache = self.deepspeed_clone_model_with_optimizer(model, optim)
                    # print(type(optim),type(optim_cache))
                    # print(type(model),type(model_cache))
                else:
                    model_cache, optim_cache = self.clone_model_with_optimizer(model, optim)
                model_vec_cache = parameters_to_vector(model_cache.parameters())
                # print('Model vec cache', torch.mean(model_vec_cache))
                # print('Model vec cache sum', torch.sum(model_vec_cache))
                model_vec_cache *= 0.0 #zero them out
                # print('Model vec cache', torch.mean(model_vec_cache))
                self.merge_optim_state_safe(optim_cache, optim_cache, 0.0, reset=True)
                normalizing_constant = 0.0

                rewards = torch.tensor([0.0 for _ in range(max_dataset_count)])
                for dds_step_idx in range(dds_batches):
                    _, val_batches = self.get_batch(
                        train_dataloader,
                        eval_dataloader, 
                        [], 
                        args.val_datasets_count,
                        args.dds_batch_size,
                        args.inf_dds_eval_batch_size,
                        )

                    model.eval()

                    #model.train()
                    for i, _ in enumerate(sampled_datasets):
                        # print('-'*20)
                        #zero out model gradients
                        if args.deepspeed:
                            train_model_i, optim_model_i = self.deepspeed_clone_model_with_optimizer(model, optim)
                        else:
                            train_model_i, optim_model_i = self.clone_model_with_optimizer(model, optim)
                        # if rank==0:
                        #     sanity_check_param_mapping(train_model_i, optim_model_i)
                        if offloading:
                            self.move_to_gpu(train_model_i, optim_model_i, rank=rank)
                        train_steps_l = args.inf_dds_train_steps_l
                        train_batches_update, _ = self.get_batch(
                            train_dataloader,
                            eval_dataloader, 
                            [i]*train_steps_l, 
                            0,
                            args.dds_batch_size,
                            args.inf_dds_eval_batch_size,
                            )
                        train_model_i.train()
                        for j, param_group in enumerate(optim_model_i.param_groups):
                            param_group['lr'] = lr_scheduler.get_last_lr()[j]
                        for train_batch_steps_l_cur in range(train_steps_l):
                            train_batch = train_batches_update[train_batch_steps_l_cur]
                            train_batch = batch_to_device(train_batch, rank=rank)
                            optim_model_i.zero_grad()
                            # do a forward backward pass
                            output = train_model_i(**train_batch)
                            loss = output.loss
                            loss.backward()
                            # for n,p in train_model_i.named_parameters():
                            #     if hasattr(p,'grad') and p.grad!=None:
                            #         if rank==0:
                            #             print(n, p.grad.dtype)
                            #         p.grad.to(torch.float32)
                            #torch.nn.utils.clip_grad_norm_(train_model_i.parameters(), args.max_grad_norm)
                            optim_model_i.step()
                            optim_model_i.zero_grad()
                        train_model_i.eval()

                        if offloading:
                            self.move_to_cpu(train_model_i, optim_model_i)
                        dist.barrier()


                        for j, val_batch in enumerate(val_batches):
                            # get original model predictions
                            # use theta_t+k model for each dataset to calculate rewards on the validation set
                            rewards_val = []
                            with torch.no_grad():
                                val_batch = batch_to_device(val_batch, rank=rank)
                                original_model_outputs = model(**val_batch)
                                if offloading:
                                    self.move_to_gpu(train_model_i, optimizer=None, rank=rank)
                                model_i_outputs = train_model_i(**val_batch)
                                if offloading:
                                    self.move_to_cpu(train_model_i, optimizer=None)
                                # if args.ranking_rewards: # increase in ranking reward
                                #     rewards_val.append(model_i_outputs.ranking_reward.data.item()-original_model_outputs.ranking_reward.data.item())
                                # else: # decrease in loss
                                if original_model_outputs.loss.data.item() > 1e-2: # if loss is greater than a small threshold
                                    rewards_val.append((original_model_outputs.loss.data.item()-model_i_outputs.loss.data.item())/original_model_outputs.loss.data.item())
                                else:
                                    rewards_val.append(0.0)
                                # rewards_val.append(model_i_outputs.ranking_reward.mean().data.item()-original_model_outputs.ranking_reward.mean().data.item())
                                # rewards_val.append(model_i_outputs.ranking_reward.mean().data.item()-original_model_outputs.ranking_reward.mean().data.item())
                                val_batch = batch_to_device(val_batch, rank='cpu')
                            # assert len(rewards_val)==max_dataset_count
                            # rewards[i] += max(0.0, rewards_val[0])
                            if rank==0:
                                print(f'{train_dataloader._names[i]} {eval_dataloader._names[j]}',original_model_outputs.loss.data.item(),model_i_outputs.loss.data.item())
                                print(f'{train_dataloader._names[i]} {eval_dataloader._names[j]}',original_model_outputs.scores.mean().data.item(),model_i_outputs.scores.mean().data.item())
                                print(f'{train_dataloader._names[i]} {eval_dataloader._names[j]}',original_model_outputs.scores.diagonal(dim1=-2, dim2=-1).mean().data.item(),model_i_outputs.scores.diagonal(dim1=-2, dim2=-1).mean().data.item(), original_model_outputs.scores.shape,  model_i_outputs.scores.shape)
                                mask = ~torch.eye(original_model_outputs.scores.shape[-1], dtype=bool, device=original_model_outputs.scores.device)
                                print(f'{train_dataloader._names[i]} {eval_dataloader._names[j]}',original_model_outputs.scores[mask].mean().data.item(),model_i_outputs.scores[mask].mean().data.item())
                                # print(f'{train_dataloader._names[i]} {eval_dataloader._names[j]}',original_model_outputs.ranking_reward.mean().data.item(), model_i_outputs.ranking_reward.mean().data.item(), original_model_outputs.ranking_reward.shape,  model_i_outputs.ranking_reward.shape)

                            # print(f'{train_dataloader._names[i]} {eval_dataloader._names[j]}',original_model_outputs.ranking_reward.data.item(),model_i_outputs.ranking_reward.data.item())
                            rewards[i] += rewards_val[0]
                            # if rank==0:
                            #     print(f'Rewards for val dataset {j} {eval_dataloader._names[j]} when trained on train dataset {i} {train_dataloader._names[i]}', rewards_val[0])
                        
                        gathered_rewards = [torch.zeros_like(rewards) for _ in range(dist.get_world_size())]
                        # Gather rewards from all GPUs to all GPUs
                        dist.all_gather(gathered_rewards, torch.tensor(rewards[:]).to(f'cuda:{rank}'))

                        gathered_rewards = torch.cat([x.view(1,-1) for x in gathered_rewards], dim=0)
                        #print(gathered_rewards)
                        gathered_rewards = torch.mean(gathered_rewards, dim=0) # ideally this should be mean
                        
                        model_weight_scaler = math.exp(gathered_rewards[i].data.item())
                        if not (-1e-3 <= model_weight_scaler <= 1e-3): #normalizing constant can cause weights to blow up
                            if rank==0:
                                print(f'Scaling model {i} trained on {train_dataloader._names[i]} with model weight scaler {model_weight_scaler}')
                            new_model_vec_cache = parameters_to_vector(train_model_i.parameters())
                            model_vec_cache += model_weight_scaler * new_model_vec_cache
                            normalizing_constant += model_weight_scaler
                            self.merge_optim_state_safe(optim_cache, optim_model_i, model_weight_scaler)

                        train_batch = batch_to_device(train_batch, rank='cpu')
                        cleanup_clone(train_model_i, optim_model_i, device_rank=rank)
                        del train_model_i, optim_model_i, train_batch, val_batch
                        gc.collect()
                        torch.cuda.empty_cache()
                        

                    # update global step and lr
                    state.global_step += train_steps_l
                    for _ in range(train_steps_l):
                        lr_scheduler.step()

                rewards /= dds_batches * args.val_datasets_count
                #print('Rewards', rewards)

                new_psis = copy.deepcopy(old_psis)
                rewards = torch.tensor(rewards).to(f'cuda:{rank}')
                gathered_rewards = [torch.zeros_like(rewards) for _ in range(dist.get_world_size())]
                
                # Gather rewards from all GPUs to all GPUs
                dist.all_gather(gathered_rewards, rewards)

                gathered_rewards = torch.cat([x.view(1,-1) for x in gathered_rewards], dim=0)
                #print(gathered_rewards)
                gathered_rewards = torch.mean(gathered_rewards, dim=0) # ideally this should be mean
                gathered_rewards *= reward_scaler
                # gathered_rewards = torch.clamp(gathered_rewards, min=0.0)
                

                if rank==0:
                    print('Gathered rewards')
                    print(gathered_rewards)

                if torch.all(gathered_rewards < 1e-3):
                    # If all rewards are less than an eps, use a uniform distribution
                    normalized_rewards = torch.full_like(gathered_rewards, 1.0 / gathered_rewards.numel())
                else:
                    # Replace small rewards with -inf
                    normalized_rewards = torch.where(gathered_rewards < 1e-3, torch.tensor(-float('inf'), device=gathered_rewards.device), gathered_rewards)
                    normalizing_temperature = 0.02
                    normalized_rewards = normalized_rewards/normalizing_temperature
                    normalized_rewards -= torch.max(normalized_rewards)
                    normalized_rewards = torch.softmax(normalized_rewards, dim=0)
                #normalized_rewards = torch.zeros_like(gathered_rewards).to(gathered_rewards.device)
                #normalized_rewards[torch.argmax(normalized_rewards)] = 1.0
                if rank==0:
                    print('Normalized gathered weights')
                    print(normalized_rewards)
                # update the model theta_t at timestep t to take into account model souping of models trained on each dataset
                weights = normalized_rewards
                # print('-'*30)
                # for n,p in model.named_parameters():
                #     if 'word_embedding' in n  and rank==0:
                #         print('1', p.shape, p.clone().float().sum(), p.device, p.dtype)
                #         break
                # for group in optim.param_groups:
                #     if rank==0:
                #         p = group['params'][0]
                #         print(p.shape, p.sum(), p.device, p.dtype)
                #         break
                # normalizing_constant /= (dds_batches * args.val_datasets_count)
                if normalizing_constant > 1e-3: # if overall rewards are negative then dont need to do reptile update
                    # Normalize model weights
                    # normalizing_constant = (gathered_rewards.sum() * dds_batches * args.val_datasets_count) / (reward_scaler) # since we didnt use it before aswell
                    model_vec_cache /= normalizing_constant
                    print('Normalizing constant', normalizing_constant)
                    self.normalize_optim_state(optim_cache,optim,normalizing_constant)
                    vector_to_parameters(model_vec_cache, model_cache.parameters())
                    if args.deepspeed:
                        self.deepspeed_aggregate_model_and_optimizer(model, optim, [model_cache], [optim_cache], torch.tensor([1.0]).to(weights.device), offloading=offloading, rank=rank)
                    else:
                        self.aggregate_model_and_optimizer(model, optim, [model_cache], [optim_cache], torch.tensor([1.0]).to(weights.device), offloading=offloading, rank=rank)
                else:
                    print(f'Not updating the model w/ reptile at step {state.global_step}', normalizing_constant)

                del model_cache, optim_cache

                if rank==0:
                    for i, idx in enumerate(sampled_datasets):
                        wandb.log({
                            f'Total_normalized_reward_trainset_{train_dataloader._names[idx]}': normalized_rewards[i].data.item()
                        })
                        wandb.log({
                            f'Total_reward_trainset_{train_dataloader._names[idx]}': gathered_rewards[i].data.item()
                        })

                # update psis
                if max_dataset_count==args.train_datasets_count:
                    update_rule = args.inf_dds_update_rule
                    if update_rule=='reinforce':
                        for i in range(args.train_datasets_count):
                            for j in range(args.train_datasets_count):
                                if i==j:
                                    new_psis[j] += args.dds_lr * (1.0 - old_probs[j]) * gathered_rewards[i].data.item()
                                else:
                                    new_psis[j] -= args.dds_lr * old_probs[j] * gathered_rewards[i].data.item()
                        # normalize and update
                        new_probs = np.array([np.exp(new_psis[i]) for i in range(args.train_datasets_count)])
                        new_probs = new_probs/np.sum(new_probs)
                    else:
                        gathered_rewards = torch.clamp(gathered_rewards, min=0.0, max=5.0) # clip the loss
                        log_new_train_domain_weights = torch.log(old_weights) + args.dds_lr * gathered_rewards.cpu()
                        log_new_train_domain_weights = log_new_train_domain_weights - torch.logsumexp(log_new_train_domain_weights, dim=0)
                        train_domain_weights = (1.0 - args.dds_reweight_eps) * torch.exp(log_new_train_domain_weights) + args.dds_reweight_eps / len(log_new_train_domain_weights)

                        new_train_datasets_weight = train_domain_weights[:].numpy()
                        new_train_datasets_probs = train_domain_weights[:].numpy()
                        new_psis = new_train_datasets_weight
                        new_probs = new_train_datasets_probs
                        # train_dataloader._avg_weights = np.array(train_dataloader._avg_weights) + new_train_datasets_weight
                        # train_dataloader._weight_updates += 1
                else:
                    old_psis = copy.deepcopy(train_dataloader._psis) 
                    old_probs = copy.deepcopy(train_dataloader._probs)
                    pmf_sum = sum([old_probs[i] for i in range(len(old_probs)) if i in sampled_datasets])
                    for idx, i in enumerate(sampled_datasets):
                        R_i = gathered_rewards[idx].data.item()
                        for j in sampled_datasets:
                            p_j_S = old_probs[j] / pmf_sum
                            advantage = (1 if j == i else 0) - p_j_S
                            new_psis[j] += args.dds_lr * advantage * R_i
                    exp_S = np.exp([new_psis[j] for j in sampled_datasets])
                    q_S = exp_S / exp_S.sum()
                    new_probs = copy.deepcopy(old_probs)
                    for idx, j in enumerate(sampled_datasets):
                        new_probs[j] = pmf_sum * q_S[idx]
                    if rank==0:
                        print('Old prob sum S', pmf_sum)
                        l1 = np.sum(np.abs(np.array(new_probs) - np.array(old_probs)))
                        tv  = 0.5 * l1
                        print(f"Total L1 change in mass: {l1:.6f}")
                        print(f"Total probability mass moved (TV distance): {tv:.6f}")

                model.train()
                # At this point, sum(p) may be 1.0000000000000002 etc.
                # Force exact sum by adjusting one element:
                print(f'before diff -> {dist.get_rank()} {new_probs}')
                diff = 1.0 - new_probs.sum()
                new_probs[0] += diff #last element probability in multinomial is assumed to be 1-new_probs[:-1]
                train_dataloader._probs = new_probs
                #train_dataloader._psis = np.log(new_probs) #new_psis
                train_dataloader._psis = new_psis
                train_dataloader._avg_weights = np.array(train_dataloader._avg_weights) + new_probs[:]
                train_dataloader._weight_updates += 1
                if rank==0:
                    print('\nnew psis', new_psis, '\n')
                    print('\nnew probs')
                if rank==0:
                    for i in range(args.train_datasets_count):
                        print(f'Sampling probability for {train_dataloader._names[i]}', new_probs[i]*100.0)
                        wandb.log({
                            f'Sampling probability for {train_dataloader._names[i]}': new_probs[i]
                        })
                    for i in range(args.train_datasets_count):
                        print( f'Sampling avg weights for {train_dataloader._names[i]}', train_dataloader._avg_weights[i]/train_dataloader._weight_updates)
                        wandb.log({
                            f'Sampling avg weights for {train_dataloader._names[i]}': train_dataloader._avg_weights[i]/train_dataloader._weight_updates
                        })
                # reset eval dataloader after dds usage
                eval_dataloader.reset_iter_after_dds()
                torch.cuda.empty_cache()





        #----------------------------------------------------------------------------------------------------------------#    
        # MultiDDS https://aclanthology.org/2020.acl-main.754.pdf
        if args.dds_warmup_steps>state.global_step:
            return
        if state.global_step%args.dds_update_step==1:
            #print('dds update')
            #if state.is_world_process_zero:
            #check is_world_process_zero
            #add args.dds_update_step to trainer arguments
            #set the accumulated gradients to zero
            #get the gradient on train and dev set
            #calculate reward
            #distribute the reward
            #update the dds trainer dataloader
            optim = kwargs.get('optimizer', None)
            model = kwargs.get('model', None)
            train_dataloader = kwargs.get('train_dataloader', None)
            eval_dataloader = kwargs.get('eval_dataloader', None)
            if optim == None:
                print('Optimizer not loaded from trainer callback')
                return
            if model == None:
                print('model not loaded from trainer callback')
                return
            if train_dataloader == None:
                print('train_dataloader not loaded from trainer callback')
                return
            if eval_dataloader == None:
                print('eval_dataloader not loaded from trainer callback')
                return
            train_dataloader = train_dataloader
            eval_dataloader = eval_dataloader

            rewards = [0]*args.train_datasets_count

            old_probs = train_dataloader._probs
            old_psis = train_dataloader._psis 
            #print('old probs', old_probs)
            #print('old psis', old_psis)

            # 1 update step 
            dds_batches = args.dds_repeat_batches
            for dds_step_idx in range(dds_batches):
                train_batches, val_batches = self.get_batch(
                    train_dataloader,
                    eval_dataloader, 
                    args.train_datasets_count, 
                    args.val_datasets_count,
                    args.dds_batch_size)

                # for each of the n train set and each of the m val set store the reward index of the train set
                train_idxs = [i for i in range(args.train_datasets_count) for _ in range(args.val_datasets_count)]
                reward_idx = 0
                model.train()
                train_data_idx = 0
                val_data_idx = 0
                val_batch_cache = []
                val_batch_norm_cache = []
                for train_batch, val_batch in tqdm(product(train_batches, val_batches)): #Do this for k batches for each train-dev combo
                    train_batch = batch_to_device(train_batch, rank=rank)
                    optim.zero_grad()
                    #print('Train batch',train_batch)
                    output = model(**train_batch)
                    loss = output.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # store train gradients
                    train_grads = []
                    for idx, param in enumerate(model.parameters()):
                        if param.grad!=None:
                            train_grads.append(param.grad.clone())

                    # norm of train grads
                    sq_train = torch.tensor(0.0,device=model.model.device)
                    for idx in range(len(train_grads)):
                        sq_train += torch.sum(train_grads[idx] * train_grads[idx])
                    
                    # reset grads to zero to not affect dev grads
                    optim.zero_grad()
                    #print('Val batch',val_batch)
                    if len(val_batch_cache)<args.val_datasets_count:
                        output = model(**val_batch)
                        loss = output.loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                        # store val grads
                        val_grads = []
                        for idx, param in enumerate(model.parameters()):
                            if param.grad!=None:
                                val_grads.append(param.grad.clone())
                        val_batch_cache.append(val_grads)
                        # norm of val grads
                        sq_val = torch.tensor(0.0,device=model.model.device)
                        for idx in range(len(val_grads)):
                            sq_val += torch.sum(val_grads[idx] * val_grads[idx])
                        val_batch_norm_cache.append(sq_val)
                    else:
                        #print(train_data_idx, val_data_idx, 'using cached grads')
                        val_grads = val_batch_cache[val_data_idx]
                        sq_val = val_batch_norm_cache[val_data_idx]
                        
                    # cosine similarity of train and val grad for a (train_data, dev_data) combo
                    dot_pdt = torch.tensor(0.0,device=model.model.device)
                    for i in range(len(train_grads)):
                        dot_pdt += torch.sum(train_grads[i]*val_grads[i])
                    reward = dot_pdt / torch.sqrt(sq_train*sq_val + 1e-7)

                    #print(f'Reward between train {train_idxs[reward_idx]} and dev {int(reward_idx/args.train_datasets_count)} = ', reward)
                    if rank==0:
                        print(f'{dds_step_idx} Reward between {train_dataloader._names[train_data_idx]} and {eval_dataloader._names[val_data_idx]} = ', reward.data.item())
                        wandb.log({
                            f'reward_trainset_{train_dataloader._names[train_data_idx]}_valset_{eval_dataloader._names[val_data_idx]}': reward.data.item()
                        })
                    # sum rewards across each dev data
                    rewards[train_idxs[reward_idx]] += reward
                    reward_idx+=1
                    optim.zero_grad()
                    
                    val_data_idx += 1
                    if val_data_idx%args.val_datasets_count==0:
                        val_data_idx = 0
                        train_data_idx += 1
                    

                # normalize rewards for number of dev data
                # for i in range(args.train_datasets_count):
                #     rewards[i] /= (args.val_datasets_count * dds_batches)
                #     rewards[i] = rewards[i].data.item()
                #     wandb.log({
                #         f'Total_reward_trainset_{train_dataloader._names[i]}': rewards[i]
                #     })
                

            new_psis = old_psis
            rewards = torch.tensor(rewards).to(model.model.device)
            gathered_rewards = [torch.zeros_like(rewards) for _ in range(dist.get_world_size())]
            
            # Gather rewards from all GPUs to all GPUs
            dist.all_gather(gathered_rewards, rewards)


            gathered_rewards = torch.cat([x.view(1,-1) for x in gathered_rewards], dim=0)
            #print(gathered_rewards)
            gathered_rewards = torch.mean(gathered_rewards, dim=0) # ideally this should be mean
            #print(gathered_rewards)

            if rank==0:
                for i in range(args.train_datasets_count):
                    wandb.log({
                        f'Total_reward_trainset_{train_dataloader._names[i]}': gathered_rewards[i].data.item()
                    })

            # update psis
            for i in range(args.train_datasets_count):
                for j in range(args.train_datasets_count):
                    if i==j:
                        new_psis[j] += args.dds_lr * (1.0 - old_probs[j]) * gathered_rewards[i].data.item()
                    else:
                        new_psis[j] -= args.dds_lr * old_probs[j] * gathered_rewards[i].data.item()

            # normalize and update
            new_probs = np.array([np.exp(new_psis[i]) for i in range(args.train_datasets_count)])
            new_probs = new_probs/np.sum(new_probs)
            train_dataloader._probs = new_probs
            #train_dataloader._psis = np.log(new_probs) #new_psis
            train_dataloader._psis = new_psis
            if rank==0:
                print('\nnew psis', new_psis, '\n')
                print('\nnew probs', new_probs, '\n')
            if rank==0:
                for i in range(args.train_datasets_count):
                    wandb.log({
                        f'Sampling probability for {train_dataloader._names[i]}': new_probs[i]
                    })
            #To do: save the optimizer states and restore them afterwards if dataloader is messed up
            eval_dataloader.reset_iter_after_dds()
            torch.cuda.empty_cache()
        else:
            return 


class BiencoderMLTrainer(Trainer):
    def __init__(self, dataset_handler, eval_dataset_handler, data_args=None, *pargs, **kwargs):
        super(BiencoderMLTrainer, self).__init__(*pargs, **kwargs)
        self.handler = dataset_handler
        self.eval_handler = eval_dataset_handler
        self.model: BiEncoderModel
        self.args.train_datasets_count = len(self.handler.train_datasets) 
        self.args.val_datasets_count = len(self.eval_handler.eval_datasets) 
        self.data_args=data_args

        
    def get_train_dataloader(self) -> DataLoader:
        ''' create a dataloader that merges underlying dataloaders batch-by-batch
            the intent is that each batch comes from a single dataset
            we create a single dataloader for each set, using super().get_train_dataloader()
            this has the side effect of also having a collator for each dataset
            we return a merging iterator that (multinomial)samples from its list of dataloader
            and returns the (already-collated) batches, without an additional collator
            We also depend upon the MultiLoaderIterator having the same sequence of random
            numbers in each process in order to make the multi-gpu batch homogenous
        '''
        datasets=self.handler.train_datasets
        names=self.handler.train_datasets_names
        probs=self.handler.train_datasets_probs 
        psis=self.handler.train_datasets_psis
        weights=self.handler.train_datasets_weight
        datasets_num_instances=self.handler.train_datasets_num_instances
        self._loaders = []
        num_workers_tmp = self.args.dataloader_num_workers 
        self.args.dataloader_num_workers=min(1,num_workers_tmp)
        for i, (dc, name) in enumerate(zip(datasets,names)):
            self.train_dataset=self.handler.train_datasets[i]
            # dl = super().get_train_dataloader()
            if self.data_args.is_train_data_streaming:
                dl = super().get_train_dataloader()
            else:
                base = super().get_train_dataloader()
                shard = base.dataset
                from torch.utils.data import RandomSampler
                sampler = RandomSampler(shard, replacement=True, num_samples=int(1e12))
                # from torch.utils.data import BatchSampler
                true_bs = base.batch_sampler.batch_size
                # drop = base.batch_sampler.drop_last
                # new_batch_sampler = BatchSampler(
                #      sampler=sampler,
                #      batch_size=true_bs,
                #      drop_last=drop,
                #  )
                # re-create DataLoader with infinite sampler
                # print('Batch size', base.batch_size, base.num_workers, base.batch_sampler)
                # print(base.batch_sampler.batch_size)
                dl = base.__class__(
                    dataset=shard,
                    batch_size=true_bs,
                    sampler=sampler,
                    collate_fn=base.collate_fn,
                    drop_last=base.drop_last,
                    num_workers=base.num_workers,
                    pin_memory=base.pin_memory,
                    worker_init_fn=base.worker_init_fn,
                    persistent_workers=False,      # keep default for max throughput
                )
                # print('Batch size', dl.batch_size, dl.num_workers, dl.batch_sampler)
                # print(dl.batch_sampler.batch_size)
            #     dl = super().get_train_dataloader().__class__(
            #     dataset=super().get_train_dataloader().dataset,
            #     batch_size=self.args.train_batch_size,
            #     sampler=super().get_train_dataloader().sampler,
            #     collate_fn=super().get_train_dataloader().collate_fn,
            #     drop_last=self.args.dataloader_drop_last,
            #     num_workers=self.args.dataloader_num_workers,
            #     pin_memory=self.args.dataloader_pin_memory,
            #     worker_init_fn=super().get_train_dataloader().worker_init_fn,
            #     persistent_workers=True,       
            # )
            self.train_dataset=None
            self._loaders.append(dl)
        self.args.dataloader_num_workers=num_workers_tmp
        mdl = MultiLoaderIterator(self.args.output_dir, self._loaders, names, probs, psis, weights, datasets_num_instances, self.args.train_batch_size, True)
        return mdl
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        ''' create a dataloader that merges underlying dataloaders batch-by-batch
            the intent is that each batch comes from a single dataset
            we create a single dataloader for each set, using super().get_train_dataloader()
            this has the side effect of also having a collator for each dataset
            we return a merging iterator that (multinomial)samples from its list of dataloader
            and returns the (already-collated) batches, without an additional collator
            We also depend upon the MultiLoaderIterator having the same sequence of random
            numbers in each process in order to make the multi-gpu batch homogenous
        '''
        #print('Eval dataset', eval_dataset)
        datasets=self.eval_handler.eval_datasets
        names=self.eval_handler.eval_datasets_names
        probs=self.eval_handler.eval_datasets_probs 
        psis=self.eval_handler.eval_datasets_psis
        weights=self.eval_handler.eval_datasets_weight
        datasets_num_instances=self.eval_handler.eval_datasets_num_instances
        self._loaders = []
        num_workers_tmp = self.args.dataloader_num_workers 
        self.args.dataloader_num_workers=min(1,num_workers_tmp)
        for i, (dc, name) in enumerate(zip(datasets,names)):
            self.train_dataset=self.eval_handler.eval_datasets[i]
            dl = super().get_eval_dataloader(self.train_dataset)
            self.train_dataset=None
            self._loaders.append(dl)
        self.args.dataloader_num_workers=num_workers_tmp
        mdl = MultiLoaderIterator(self.args.output_dir, self._loaders, names, probs, psis, weights, datasets_num_instances, self.args.eval_batch_size, False)
        return mdl

    def _save(self, output_dir: Optional[str] = None,state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir) #@meetdoshi ToDo: Save probs
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            save_ckpt_for_sentence_transformers(output_dir, pooling_mode=self.args.sentence_pooling_method)

    
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False): #@meetdoshi ToDo: Update the compute_loss function for bi-level optimisation
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
    
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        if inputs is None:
            return (None, None, None)
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        labels = None

        with torch.no_grad():
            loss = None
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index - 1]
            loss = outputs.loss

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    
    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint)
        model = self.model.load(best_model_path)
        return model




    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     ''' overloaded mainly for logging
    #     '''
    #     ts=super().training_step(model, inputs) 
    #     #print('pos embs:', model.lm_p.embeddings.position_embeddings.weight.sum(), 
    #     #            model.lm_q.embeddings.position_embeddings.weight.sum()
    #     #        )        
    #     if self.state.global_step<10: 
    #         with open(self.args.output_dir+'/trainer.'+str(os.getpid()), 'at') as pidOut:
    #             print(self.state.global_step, ' query_ids: ', str(inputs['q_query_ids']), ' q_input_ids.device: ', str(inputs['q_input_ids'].device), file=pidOut)
    #             print('training_step: ', ts, file=pidOut)
    #     return ts