import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    infonce: bool = field(default=False,metadata={"help":"Use infonce loss"}
                          )
    normlized: bool = field(default=True)
    
    lora: bool = field(default=True,metadata={"help":"use LORA FT or not"})
    


@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_data_config: str = field(
        default=None, metadata={"help": "Path to train data configuration file"}
    )
    train_datasets_sampling_init: str = field(default='weight', metadata={"help": "Init weights of training datasets to take weight or lines"})
    train_datasets_sampling_temp: float = field(default=1.0)
    eval_data: str = field(
        default=None, metadata={"help": "Path to eval data"}
    )
    eval_data_config: str = field(
        default=None, metadata={"help": "Path to eval data configuration file"}
    )
    train_group_size: int = field(default=8)
    is_train_data_streaming: bool = field(default=True)
    is_neg_distil_collator: bool = field(default=False)

    query_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage"}
    )
    in_batch_neg: bool = field(
        default=False, metadata={"help": "use only in batch negatives"}
    )
    interleave: bool = field(
        default=False, metadata={"help": "use interleave datasets"}
    )
    
    # def __post_init__(self):
    #     if not os.path.exists(self.train_data) or os.path.exists(self.eval_data) :
    #         raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    full_contrastive: bool = field(default=False, metadata={"help": "Use Full contrastive loss"})
    #MultiDDS and InfDDS sampling hps
    dds_update_step: int = field(default=100000000, metadata={"help": "Number of update steps before updating psi values of the actor model"})
    dds_batch_size: int = field(default=32, metadata={"help": "Batch size while computing gradient similarity for dds"})
    dds_warmup_steps: int = field(default=100000000, metadata={"help": "Number of warmup steps before updating starting dds updates"})
    dds_lr: float = field(default=1.0, metadata={"help": "Learning rate for the actor psi probs"})
    dds_reweight_eps: float = field(default=0.001, metadata={"help": "reweigh eps"})
    dds_repeat_batches: int = field(default=1, metadata={"help": "Repeat of reward calculation for dds"})
    inf_dds_eval_batch_size: int = field(default=32, metadata={"help": "Eval Batch size while computing scores for infdds"})
    inf_dds_train_steps_l: int = field(default=5, metadata={"help": "Train steps of each model for infdds"})
    inf_dds_subsample_size: int = field(default=-1, metadata={"help": "Number of datasets to subsample from for inf-dds"})
    infdds_rewards_scaler: float = field(default=10.0, metadata={"help": "Reward scaler"})
    inf_dds_reweighing: bool = field(default=False, metadata={"help": "Use InfDDS"})
    inf_dds_update_rule: str = field(default='weighted', metadata={"help": "Update rule for reweighing. Choose between reinforce or weighted"})
    train_datasets_count: int = field(default=1, metadata={"help": "Number of training datasets"})
    val_datasets_count: int = field(default=1, metadata={"help": "Number of dev datasets"})
    #Cooldown sampling hps
    cooldown_sampling: bool = field(default=False, metadata={"help": "Cooldown sampling with t=5 for 1st half of training and t=1 for second half"})
    #DoReMi sampling hps
    doremi_reweighing: bool = field(default=False, metadata={"help": "Doremi step 2 training"})
    doremi_reference_model_path: str = field(default=None, metadata={"help": "Path to reference model for optimization with group DRO loss"})
    doremi_update_step: int = field(default=100000000, metadata={"help": "Number of update steps before updating alpha parameters for doremi"})
    doremi_batch_size: int = field(default=32, metadata={"help": "Batch size while computing DRO loss"})
    doremi_warmup_steps: int = field(default=100000000, metadata={"help": "Number of warmup steps before updating starting doremi updates"})
    # doremi_lr: float = field(default=0.001, metadata={"help": "Learning rate for updating domain weights"})
    doremi_reweight_eta: float = field(default=1.0, metadata={"help": "reweigh eta"})
    doremi_reweight_eps: float = field(default=0.001, metadata={"help": "reweigh eps"})
    #DoGE sampling hps
    doge_reweighing: bool = field(default=False, metadata={"help": "DoGE step 1 training"})
    doge_update_step: int = field(default=100000000, metadata={"help": "Number of update steps before updating alpha parameters for doge"})
    doge_batch_size: int = field(default=32, metadata={"help": "Batch size while computing DoGE updates"})
    doge_warmup_steps: int = field(default=100000000, metadata={"help": "Number of warmup steps before updating starting doge updates"})
    doge_lr: float = field(default=0.001, metadata={"help": "Learning rate for updating domain weights"})
    doge_bregman_coeff: float = field(default=0.01, metadata={"help": "Hyperparam for Bregman Divergence."})
    doge_dw_min: float = field(default=0.0, metadata={"help": "Score clip upper bound (*lr_t)."})
    doge_dw_max: float = field(default=5.0, metadata={"help": "Score clip lower bound (*lr_t)."})

