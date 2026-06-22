
# Dynamic Data Sampling Codebase (Inf-DDS and Other DDS Approaches)

This repository provides implementations of multiple **dynamic data sampling (DDS)** approaches for multi-dataset training and domain adaptation, including **Inf-DDS**.

The common goal across all methods is to learn a **sampling distribution** over training datasets that improves downstream performance after fine-tuning.

---

## Overview

The core implementation for dynamic data sampling is located in  
`FlagEmbedding/baai_general_embedding/finetune/multiloader_trainer.py`.

The logic is implemented using the Hugging Face Trainer’s `on_step_end` callback. This callback starts receiving the evaluation datasets after the first evaluation run is finished according to value set by `eval_steps`. So warmup for dds needs to be set atleast higher than `eval_steps`.
Sampling distribution updates begin once `state.global_step > dds_warmup_steps` and are performed at the following interval: `state.global_step % dds_update_step == 1`. Sampling weights can be found in the job and wandb logs.

You can set env variables `WANDB_API_KEY` and `WANDB_DIR` for wandb.


---

## Notation and Setup

- **θ (theta)**: model parameters  
- **ψ (psi)**: sampling distribution over training datasets  

We assume:
- **M training datasets**
- Optionally, **N development (dev) datasets**, depending on the method

In rough terms, the objective is to learn a sampling distribution **ψ** that maximizes the performance of **θ** after fine-tuning on the training data.

---

## Sampling Initialization

The following options control how datasets are initially sampled:

- `train_datasets_sampling_init: "lines"`  
  Samples datasets proportionally to the number of lines specified in the config file.

- `train_datasets_sampling_init: "weight"`  
  Samples datasets according to explicit weights specified in the config file.

**Recommended usage:**
- For **proportional sampling**, set the correct line counts and use uniform weights.
- For **uniform sampling**, set all weights to `1`.

---

## Supported Methods

The codebase supports both **non-parametric** and **parametric** sampling approaches. 

### Method Categories

| Category | Methods |
|--------|--------|
| Non-parametric | Static Sampling, Cooldown Sampling |
| Optimize on dev set | Inf-DDS, Multi-DDS, DoGE, CRISP |
| Optimize on train set | DoReMi |

---

## Non-Parametric Approaches

### 1. Static Sampling

Uses a fixed sampling distribution throughout training.

---

### 2. Cooldown Sampling

A two-phase sampling schedule:

- **First half of training**: near-uniform sampling (`t = 5`)
- **Second half of training**: proportional sampling (`t = 1`)

To enable:
set `cooldown_sampling=True`. It will ignore existing sampling distribution after the first update step and use the above logic.


---

## Approaches That Optimize on the Dev Set

### 1. Inf-DDS

**Paper:** [https://arxiv.org/pdf/2601.21759](https://arxiv.org/pdf/2601.21759)

**Inf-DDS (Influence-guided Dynamic Data Sampling)** is an influence-based approach that iteratively updates dataset sampling weights based on their impact on dev set performance.

Training can be performed in:

* **Single-step mode**: ψ weights are learned online during training along with model parameters θ. In single step mode, try out different temperature initializations.
* **Two-step mode**: final weights from step 1 are used to retrain a new model with static sampling.

#### Training Parameters

* `inf_dds_reweighing`: must be set to `True`
* `dds_update_step`: update interval for Inf-DDS
* `dds_batch_size`: batch size for online proxy training on train datasets
* `inf_dds_train_steps_l`: number of update steps before influence computation
* `inf_dds_eval_batch_size`: batch size for influence computation on dev data
* `infdds_rewards_scaler`: reward scaling factor (recommended `1.0/5.0/10.0`)
* `dds_warmup_steps`: warmup steps for stability
* `dds_lr`: learning rate (typically ~5.0 for faster learning; lower if rewards are too aggressive)
* `dds_repeat_batches`: number of repeated batches for reward estimation
* `dds_reweight_eps`: epsilon for weighted update rule

---

### 2. Multi-DDS

**Paper:** [https://aclanthology.org/2020.acl-main.754.pdf](https://aclanthology.org/2020.acl-main.754.pdf)

Multi-DDS reweights datasets based on **gradient similarity** between training and dev gradients. Training is single step where both the model parameters and scorer distribution are optimized simultaneously.

#### Training Parameters

* `dds_update_step`: update interval
* `dds_batch_size`: batch size for gradient computation
* `dds_warmup_steps`: warmup steps for stability
* `dds_lr`: learning rate (recommended: `0.9`)
* `dds_repeat_batches`: number of repeated batches for reward estimation

---

### 3. DoGE

**Paper:** [https://arxiv.org/abs/2310.15393](https://arxiv.org/abs/2310.15393)

DoGE uses **generalization gain** computed from training and dev gradients to update dataset sampling weights.

#### Training Procedure

1. Learn adaptive sampling weights (step 1)
2. Retrain a new model using the learned weights (static sampling)

#### Training Parameters

* `doge_reweighing`: must be set to `True` for step 1
* `doge_update_step`: update interval
* `doge_batch_size`: batch size for DoGE updates
* `doge_warmup_steps`: warmup steps
* `doge_lr`: learning rate for sampling updates (recommended `1e-4`)
* `doge_bregman_coeff`: Bregman divergence coefficient
* `doge_dw_min`: lower bound for score clipping (default: `0.0`)
* `doge_dw_max`: upper bound for score clipping (default: `5.0`)

---

### 4. CRISP

**Paper:** [https://arxiv.org/abs/2410.03735](https://arxiv.org/abs/2410.03735)

CRISP differs from other approaches by computing sampling weights at the **cluster level** rather than the dataset level.

#### High-Level Workflow

1. Generate embeddings (queries or documents) using a pretrained embedding model for all train and dev samples
2. Perform hierarchical clustering on training samples and assign each train sample to a cluster and keep track of how many samples fall into each cluster.
3. Assign dev samples to the nearest clusters and keep track of how many samples fall in each cluster.
4. Compute cluster-level importance weights (`crisp_clustering.py`) -> w(c) = P(c | Ds) / P(c | Dg)
5. Apply importance sampling over clusters during training -> Expectation over P(c | Dg) distribution for w(c) with the loss.

During training:

* Clusters are sampled according to importance weights or simply `weights` in your data_config.json (specialist histogram).
* Samples within each cluster are drawn uniformly and `weight` values that were stored for each training sample are used to rescale the instance level loss values.

---

#### CRISP Preprocessing Scripts (`run_config/`)

1. `crisp_clustering.py`
   Generates hierarchical clusters and assigns samples

2. `crisp_merge.py`
   Merges cluster files to reduce the number of datasets loaded during training
   (e.g., depth 2 with 32 clusters produces 1024 files, which is not recommended)

3. `crisp_aggregate.py`
   Aggregates sampling probabilities so parent cluster probabilities equal the sum of child probabilities

After preprocessing, train using:

```yaml
train_datasets_sampling_init: "weight"
```

If each sample includes a weight, loss scaling is handled automatically.

---

#### CRISP Configuration

```python
ROOTS = {
    "32-clusters": {"depth": 1},
    "1024-clusters": {"depth": 2}
}
```

#### Clustering Hyperparameters

* `K = 32`: branching factor per parent
* `SAMPLE_SIZE = 10000`: samples used for k-means
* `STEPS = 10`: k-means iterations
* `BALANCING_RATIO = 1.5 / K`: hierarchy balancing ratio
* `SEED = 42`
* `DEPTH = 2`: hierarchy depth
* `BATCH_SIZE = 1024`: k-means batch size

---

## Approaches That Optimize on the Train Set

### 1. DoReMi

**Paper:** [https://arxiv.org/abs/2305.10429](https://arxiv.org/abs/2305.10429)

DoReMi uses **GroupDRO** to optimize worst-case loss across datasets, guided by a reference model.

#### Training Procedure

1. Train a baseline model using proportional sampling (step 1)
2. Use this model (step 1 model) as a reference for DRO-LM optimization (step 2 will give you new sampling weights)
3. Average weights learned in step 2 over the whole step 2 train duration and retrain a final model using static sampling (step 3).

**Note:** The reference and proxy models are assumed to use the same tokenizer.

#### Training Parameters

* `doremi_reweighing`: must be set to `True` for step 2
* `doremi_reference_model_path`: path to the reference model
* `doremi_update_step`: update interval
* `doremi_batch_size`: batch size
* `doremi_warmup_steps`: warmup before DRO-LM starts
* `doremi_reweight_eta`: scaling factor (default: `1.0`)
* `doremi_reweight_eps`: smoothing term (default: `1e-3`)

