# cd ../
# pip3 install .
# cd meet/
export HF_HOME="/proj/irl-agentic/meet/cache/"
# export HF_HOME="/proj/irl-agentic/meet/cache/"
# export WANDB_DIR="/dccstor/sdg/meet/wandb_logs/"
if [[ -n "$WANDB_API_KEY" ]]; then
    wandb login --relogin "$WANDB_API_KEY"
fi
export WANDB_MODE="offline"
export NCCL_SOCKET_IFNAME="ib,bond"
export NCCL_IB_CUDA_SUPPORT=1
export CUBLAS_WORKSPACE_CONFIG=:16:8
#export CUBLAS_WORKSPACE_CONFIG=""
export CUDA_LAUNCH_BLOCKING=1 

NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
MASTER_ADDRESS=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
HOST=$(echo $HOSTNAME | cut -d '.' -f1)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOST | cut -d':' -f1)-1))
MASTER_PORT=5${LSB_JOBID: -4}

echo $NNODES $GPUS_PER_NODE

TOKENIZERS_PARALLELISM=false \
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=101 \
    --rdzv_endpoint=$MASTER_ADDRESS:$MASTER_PORT \
    -m FlagEmbedding.baai_general_embedding.finetune.run_20gpus_mdl \
    --output_dir /proj/irl-agentic/meet/models/dds/train_bge-ds-langwise-custom-weight-8k-full-distil_mldr13dev_doge_step2/ \
    --model_name_or_path BAAI/bge-m3-unsupervised \
    --train_data /dccstor/sdg/meet/data/DDS/bge-m3/bge-m3-langwise/ \
    --train_data_config /dccstor/sdg/meet/data/DDS/bge-m3/bge-m3-langwise/config_doge_step2.json \
    --eval_data /dccstor/sdg/meet/data/DDS/bge-m3/MLDR/ \
    --eval_data_config /dccstor/sdg/meet/data/DDS/bge-m3/MLDR/config.json \
    --learning_rate 2e-5 \
    --bf16 \
    --deepspeed config/ds_config.json \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --warmup_steps 1000 \
    --query_max_len 512 \
    --passage_max_len 8192 \
    --train_group_size 8 \
    --negatives_cross_device \
    --logging_steps 10 \
    --sentence_pooling_method 'cls' \
    --save_steps 500 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --dataloader_num_workers 8 \
    --max_steps 10000 \
    --remove_unused_columns False \
    --fix_position_embedding False \
    --dispatch_batches False \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs='{"use_reentrant":"False"}' \
    --report_to wandb \
    --train_datasets_sampling_init "weight" \
    --is_train_data_streaming True \
    --is_neg_distil_collator True \
    --load_best_model_at_end False 
    # --train_datasets_count 2 \
    # --val_datasets_count 1