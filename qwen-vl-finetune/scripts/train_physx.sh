
#!/bin/bash
#SBATCH -J HELLO
#SBATCH -p your partition
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=finetune_7b_8.out


# Distributed training configuration
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export NUM_GPUS=8
export NNODES=1
MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))

conda activate physxomni

deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-7B-Instruct

# Training hyperparameters
lr=2e-5
batch_size=2
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=physxverse64,physxnet64,physxmobility64

# Output configuration
run_name="qwen2vl-baseline_7b_64_scaleup"
output_dir=./output_7b_64_scaleup

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 262144 \
    --min_pixels 65536 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"


         

torchrun --nproc_per_node=$SENSECORE_ACCELERATE_DEVICE_COUNT \
        --nnodes=$SENSECORE_PYTORCH_NNODES \
        --node_rank $SENSECORE_PYTORCH_NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        ${entry_file} ${args}

  
