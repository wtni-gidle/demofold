#!/bin/bash
#SBATCH --job-name="train_e2e_1"
#SBATCH --account=bbgs-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --output="train_e2e_1.%j.%N.out"
#SBATCH --error="train_e2e_1.%j.%N.out"
#SBATCH --export=ALL
#SBATCH -t 48:00:00
#SBATCH --exclusive  # dedicated node for this job

# make the script stop when error (non-true exit code) occurs
set -e

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

unset LD_LIBRARY_PATH

conda activate openfold_env
cd /projects/bbgs/nwentao/projects/str_pred/demofold

mmcif_dir="/projects/bbgs/nwentao/data/demofold/mmcif_files/demofold"
ss_dir="/projects/bbgs/nwentao/data/demofold/ss_dir"
output_dir="/projects/bbgs/nwentao/data/demofold/train_e2e_1"
train_filter_path="/projects/bbgs/nwentao/data/demofold/train_filter.txt"
val_filter_path="/projects/bbgs/nwentao/data/demofold/val_filter.txt"
deepspeed_config_path="/projects/bbgs/nwentao/projects/str_pred/demofold/deepspeed_config.json"


mkdir -p "$output_dir"

srun python3 train_demofold.py "$mmcif_dir" "$ss_dir" "$output_dir" \
    --val_data_dir "$mmcif_dir" \
    --val_ss_dir "$ss_dir" \
    --train_filter_path "$train_filter_path" \
    --val_filter_path "$val_filter_path" \
    --gpus 8 --replace_sampler_ddp=True \
    --seed 42 \
    --checkpoint_every_epoch \
    --log_performance \
    --log_lr \
    --config_preset e2e \
    --batch_size 1 \
    --train_epoch_len 5000
    # --precision bf16 \ # V100不能用bf16
    # --deepspeed_config_path deepspeed_config.json # deepspped和fp16不兼容，会启动bf16（不确定）
    # --resume_from_ckpt ckpt_dir/ \