#!/bin/bash
#SBATCH --job-name="train4e2e"
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4G
#SBATCH --output="train4e2e.%j.%N.out"
#SBATCH --error="train4e2e.%j.%N.out"
#SBATCH --account=mia174
#SBATCH --export=ALL
#SBATCH -t 00:30:00

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

conda activate openfold
cd /expanse/projects/itasser/jlspzw/nwentao/projects/demofold

mmcif_dir="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/mmcif_files/"
ss_dir="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/ss_dir/"
output_dir="/expanse/projects/itasser/jlspzw/nwentao/train4e2e"
train_filter_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/train_filter.txt"
val_filter_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/val_filter.txt"
deepspeed_config_path="/expanse/projects/itasser/jlspzw/nwentao/projects/demofold/deepspeed_config.json"


mkdir -p "$output_dir"

python3 train_demofold.py "$mmcif_dir" "$ss_dir" "$output_dir" \
    --val_data_dir "$mmcif_dir" \
    --val_ss_dir "$ss_dir" \
    --train_filter_path "$train_filter_path" \
    --val_filter_path "$val_filter_path" \
    --gpus 4 --replace_sampler_ddp=True \
    --seed 4242022 \
    --checkpoint_every_epoch \
    --log_performance \
    --log_lr \
    --config_preset e2e \
    --batch_size=1
    # --precision bf16 \ # V100不能用bf16
    # --deepspeed_config_path deepspeed_config.json # deepspped和fp16不兼容，会启动bf16（不确定）
    # --resume_from_ckpt ckpt_dir/ \