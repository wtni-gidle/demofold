#!/bin/bash
#SBATCH --job-name="precompute_ss"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=512M
#SBATCH --output="precompute_ss.%j.%N.out"
#SBATCH --error="precompute_ss.%j.%N.out"
#SBATCH --account=mia174
#SBATCH --export=ALL
#SBATCH -t 02:00:00

# 由于有一些序列太长，极少序列没跑完

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

fasta_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/pdb_RNA.fasta"
output_dir="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/ss_dir"

mkdir -p "$output_dir"

python scripts/precompute_ss.py "$fasta_path" "$output_dir" --no_workers 64