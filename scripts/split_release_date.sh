#!/bin/bash
#SBATCH --job-name="precompute_ss"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=512M
#SBATCH --output="precompute_ss.%j.%N.out"
#SBATCH --error="precompute_ss.%j.%N.out"
#SBATCH --account=mia174
#SBATCH --export=ALL
#SBATCH -t 00:10:00

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
cache_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/mmcif_cache.json"
train_filter_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/train_filter.txt"
val_filter_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/val_filter.txt"

python scripts/split_release_date.py "$fasta_path" "$cache_path" "$train_filter_path" "$val_filter_path"