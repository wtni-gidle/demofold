#!/bin/bash
#SBATCH --job-name="fasta"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --output="fasta.%j.%N.out"
#SBATCH --error="fasta.%j.%N.out"
#SBATCH --account=mia174
#SBATCH --export=ALL
#SBATCH -t 00:10:00

# < 10 min

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
log_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/mmcif_cache_log.json"

python scripts/generate_pdb_fasta.py "$fasta_path" "$cache_path" "$log_path"