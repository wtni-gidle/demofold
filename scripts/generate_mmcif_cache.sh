#!/bin/bash
#SBATCH --job-name="parse_data"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem-per-cpu=1G
#SBATCH --output="parse_data.%j.%N.out"
#SBATCH --error="parse_data.%j.%N.out"
#SBATCH --account=mia174
#SBATCH --export=ALL
#SBATCH -t 00:30:00

# about 20 min

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
output_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/mmcif_cache.json"
log_path="/expanse/ceph/projects/itasser/jlspzw/nwentao/pdb_mmcif/mmcif_cache_log.json"

python scripts/generate_mmcif_cache.py "$mmcif_dir" "$output_path" "$log_path" --no_workers 128