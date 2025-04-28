#!/bin/bash
#SBATCH   --partition=normal            # submit   to the normal(default) partition
#SBATCH   --job-name=pubmed-bo        # name the job
#SBATCH   --output=pubmed-bo-%j.out   # write stdout/stderr   to named file
#SBATCH   --error=pubmed-bo-%j.err      
#SBATCH   --time=2-0:00:00              # Run for max of 02 hrs, 00 mins, 00 secs
#SBATCH   --nodes=1                     # Request N nodes
#SBATCH   --cpus-per-task=18            # Request n   cores per node
#SBATCH   --mem-per-cpu=4GB             # Request nGB RAM per core

# User specific aliases and functions
export CONDA_PKGS_DIRS=/projects/mparsa/ssnyde9/.conda/cache/
export CONDA_ENVS_DIRS=/projects/mparsa/ssnyde9/.conda/envs/

# SLURM Aliases
alias salloc4cpu='salloc -p normal --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=5GB --time=0-04:00:00'
alias salloc4gpu='salloc -p gpuq -q gpu -n 1 --cpus-per-task=4 --mem=32GB --gres=gpu:A100.40gb:1 -t 0-04:00:00'

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/etc/profile.d/conda.sh" ]; then
        . "/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/etc/profile.d/conda.sh"
    else
        export PATH="/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/miniconda3-22.11.1-gy/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate sgnn-sparse

python spawn_bo_pubmed.py --config ./configs/pubmed/default_pubmed_config.yaml
