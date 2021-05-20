#!/usr/bin/env bash

#SBATCH --output test.out
#SBATCH --error  test.err
#SBATCH -p       production
#SBATCH -t       02:00:00
#SBATCH --mem    30
#SBATCH --gres   gpu:1,gpu_mem:50

# --mail-type ALL
# --mail-user kjfraga@ucdavis.edu

hostname
nvidia-smi
nvcc --version
module load anaconda3 gcc
gcc --version

__conda_setup="$('/software/anaconda3/4.8.3/lssc0-linux/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
	eval "$__conda_setup" 
else
	if [ -f "/software/anaconda3/4.8.3/lssc0-linux/etc/profile.d/conda.sh" ]; then
		. "/software/anaconda3/4.8.3/lssc0-linux/etc/profile.d/conda.sh"
	else
		export PATH="/software/anaconda3/4.8.3/lssc0-linux/bin:$PATH"
	fi
fi
unset __conda_setup

conda activate myenv

echo " "
python graphdot_slurm.py data/data.json.xz

echo "END"