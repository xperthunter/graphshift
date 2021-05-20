#!/usr/bin/env bash

#SBATCH --output pred3600.out
#SBATCH --error  pred3600.err
#SBATCH -p       production
#SBATCH -t       1000:00:00
#SBATCH -x       rafter-33,rafter-34
#SBATCH --mem    9000
#SBATCH --gres   gpu:1,gpu_mem:3500

#SBATCH --mail-type ALL
#SBATCH --mail-user kjfraga@ucdavis.edu

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
python mae.py -p saved_models/ -n 3600 -m model_3600_.pickle -d data/nmrshift_training.pickle.xz

echo "END"
