#!/usr/bin/env bash

#SBATCH --output slurm_outs/tang19_1000.out
#SBATCH --error  slurm_outs/tang19_1000.err
#SBATCH -p       production
#SBATCH -t       1000:00:00
#SBATCH -x       rafter-34,rafter-33
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
python tang19_training -d nmrshift_Kang20filter_training.pickle.xz -n 1000 -b 1000 -o saved_models -p 1e-2 -v -1 -f model_tang19_1000.pickle

echo "END"