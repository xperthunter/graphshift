#!/usr/bin/env bash

while getopts e:o:m:g:l:j: flag
do
	case "${flag}" in
		e) errfile=${OPTARG};;
		o) outfile=${OPTARG};;
		m) mem=${OPTARG};;
		g) gpumem=${OPTARG};;
		l) slurmdir=${OPTARG};;
		j) config=${OPTARG};; 
	esac
done

if [ -z "$errfile" ] ; then
	echo "ERROR: Must provide an out file name"
	exit 1
fi

if [ -z "$outfile" ] ; then
	echo "ERROR: Must provide an out file name"
	exit 1
fi

if [ -z "$mem" ] ; then
	echo "ERROR: Must provide node memory"
	exit 1
fi

if [ -z "$gpumem" ] ; then
	echo "ERROR: Must provide gpu memory"
	exit 1
fi

if [ -z "$config" ] ; then
	echo "ERROR: Must provide a configuration set"
	exit 1
fi

if [ -z "$slurmdir" ] ; then
	echo "ERROR: Must provide directory for slurm error/out files"
	exit 1
fi

errpath="$slurmdir/$errfile"
outpath="$slurmdir/$errfile"

sbatch <<EOT
#!/bin/bash

#SBATCH --output $outpath
#SBATCH --error  $errpath
#SBATCH -p       production
#SBATCH -t       1000:00:00
#SBATCH -x       rafter-34,rafter-33
#SBATCH --mem    $mem
#SBATCH --gres   gpu:1,gpu_mem:$gpumem

#SBATCH --mail-type ALL
#SBATCH --mail-user kjfraga@ucdavis.edu
 
hostname
nvidia-smi
nvcc --version
module load anaconda3 gcc
gcc --version
EOT

# __conda_setup="$('/software/anaconda3/4.8.3/lssc0-linux/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
# 	eval "$__conda_setup" 
# else
# 	if [ -f "/software/anaconda3/4.8.3/lssc0-linux/etc/profile.d/conda.sh" ]; then
# 		. "/software/anaconda3/4.8.3/lssc0-linux/etc/profile.d/conda.sh"
# 	else
# 		export PATH="/software/anaconda3/4.8.3/lssc0-linux/bin:$PATH"
# 	fi
# fi
# unset __conda_setup
# 
# conda activate myenv
# 
# echo " "
# python train_batchedgpr -d data/nmrshift_training.pickle.xz -n $trainum -b $batchsize -o saved_models -p $alpha -v -1 -f model_1800_alpha2_.pickle
# 
# echo "END"