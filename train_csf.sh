#!/bin/bash --login 
#$ -cwd
#$ -l a100=1
#$ -pe smp.pe 8 

export OMP_NUM_THREADS=$NSLOTS

echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"

conda activate deep3dPytorch

module load compilers/gcc/9.3.0

python train.py --name=total-multifusion-multitask-MLKA --n_epochs=40 --gpu_ids=$CUDA_VISIBLE_DEVICES
