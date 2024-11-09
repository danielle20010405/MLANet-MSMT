#!/bin/bash --login 
#$ -cwd
#$ -l v100=1
#$ -pe smp.pe 8 

export OMP_NUM_THREADS=$NSLOTS

echo "Job is using $NGPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $NSLOTS CPU core(s)"

conda activate deep3dpytorch

module load compilers/gcc/9.3.0

python test.py --name=bottle_co-downaff --epoch=34 --gpu_ids=$CUDA_VISIBLE_DEVICES 
