#!/bin/bash

#SBATCH --job-name=fr_t
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# To clean and load modules defined at the compile and link phases
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment
source activate /gpfs/users/piatc/.conda/envs/font

# Go to the directory where the job has been submitted 
cd ${SLURM_SUBMIT_DIR}

# Execution
python -u training.py -d dafont_10_100 -pkb 4 -e 5 -lr 1e-3