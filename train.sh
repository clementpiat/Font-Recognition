#!/bin/bash

#PBS -S /bin/bash
#PBS -N train
#PBS -j oe
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8:mem=8gb
#PBS -P gr
#PBS -M clement.piat@student.ecp.fr

# Load necessary modules
module purge
module load anaconda3/5.3

# Activate anaconda environment
source activate /gpfs/users/piatc/.conda/envs/font

# Go to the directory where the job has been submitted 
cd $PBS_O_WORKDIR

# Execution
python training.py