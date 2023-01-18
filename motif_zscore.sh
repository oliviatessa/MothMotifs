#!/bin/bash

#SBATCH --job-name=motif_zscore
#SBATCH --mail-type=ALL
#SBATCH --mail-user=otthomas@uw.edu

#SBATCH --account=dynamicsai
#SBATCH --partition=gpu-a40 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 
#SBATCH --mem=128G 
#SBATCH --gpus=0 
#SBATCH --time=00-24:00:00 

#SBATCH --chdir=/mmfs1/gscratch/dynamicsai/otthomas/MothMotifs/MothMotifs
#SBATCH --export=all
#SBATCH --output=/mmfs1/gscratch/dynamicsai/otthomas/MothMotifs/MothMotifs/DataOutput/slurmOutputs/test.out
#SBATCH --error=/mmfs1/gscratch/dynamicsai/otthomas/MothMotifs/MothMotifs/DataOutput/slurmErrors/test.err

python3 /mmfs1/gscratch/dynamicsai/otthomas/MothMotifs/MothMotifs/motif_zscore.py