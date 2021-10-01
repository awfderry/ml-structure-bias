#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=64G
# #SBATCH --begin=now+4hours
#SBATCH -C GPU_MEM:24GB
#SBATCH --output=./runs/mqa-%j.out
#SBATCH --error=./runs/mqa-%j.err
#SBATCH --partition=rbaltman,gpu,owners
#SBATCH --gres gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aderry@stanford.edu
#SBATCH --job-name=mqa

source /oak/stanford/groups/rbaltman/aderry/miniconda3/etc/profile.d/conda.sh
conda activate /oak/stanford/groups/rbaltman/aderry/miniconda3/envs/gvp

python run_mqa.py --data_dir /oak/stanford/groups/rbaltman/aderry/protein-data-bias/Task2/casp_mixed_lmdb/data --batch_size 12 --mode test --name casp_mixed --num_epochs 25
