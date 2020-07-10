#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=32g
#SBATCH --cpus-per-task=14
#SBATCH --output=/data/sarroutim2/baseline/jobs/job_%j.out
#SBATCH --error=/data/sarroutim2/baseline/jobs/job_%j.error
#SBATCH --gres=gpu:p100:1
#SBATCH --time=9-16:30:00


export CUDA_HOME=/usr/local/CUDA/10.1.105/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}




batch_size=128




python ./train_iq.py --batch-size ${batch_size}
