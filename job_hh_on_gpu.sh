#!/bin/bash

#SBATCH --job-name=train_pinn_on_hh_data.py
#SBATCH --partition=ampere-q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu
#SBATCH --time=40:00:0

module load python-uoneasy/3.11.5-GCCcore-13.2.0
source ../../venvs/env_pinns/bin/activate
python train_pinn_on_hh_data.py > log_training_on_hh_data_gpu.txt