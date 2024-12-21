#!/bin/bash

#SBATCH --job-name=train_pinn_on_hh_data.py
#SBATCH --partition=ampereq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gres=gpu
#SBATCH --time=120:00:0

module load python-uoneasy/3.11.5-GCCcore-13.2.0
module load cuda-uoneasy/12.1.1 
source ../../venvs/env_pinns/bin/activate
python train_pinn_on_wang_data.py
