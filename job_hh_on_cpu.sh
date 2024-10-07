#!/bin/bash

#SBATCH --job-name=train_pinn_on_hh_data.py
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCh --mem=64g
#SBATCH --time=60:00:0

module load python-uoneasy/3.11.5-GCCcore-13.2.0
source ../../venvs/env_pinns/bin/activate
python train_pinn_on_hh_data.py > log_training_on_hh_data_cpu.txt
