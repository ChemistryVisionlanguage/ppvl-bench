#!/bin/bash
#SBATCH -J Chemistry_Blip
#SBATCH --mem=36GB  
#SBATCH --gres=gpu:1 -C gmem24
#SBATCH --output=finetune20_job.out

module load anaconda3
module load cuda/11.0
source activate blip
python3 finetune.py
