#!/bin/bash
#SBATCH --partition=a100q                 # Specify the partition a100q for a100
#SBATCH --nodelist=sabercore-a100-001      # Request specific node
#SBATCH --cpus-per-task=4                # Allocate 4 CPUs (adjust as needed)
#SBATCH --time=72:30:00                   # Set a time limit of 1 hour
#SBATCH --job-name=ScRDiT_Exp_Train              # Name the job as "clinpreai"
#SBATCH --output=job_output.txt    # Store logs in 'logs/' directory with Job ID
#SBATCH --reservation=jeffz_1178

source /cm/shared/apps/anaconda3/3.11/etc/profile.d/conda.sh
conda activate dl1
echo 'conda'

#python hpo.py --config scrdit_match
python train.py