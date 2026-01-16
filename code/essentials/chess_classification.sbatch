#!/bin/bash
#SBATCH --partition=gtx1080
#SBATCH --qos=course
#SBATCH --time=0-04:00:00
#SBATCH --job-name=chess_final_project
#SBATCH --output=chess_job-%J.out
#SBATCH --mail-user=noayaak@post.bgu.ac.il
#SBATCH --mail-type=END,FAIL
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
nvidia-smi

module load anaconda
source activate my_env

ARGS=(
    --data_root /home/noayaak/final_project/Introduction_to_deep_learning_project/data
    --games game2_per_frame game4_per_frame game5_per_frame
    --out experiments
    --epochs 50
    --batch 32
    --lr 0.0001
    #--mode zero_shot
    --mode finetune      
    --real_percent 0.1   
)

cd /home/noayaak/final_project/Introduction_to_deep_learning_project/code/
python -u main.py "${ARGS[@]}"
    


    
