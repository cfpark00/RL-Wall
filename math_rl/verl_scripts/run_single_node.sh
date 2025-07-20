#!/bin/sh

#SBATCH --cpus-per-task=24        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 1-16:00 # Runtime in minutes
#SBATCH -p kempner_h100 # Partition to submit to
#SBATCH --mem=250G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this file
#SBATCH --account=kempner_barak_lab
#SBATCH --job-name=grpo_1.5B_S1


module purge
module load Mambaforge
module load cuda cudnn
module load gcc/12.2.0-fasrc01 
mamba activate verl

cd /n/home05/sqin/wall/verl
pwd


python3 -m verl.trainer.main_ppo sunny_scripts/grpo_math_qwen_1.5Bs1.yaml