#!/bin/sh

#SBATCH --cpus-per-task=24        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 0-04:00 # Runtime in minutes
#SBATCH -p gpu # Partition to submit to
#SBATCH --mem=120G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this file
#SBATCH --account=barak_lab
#SBATCH --job-name=eval_grpo


module purge
module load Mambaforge
module load cuda cudnn
module load gcc/12.2.0-fasrc01 
mamba activate verl

cd /n/home05/sqin/wall/verl/eval

set -x 

python generate_responses.py --model_name llama-3.1-8b-gsm8k --dataset_name math_500 --temperature 1.2 --n 64 --seed=0 --max_tokens=1024

