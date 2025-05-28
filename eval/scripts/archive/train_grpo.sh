#!/bin/sh

#SBATCH -c 32 # Number of cores requested
#SBATCH -t 0-16:00 # Runtime in minutes
#SBATCH -p kempner_h100 # Partition to submit to
#SBATCH --mem=250G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH -o ../slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e ../slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_barak_lab
#SBATCH --exclude=holygpu8a19604,holygpu8a19303,holygpu8a17402,holygpu8a17402

module purge
module load Mambaforge
module load cuda cudnn
mamba activate wall

cd ../


accelerate launch \
  --config_file configs/accelerate.yaml \
   train_grpo.py \
  --config configs/train_grpo.conf 


