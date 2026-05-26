#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH --job-name=qwen_intents
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/generate_%j.out

module purge
module load pytorch-gpu/py3/2.5.0

source ~/vllm_env_clean/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

cd $WORK/intent-schema-project

python src/generate_intents.py