#!/bin/bash
#SBATCH --job-name=score_intents

#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.err

#SBATCH -A ruk@a100
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100

#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16

#SBATCH --time=20:00:00

source ~/activate_intent.sh

export PYTHONPATH=$WORK/intent-schema-project/src:$PYTHONPATH

export HF_HOME=$WORK/hf_cache
export TRITON_CACHE_DIR=$WORK/triton_cache
export XDG_CACHE_HOME=$WORK/xdg_cache

python src/score_intents.py
