#!/bin/bash
#SBATCH --job-name=gen_intents_h100
#SBATCH --output=logs/gen_h100_%j.out
#SBATCH --error=logs/gen_h100_%j.err
#SBATCH -A ruk@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

source ~/.bashrc
cd $WORK/intent-schema-project
module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
source $WORK/env_intent_h100/bin/activate

export PYTHONPATH=$WORK/intent-schema-project/src:$PYTHONPATH
export VLLM_GDN_PREFILL_BACKEND=triton
export FLASHINFER_JIT_DISABLE=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=60   # Prevents early worker timeouts
python src/generate_intents_h100.py
