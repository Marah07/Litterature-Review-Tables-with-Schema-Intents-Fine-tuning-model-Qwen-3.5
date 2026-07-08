import torch
from vllm import LLM

MODEL_NAME = "/lustre/fswork/projects/rech/ruk/uab84ny/hf_cache/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

num_gpus = torch.cuda.device_count()

print(f"Detected GPUs: {num_gpus}")

print("Loading model...")

llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=num_gpus,
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.60,
    enforce_eager=True,
    disable_custom_all_reduce=True,
)

print("Model loaded successfully!")

input("Press Enter to exit...")
