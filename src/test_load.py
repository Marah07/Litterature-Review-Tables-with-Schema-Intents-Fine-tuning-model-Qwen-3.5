from vllm import LLM

llm = LLM(
    model="/lustre/fswork/projects/rech/ruk/uab84ny/llama/checkpoints/Llama3.3-70B-Instruct",
    load_format="meta",
    tensor_parallel_size=4,
    dtype="bfloat16",
    trust_remote_code=True,
)

print("OK")
