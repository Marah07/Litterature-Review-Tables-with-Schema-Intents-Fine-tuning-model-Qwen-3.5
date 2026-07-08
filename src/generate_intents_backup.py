import os
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

from prompts import S2_SYSTEM_PROMPT, GENERATE_SYNTHETIC_GOALS_PAPERS_QUESTION


INPUT_PATH = "data/processed/prepared_dataset.jsonl"
OUTPUT_PATH = "data/intents/candidate_intents_full_run.jsonl"

MODEL_NAME = "/lustre/fswork/projects/rech/ruk/uab84ny/hf_cache/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 1024
NUM_CANDIDATES = 5


def build_prompt(example):
    return GENERATE_SYNTHETIC_GOALS_PAPERS_QUESTION.format(
        table=example["table_text"],
        papers=example["paper_text"]
    )


def load_examples(path):
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def main():

    os.makedirs("data/intents", exist_ok=True)

    print("Loading model...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        enforce_eager=True,  # keep this to avoid cuda graph cache issues
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        n=NUM_CANDIDATES,
        # chat_template_kwargs={"enable_thinking": False}
    )

    examples = load_examples(INPUT_PATH)

    # temporary debug subset
    # examples = examples[:1]

    print(f"Loaded {len(examples)} examples")
    print(f"Writing results to: {OUTPUT_PATH}")
    print("Building prompts...")

    BATCH_SIZE = 100
    tokenizer = llm.get_tokenizer()
    with open(OUTPUT_PATH, "w") as out:

        for start in range(0, len(examples), BATCH_SIZE):

            batch = examples[start:start+BATCH_SIZE]

            prompts = []

            for ex in batch:

                user_prompt = build_prompt(ex)

                full_prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": S2_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                prompts.append(full_prompt)

            print(
                f"Processing batch {start//BATCH_SIZE + 1} "
                f"({start} -> {start + len(batch)})"
            )

            outputs = llm.generate(prompts, sampling_params)

            for ex, output in zip(batch, outputs):

                candidates = [
                    candidate.text.strip()
                    for candidate in output.outputs
                ]

                result = {
                    "tabid": ex["tabid"],
                    "arxiv_id": ex["arxiv_id"],
                    "table_text": ex["table_text"],
                    "paper_text": ex["paper_text"],
                    "candidate_goals": candidates,
                }

                out.write(json.dumps(result) + "\n")

            out.flush()
            print(f"Completed {min(start + BATCH_SIZE, len(examples))}/{len(examples)} examples")


if __name__ == "__main__":
    main()
