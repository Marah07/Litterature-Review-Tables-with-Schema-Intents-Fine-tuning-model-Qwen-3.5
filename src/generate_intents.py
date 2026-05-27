import os
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

from prompts import S2_SYSTEM_PROMPT, GENERATE_SYNTHETIC_GOALS_PAPERS_QUESTION


INPUT_PATH = "data/processed/prepared_dataset.jsonl"
OUTPUT_PATH = "data/intents/candidate_intents.jsonl"

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 3000
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
        tensor_parallel_size=4,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        n=NUM_CANDIDATES
    )

    examples = load_examples(INPUT_PATH)

    # temporary debug subset
    examples = examples[:3]


    prompts = []

    for ex in examples:
        user_prompt = build_prompt(ex)

        full_prompt = (
            f"<|system|>\n{S2_SYSTEM_PROMPT}\n"
            f"<|user|>\n{user_prompt}\n"
            f"<|assistant|>\n"
        )

        prompts.append(full_prompt)

    print("Generating intents...")

    outputs = llm.generate(prompts, sampling_params)

    with open(OUTPUT_PATH, "w") as out:

        for ex, output in tqdm(zip(examples, outputs), total=len(outputs)):

            candidates = []

            for candidate in output.outputs:
                candidates.append(candidate.text.strip())

            result = {
                "tabid": ex["tabid"],
                "arxiv_id": ex["arxiv_id"],
                "table_text": ex["table_text"],
                "paper_text": ex["paper_text"],
                "candidate_goals": candidates
            }

            out.write(json.dumps(result) + "\n")

    print(f"Saved outputs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()