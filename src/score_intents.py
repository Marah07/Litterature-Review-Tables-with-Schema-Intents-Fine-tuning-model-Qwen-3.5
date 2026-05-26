import os
import json
from tqdm import tqdm

from vllm import LLM, SamplingParams

from prompts import S2_SYSTEM_PROMPT, EVALUATE_GOALS_TO_TABLE

INPUT_PATH = "data/intents/candidate_intents.jsonl"
OUTPUT_PATH = "data/scored/scored_intents.jsonl"

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 3000


def load_examples(path):
    examples = []

    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    return examples


def build_goal_text(candidate_goals):

    lines = []

    for idx, goal in enumerate(candidate_goals):
        lines.append(f"Candidate {idx+1}:\n{goal}\n")

    return "\n".join(lines)


def build_prompt(example):

    goal_text = build_goal_text(example["candidate_goals"])

    return EVALUATE_GOALS_TO_TABLE.format(
        table=example["table_text"],
        goal_text=goal_text
    )


def main():

    os.makedirs("data/scored", exist_ok=True)

    print("Loading judge model...")

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
        max_tokens=MAX_TOKENS
    )

    #examples = load_examples(INPUT_PATH)

    # debug mode
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

    print("Scoring candidate intents...")

    outputs = llm.generate(prompts, sampling_params)

    with open(OUTPUT_PATH, "w") as out:

        for ex, output in tqdm(zip(examples, outputs), total=len(outputs)):

            result = {
                "tabid": ex["tabid"],
                "arxiv_id": ex["arxiv_id"],
                "table_text": ex["table_text"],
                "paper_text": ex["paper_text"],
                "candidate_goals": ex["candidate_goals"],
                "judge_output": output.outputs[0].text.strip()
            }

            out.write(json.dumps(result) + "\n")

    print(f"Saved scored outputs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()