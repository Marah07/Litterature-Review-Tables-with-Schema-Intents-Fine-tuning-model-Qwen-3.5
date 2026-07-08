import os
import json
from tqdm import tqdm

from vllm import LLM, SamplingParams

from prompts import (
    S2_SYSTEM_PROMPT,
    EVALUATE_GOALS_TO_TABLE,
)

INPUT_PATH = "data/intents/candidate_intents_for_judge.jsonl"
OUTPUT_PATH = "data/scored/scored_intents.jsonl"

MODEL_NAME = "/lustre/fswork/projects/rech/ruk/uab84ny/llama/checkpoints/Llama3.3-70B-Instruct"

TEMPERATURE = 0.7
TOP_P = 1.0
MAX_TOKENS = 2000

BATCH_SIZE = 100

DEBUG = False
DEBUG_N = 1


def load_examples(path):
    examples = []

    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))

    return examples


def build_goal_text(candidate_goals):

    lines = []

    for idx, goal in enumerate(candidate_goals):

        lines.append(
            f"Candidate {idx+1}:\n"
            f"{goal['goal']}\n"
        )

    return "\n".join(lines)


def build_prompt(example):

    goal_text = build_goal_text(
        example["candidate_goals"]
    )

    return EVALUATE_GOALS_TO_TABLE.format(
        table=example["table_text"],
        goal_text=goal_text,
    )


def main():

    os.makedirs("data/scored", exist_ok=True)

    print("Loading model...")

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        load_format="meta",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=32768,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    examples = load_examples(INPUT_PATH)

    if DEBUG:
        examples = examples[:DEBUG_N]

    print("Examples:", len(examples))

    with open(OUTPUT_PATH, "w") as fout:

        for start in tqdm(
            range(0, len(examples), BATCH_SIZE)
        ):

            batch = examples[start:start+BATCH_SIZE]

            prompts = []

            for ex in batch:

                user_prompt = build_prompt(ex)

                prompt = (
                    llm.llm_engine
                    .tokenizer
                    .tokenizer
                    .apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": S2_SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

                prompts.append(prompt)

            outputs = llm.generate(
                prompts,
                sampling_params,
            )

            for ex, output in zip(batch, outputs):

                row = {
                    "tabid": ex["tabid"],
                    "arxiv_id": ex["arxiv_id"],
                    "candidate_goals": ex["candidate_goals"],
                    "judge_output":
                        output.outputs[0].text.strip(),
                }

                fout.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            fout.flush()

    print(
        f"Saved results to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
