import os
import json
from tqdm import tqdm

INPUT_PATH = "data/final/final_intents.jsonl"
OUTPUT_PATH = "data/training/schema_generation_examples.jsonl"


SYSTEM_PROMPT = """
You are an intelligent and precise assistant that can understand the contents of research papers.
You are knowledgeable on different scientific domains.
Your task is to generate literature review table schemas.
""".strip()


def extract_schema(table_text):

    try:
        table_json = json.loads(table_text)

        columns = list(table_json.keys())

        return {
            "columns": columns
        }

    except Exception:

        return {
            "columns": []
        }


def build_input_text(intent, paper_text):

    return f"""
Table Intent:
{intent}

Papers:
{paper_text}
""".strip()


def main():

    os.makedirs("data/training", exist_ok=True)

    with open(INPUT_PATH, "r") as infile, \
         open(OUTPUT_PATH, "w") as outfile:

        for line in tqdm(infile):

            item = json.loads(line)

            schema = extract_schema(item["table_text"])

            input_text = build_input_text(
                item["best_intent"],
                item["paper_text"]
            )

            result = {
                "tabid": item["tabid"],
                "arxiv_id": item["arxiv_id"],
                "system_prompt": SYSTEM_PROMPT,
                "input": input_text,
                "output": json.dumps(schema)
            }

            outfile.write(json.dumps(result) + "\n")

    print(f"Saved training examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()