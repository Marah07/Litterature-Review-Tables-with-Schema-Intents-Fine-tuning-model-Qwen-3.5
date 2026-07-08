"""
Lire le fichier brut.
Extraire le JSON final de chaque candidat.
Construire directement le fichier pour le judge.
"""
import json
import re
from pathlib import Path

INPUT_FILE = "data/intents/all_intents.jsonl"
OUTPUT_FILE = "data/intents/candidate_intents_for_judge.jsonl"

def is_valid(parsed):
    return (
        isinstance(parsed, dict)
        and "goal" in parsed
        and "justification" in parsed
        and len(parsed["goal"]) > 0
        and len(parsed["justification"]) > 0
    )

def extract_json_from_candidate(text):

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 1. JSON block
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)

    for block in reversed(json_blocks):
        try:
            obj = json.loads(block)
            if is_valid(obj):
                return obj
        except:
            pass

    # 2. brute force JSON scan
    decoder = json.JSONDecoder()

    for i in range(len(text)):
        if text[i] == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if is_valid(obj):
                    return obj
            except:
                continue

    return None

def main():

    total = 0
    parsed_ok = 0
    failed_examples = []

    with open(INPUT_FILE) as fin, open(OUTPUT_FILE, "w") as fout:

        for line in fin:

            row = json.loads(line)

            candidates = []

            for candidate_text in row["candidate_goals"]:

                parsed = extract_json_from_candidate(candidate_text)

                if parsed is not None:

                    candidates.append({
                        "goal": parsed["goal"],
                        "justification": parsed["justification"]
                    })

                    parsed_ok += 1
                else:
                    if len(failed_examples) < 20:
                        failed_examples.append(candidate_text)

            total += len(row["candidate_goals"])

            output_row = {
                "tabid": row["tabid"],
                "arxiv_id": row["arxiv_id"],
                "table_text": row["table_text"],
                "paper_text": row["paper_text"],
                "candidate_goals": candidates
            }

            fout.write(
                json.dumps(output_row, ensure_ascii=False)
                + "\n"
            )

            with open("failed_candidates.txt", "w") as f:
                for ex in failed_examples:
                    f.write(ex)
                    f.write("\n\n" + "="*80 + "\n\n")

    print(f"Total candidates : {total}")
    print(f"Parsed OK        : {parsed_ok}")
    print(f"Output           : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
