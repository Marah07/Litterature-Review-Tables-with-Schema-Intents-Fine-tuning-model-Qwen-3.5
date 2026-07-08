import os
import re
import json
from tqdm import tqdm

INPUT_PATH = "data/scored/scored_intents.jsonl"
OUTPUT_PATH = "data/final/final_intents.jsonl"


def extract_json(text):
    """
    Try extracting JSON from raw LLM output.
    """

    # remove markdown fences if present
    text = text.strip()

    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        text = text[start:end].strip()

    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()

    # try direct parsing
    try:
        return json.loads(text)
    except:
        pass

    # try extracting first {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None


def normalize_text(text):

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def find_best_goal(best_goal, candidate_goals):
    if not isinstance(best_goal, str):
        return candidate_goals[0]
    
    normalized_best = normalize_text(best_goal)
    
    # none fallback — check first before any matching
    if normalized_best == "none":
        return candidate_goals[0]
    
    # exact/partial text matching
    for candidate in candidate_goals:
        normalized_candidate = normalize_text(candidate)
        if normalized_best in normalized_candidate:
            return candidate
        if normalized_candidate in normalized_best:
            return candidate
    
    # candidate reference matching
    candidate_patterns = {
        "candidate 1": 0,
        "candidate 2": 1,
        "candidate 3": 2,
        "candidate 4": 3,
        "candidate 5": 4,
    }
    for pattern, idx in candidate_patterns.items():
        if pattern in normalized_best and idx < len(candidate_goals):
            return candidate_goals[idx]
    
    # final fallback
    print(f"[WARN] Could not match best_goal to any candidate, falling back to candidate 1. best_goal was: {normalized_best[:100]}")
    return candidate_goals[0]

def main():

    os.makedirs("data/final", exist_ok=True)

    total = 0
    parsed = 0
    failed = 0

    with open(INPUT_PATH, "r") as infile, \
         open(OUTPUT_PATH, "w") as outfile:

        for line in tqdm(infile):

            total += 1

            item = json.loads(line)

            judge_output = item["judge_output"]

            parsed_json = extract_json(judge_output)

            if parsed_json is None:

                failed += 1

                best_choice = item["candidate_goals"][0]

                justification = "FAILED_TO_PARSE"

            else:

                parsed += 1

                best_goal = parsed_json.get("best_goal", "")

                best_choice = find_best_goal(
                    best_goal,
                    item["candidate_goals"]
                )

                justification = parsed_json.get(
                    "justification",
                    ""
                )

            result = {
                "tabid": item["tabid"],
                "arxiv_id": item["arxiv_id"],
                "table_text": item["table_text"],
                "paper_text": item["paper_text"],
                "candidate_goals": item["candidate_goals"],
                "best_intent": best_choice,
                "judge_justification": justification,
                "raw_judge_output": judge_output
            }

            outfile.write(json.dumps(result) + "\n")

    print("\nDone.")
    print(f"Total examples: {total}")
    print(f"Successfully parsed: {parsed}")
    print(f"Failed parsing: {failed}")
    if total > 0:
        fail_rate = failed / total * 100
        if fail_rate > 10:
            print(f"[WARN] High parse failure rate: {fail_rate:.1f}% — check judge output format")


if __name__ == "__main__":
    main()
