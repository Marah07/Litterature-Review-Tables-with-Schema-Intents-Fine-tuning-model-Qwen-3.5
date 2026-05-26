import json
import os
from tqdm import tqdm

INPUT_PATH = "data/raw/tables_medium_quality.jsonl"
OUTPUT_PATH = "data/processed/prepared_dataset.jsonl"


def parse_table(table_json_str):
    """
    Convert nested table JSON into linear text representation.
    """
    table = json.loads(table_json_str)

    lines = []
    for col_name, col_values in table.items():
        for paper_id, values in col_values.items():
            value = values[0] if isinstance(values, list) else values
            lines.append(f"{col_name} | {paper_id} | {value}")

    return "\n".join(lines)


def build_paper_text(row_bib_map):
    """
    Concatenate titles + abstracts of all referenced papers.
    """
    papers = []
    for item in row_bib_map:
        title = item.get("title", "").strip()
        abstract = item.get("abstract", "").strip()

        if not title and not abstract:
            continue

        papers.append(f"Title: {title}\nAbstract: {abstract}")

    return "\n\n".join(papers), papers


def main():
    os.makedirs("data/processed", exist_ok=True)

    with open(INPUT_PATH, "r") as f, open(OUTPUT_PATH, "w") as out:
        for line in tqdm(f):
            ex = json.loads(line)

            tabid = ex.get("tabid")
            arxiv_id = ex.get("arxiv_id")

            table_text = parse_table(ex["table"])
            paper_text, papers = build_paper_text(ex["row_bib_map"])

            # skip empty cases (safety)
            if not table_text or not paper_text:
                continue

            out_obj = {
                "table_text": table_text,
                "paper_text": paper_text,
            }

            out.write(json.dumps(out_obj) + "\n")

    print(f"Saved to {OUTPUT_PATH}")


    with open("data/processed/prepared_dataset.jsonl") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            print(ex["table_text"][:300])
            print(ex["paper_text"][:300])
            break


if __name__ == "__main__":
    main()