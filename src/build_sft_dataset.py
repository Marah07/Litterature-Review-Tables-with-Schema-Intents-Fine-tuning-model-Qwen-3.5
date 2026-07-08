import os
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/training/schema_generation_examples.jsonl"
OUTPUT_DIR = "data/sft"


TRAIN_SIZE = 0.95
RANDOM_SEED = 42


def load_examples(path):

    examples = []

    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    return examples


def build_chat_example(example):

    return {
        "messages": [
            {
                "role": "system",
                "content": example["system_prompt"]
            },
            {
                "role": "user",
                "content": example["input"]
            },
            {
                "role": "assistant",
                "content": example["output"]
            }
        ]
    }


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading examples...")
    examples = load_examples(INPUT_PATH)

    print(f"Loaded {len(examples)} examples")

    print("Building chat-formatted dataset...")

    formatted_examples = [
        build_chat_example(ex)
        for ex in examples
    ]

    train_examples, val_examples = train_test_split(
        formatted_examples,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    dataset_dict.save_to_disk(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "train.jsonl"), "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(os.path.join(OUTPUT_DIR, "val.jsonl"), "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print("\nDataset created successfully.")
    print(dataset_dict)

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
