import os
import torch
from datasets import load_from_disk
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)

from trl import SFTTrainer


<<<<<<< Updated upstream
MODEL_NAME = "Qwen/Qwen3.5-4B-Instruct"
=======
MODEL_NAME = "/lustre/fswork/projects/rech/ruk/uab84ny/hf_cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
>>>>>>> Stashed changes
DATA_PATH = "data/sft"
OUTPUT_DIR = "models/qwen_schema_sft"

MAX_SEQ_LEN = 4096

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()

def main():
    args_cli = parse_args()

    print("Loading dataset...")
    dataset = load_from_disk(DATA_PATH)

    print(dataset)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

    def format_example(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
        }

    train_dataset = dataset["train"].map(format_example)
    eval_dataset = dataset["validation"].map(format_example)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args_cli.learning_rate,
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        bf16=True,
        report_to="none",
        optim="adamw_torch_fused",          # faster than adamw_torch on A100
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,           # parallel data loading
        group_by_length=True,               # reduces padding, speeds up training
        ddp_find_unused_parameters=False,   # cleaner DDP on Jean Zay
)

    print("Starting SFT training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args
    )

    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
