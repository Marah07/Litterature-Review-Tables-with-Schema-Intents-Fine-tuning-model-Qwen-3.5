import os
import torch
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)

from trl import SFTTrainer


MODEL_NAME = "Qwen/Qwen3.5-4B-Instruct"
DATA_PATH = "data/sft"
OUTPUT_DIR = "models/qwen_schema_sft"

MAX_SEQ_LEN = 4096


def main():

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
        device_map="auto",
        trust_remote_code=True
    )

    def format_example(example):

        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

    print("Formatting dataset...")

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        bf16=True,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine"
    )

    print("Starting SFT training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=format_example,
        max_seq_length=MAX_SEQ_LEN,
        args=training_args
    )

    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()