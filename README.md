# Intent-Aware Schema Generation — Reproduction & Extension

This repository reproduces and extends the pipeline from:

> **Intent-Aware Schema Generation and Refinement for Literature Review Tables**
> Padmakumar et al., EMNLP 2025

The goal is to fine-tune a small open-weight model (Qwen2.5-4B) to generate literature review table schemas guided by a **table intent** — an open-ended question describing the information need behind the table.

This work is also related to:

> **TableFact: Evaluating LLM-as-a-Judge for Factual Verification of Literature Review Tables**
> Baccari, Boudin, Dufour, LS2N — Nantes Université

which evaluates the factual reliability of LLMs as judges for generated table content.

---

## Project Structure

```
intent-schema-project/
├── src/                        # All Python scripts
│   ├── prepare_data.py
│   ├── generate_intents.py
│   ├── score_intents.py
│   ├── select_best_intent.py
│   ├── create_training_examples.py
│   ├── build_sft_dataset.py
│   ├── train_sft.py
│   └── prompts.py
├── slurm/                      # SLURM job scripts for Jean Zay
│   ├── job_prepare_data.sh
│   ├── job_generate_intents.sh
│   ├── job_score_intents.sh
│   └── job_train_sft.sh
├── data/
│   ├── processed/              # Output of prepare_data.py
│   ├── intents/                # Output of generate_intents.py
│   ├── scored/                 # Output of score_intents.py
│   ├── final/                  # Output of select_best_intent.py
│   ├── training/               # Output of create_training_examples.py
│   └── sft/                    # Output of build_sft_dataset.py
├── models/                     # Fine-tuned model checkpoints
├── logs/                       # SLURM job logs
└── README.md
```

---

## Pipeline Overview

The pipeline consists of 6 sequential steps:

```
ArxivDIGESTables-Silver (22k tables)
        │
        ▼
1. prepare_data.py          → data/processed/prepared_dataset.jsonl
        │
        ▼
2. generate_intents.py      → data/intents/candidate_intents.jsonl
        │
        ▼
3. score_intents.py         → data/scored/scored_intents.jsonl
        │
        ▼
4. select_best_intent.py    → data/final/final_intents.jsonl
        │
        ▼
5. create_training_examples.py → data/training/schema_generation_examples.jsonl
        │
        ▼
6. build_sft_dataset.py     → data/sft/  (train + validation splits)
        │
        ▼
7. train_sft.py             → models/qwen_schema_sft/
```

---

## Script Descriptions

### `src/prepare_data.py`
Loads the ArxivDIGESTables-Silver dataset from HuggingFace (`Tabellio/ArxivDIGESTables`) and converts it into a flat JSONL format suitable for the rest of the pipeline.

**Input:** HuggingFace dataset (downloaded automatically)
**Output:** `data/processed/prepared_dataset.jsonl`

Each output record contains:
- `tabid`: unique table identifier
- `arxiv_id`: arXiv paper ID
- `table_text`: linearized table content (`column | paper_id | value`)
- `table_json`: original table JSON (used for schema extraction)
- `paper_text`: concatenated titles and abstracts of referenced papers
- `caption`: table caption (empty if unavailable)
- `in_text_refs`: in-text references to the table (empty if unavailable)

---

### `src/generate_intents.py`
Generates 5 candidate table intents per example using **Qwen3.6-35B-A3B** via vLLM. A table intent is an open-ended question describing the information need behind a literature review table (e.g., *"How do different network pruning methods compare in terms of accuracy and compression rate?"*).

This step replaces the original paper's use of GPT-4o with a local open-weight model.

**Input:** `data/processed/prepared_dataset.jsonl`
**Output:** `data/intents/candidate_intents.jsonl`

Key parameters (set at top of script):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | path to Qwen3.6-35B | Local model path on Jean Zay |
| `NUM_CANDIDATES` | 5 | Number of intent candidates per example |
| `TEMPERATURE` | 0.7 | Sampling temperature |
| `MAX_TOKENS` | 500 | Max output tokens per candidate |

> **Note:** The script supports **resume** — if interrupted, it skips already-processed examples on restart.

---

### `src/score_intents.py`
Scores the 5 candidate intents using **LLaMA-3.3-70B** as an LLM-as-a-judge. The judge selects the best intent based on how well it matches the table's actual content and purpose.

This replaces the original paper's use of GPT-4o as judge.

**Input:** `data/intents/candidate_intents.jsonl`
**Output:** `data/scored/scored_intents.jsonl`

Key parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | path to LLaMA-3.3-70B | Local model path on Jean Zay |
| `TEMPERATURE` | 0.0 | Greedy decoding for deterministic judgment |

> **Note:** Supports resume on restart.

---

### `src/select_best_intent.py`
Parses the judge output and extracts the best intent for each example. Handles JSON parsing failures and fallback strategies.

**Input:** `data/scored/scored_intents.jsonl`
**Output:** `data/final/final_intents.jsonl`

Each output record adds:
- `best_intent`: the selected intent text
- `judge_justification`: the judge's reasoning
- `raw_judge_output`: full raw judge response

---

### `src/create_training_examples.py`
Formats the data into input/output pairs for supervised fine-tuning. The input is the table intent + paper titles/abstracts; the output is the table schema (list of column names in JSON format).

**Input:** `data/final/final_intents.jsonl`
**Output:** `data/training/schema_generation_examples.jsonl`

Output format per example:
```json
{
  "tabid": "...",
  "system_prompt": "You are an intelligent and precise assistant...",
  "input": "Table Intent:\n...\nPapers:\n...",
  "output": "{\"columns\": [\"Method\", \"Dataset\", \"Accuracy\"]}"
}
```

---

### `src/build_sft_dataset.py`
Converts training examples into a HuggingFace `DatasetDict` with chat-formatted messages. Splits into train (95%) and validation (5%), matching the paper's split of ~21k train / ~1.1k validation.

**Input:** `data/training/schema_generation_examples.jsonl`
**Output:** `data/sft/` (HuggingFace Arrow format + `train.jsonl` / `val.jsonl`)

---

### `src/train_sft.py`
Fine-tunes **Qwen2.5-4B-Instruct** on the schema generation task using supervised fine-tuning (SFT) with the TRL library.

**Input:** `data/sft/`
**Output:** `models/qwen_schema_sft/`

Key parameters (passed via CLI):
```bash
python src/train_sft.py --learning_rate 2e-5 --output_dir models/qwen_schema_sft
```

Training configuration:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_train_epochs` | 4 | As in the paper |
| `per_device_train_batch_size` | 4 | Per GPU |
| `gradient_accumulation_steps` | 4 | Effective batch ~64 |
| `learning_rate` | sweep 1e-6 to 1e-4 | Select best via validation loss |
| `max_seq_length` | 4096 | Max input length |
| `bf16` | True | Bfloat16 precision |

---

### `src/prompts.py`
Contains all prompt templates used across the pipeline:
- `S2_SYSTEM_PROMPT`: system prompt for all LLM calls
- `GENERATE_SYNTHETIC_GOALS_PAPERS_QUESTION`: prompt for intent generation
- `EVALUATE_GOALS_TO_TABLE`: prompt for LLM-as-judge scoring

---

## Data

### Source Dataset
**ArxivDIGESTables-Silver** — "medium quality" subset of ArxivDIGESTables.
- HuggingFace: `Tabellio/ArxivDIGESTables`
- Split used: `validation` (22,283 examples)
- Features: `tabid`, `table`, `row_bib_map`, `arxiv_id`

The dataset contains literature review tables from scientific papers along with titles and abstracts of the referenced papers. Captions and in-text references are not available in this subset.

---

## Environment Setup

### Requirements
- Python 3.11+
- CUDA 12.x
- Jean Zay HPC cluster (A100 80GB GPUs)

### Installation
```bash
# On Jean Zay login node
module purge
module load pytorch-gpu/py3/2.4.0

python -m venv $WORK/env_intent
source $WORK/env_intent/bin/activate

pip install vllm transformers datasets trl accelerate scikit-learn tqdm peft
```

### Required environment variables
Add to `~/.bashrc`:
```bash
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
export FLASHINFER_WORKSPACE_BASE=$SCRATCH
export XDG_CACHE_HOME=$SCRATCH/cache
export VLLM_CONFIG_ROOT=$SCRATCH/vllm_config
export HF_HOME=$WORK/hf_cache
export TORCH_HOME=$SCRATCH/torch_cache
export TMPDIR=$SCRATCH/tmp
export NCCL_NET_PLUGIN=none
export NCCL_NET=Socket
export PYTHONNOUSERSITE=1
```

> **Important:** `NCCL_NET=Socket` is required on Jean Zay A100 nodes to avoid a segfault caused by incompatibility between the system NCCL plugin and vLLM's bundled NCCL.

---

## Running on Jean Zay

### Step 1 — Prepare data
```bash
sbatch slurm/job_prepare_data.sh
```
Uses `ruk@v100` / `gpu_p13` partition. Takes ~5 minutes.

### Step 2 — Generate intents
```bash
# Single job (resumes automatically if interrupted)
sbatch slurm/job_generate_intents.sh

# Chain multiple jobs for full dataset (~6 runs of 20h needed)
JOB1=$(sbatch --parsable slurm/job_generate_intents.sh)
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 slurm/job_generate_intents.sh)
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 slurm/job_generate_intents.sh)
```
Uses `ruk@a100` / `gpu_p5` partition with 4 GPUs.

### Step 3 — Score intents
```bash
sbatch slurm/job_score_intents.sh
```

### Steps 4–6 — Post-processing
```bash
# These are lightweight and can run on v100
python src/select_best_intent.py
python src/create_training_examples.py
python src/build_sft_dataset.py
```

### Step 7 — Fine-tune
```bash
sbatch slurm/job_train_sft.sh
```

---

## Models Used

| Role | Model | Notes |
|------|-------|-------|
| Intent generation | Qwen3.6-35B-A3B | Replaces GPT-4o from the paper |
| Intent scoring (judge) | LLaMA-3.3-70B-Instruct | Replaces GPT-4o from the paper |
| Fine-tuning target | Qwen2.5-4B-Instruct | Same as paper |

All models are loaded from local paths on Jean Zay (`$WORK/hf_cache/`).

---

## Key Differences from the Original Paper

| Aspect | Paper | This repo |
|--------|-------|-----------|
| Intent generation model | GPT-4o-2024-08-06 | Qwen3.6-35B-A3B |
| Intent scoring model | GPT-4o-2024-08-06 | LLaMA-3.3-70B-Instruct |
| Fine-tuning models | Qwen2.5-3B + LLaMA-3.2-3B | Qwen2.5-4B |
| Infrastructure | Unspecified | Jean Zay (IDRIS), A100 80GB |

---

## References

```
@inproceedings{padmakumar2025intent,
  title={Intent-Aware Schema Generation and Refinement for Literature Review Tables},
  author={Padmakumar, Vishakh and Chang, Joseph Chee and Lo, Kyle and Downey, Doug and Naik, Aakanksha},
  booktitle={Findings of EMNLP 2025},
  year={2025}
}

@inproceedings{newman2024arxivdigestables,
  title={ArxivDIGESTables: Synthesizing Scientific Literature into Tables using Language Models},
  author={Newman, Benjamin and others},
  booktitle={EMNLP 2024},
  year={2024}
}
```
