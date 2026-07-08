"""
score_intents.py  —  LLM-as-judge avec transformers (pas vLLM)
=================================================================
Utilise LLaMA-3.3-70B-Instruct via transformers.pipeline pour scorer
les 5 candidats d'intention par exemple et sélectionner le meilleur.

Format d'entrée attendu (data/intents/candidate_intents_for_judge.jsonl) :
{
  "tabid": str,
  "arxiv_id": str,
  "table_text": str,
  "paper_text": str,
  "candidate_goals": [
    {"goal": str, "justification": str},
    ... (5 fois)
  ]
}

Pourquoi transformers et pas vLLM :
  - vLLM nécessite un format de checkpoint et une config.json spécifiques
    que le chemin local fourni ne respecte pas (erreur "Invalid repository ID").
  - transformers.pipeline avec device_map="auto" répartit automatiquement
    le modèle de 70B de paramètres sur les 4 GPUs A100.

Pourquoi device_map="auto" est sûr ici :
  - On N'UTILISE PAS torchrun / DDP (un seul processus Python).
  - device_map="auto" répartit les COUCHES du modèle entre les GPUs
    disponibles (model parallelism), adapté à une inférence en un seul
    processus.

Traitement séquentiel :
  - pipeline() ne fait pas de vrai batching dynamique comme vLLM.
  - Sauvegarde au fil de l'eau (flush après chaque écriture).

Usage :
    # Test sur 1 seul exemple d'abord :
    python src/score_intents.py --debug

    # Run complet (22283 exemples) :
    python src/score_intents.py
"""

import argparse
import glob
import json
import os
import sys

import torch
import transformers
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from prompts import S2_SYSTEM_PROMPT, EVALUATE_GOALS_TO_TABLE


INPUT_PATH  = "data/intents/candidate_intents_for_judge.jsonl"
OUTPUT_PATH = "data/scored/scored_intents.jsonl"


def find_model_path() -> str:
    """
    Résout le chemin du snapshot HuggingFace téléchargé.

    Historique : la commande de téléchargement initiale a été lancée avec
    cache_dir='$SCRATCH/hf_cache' en tant que CHAÎNE LITTÉRALE (le $SCRATCH
    n'a pas été interprété par bash dans le contexte python3 -c "..."), donc
    snapshot_download a créé un dossier nommé littéralement "$SCRATCH" à
    l'intérieur du répertoire courant (intent-schema-project/).

    Les 141GB de poids (.safetensors) et le tokenizer sont donc ici :
        <project_dir>/$SCRATCH/hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/
                                         snapshots/<hash>/

    On résout ce chemin par rapport au répertoire du projet (parent de src/),
    avec un fallback sur $WORK/hf_cache et $SCRATCH/hf_cache au cas où le
    modèle serait déplacé/re-téléchargé proprement plus tard.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    candidates = []

    # 1. Emplacement réel actuel (dossier littéralement nommé "$SCRATCH")
    candidates.extend(glob.glob(os.path.join(
        project_dir, "$SCRATCH", "hf_cache",
        "models--meta-llama--Llama-3.3-70B-Instruct", "snapshots", "*",
    )))

    # 2. Emplacements propres (si le modèle est re-téléchargé correctement)
    for base_env in ("WORK", "SCRATCH"):
        base = os.environ.get(base_env)
        if not base:
            continue
        candidates.extend(glob.glob(os.path.join(
            base, "hf_cache",
            "models--meta-llama--Llama-3.3-70B-Instruct", "snapshots", "*",
        )))

    # Ne garder que les dossiers qui contiennent réellement un config.json
    # (filtre les snapshots vides comme celui dans $WORK/hf_cache/)
    valid = [c for c in candidates if os.path.exists(os.path.join(c, "config.json"))]

    if not valid:
        raise FileNotFoundError(
            "Aucun snapshot Llama-3.3-70B-Instruct valide trouvé "
            "(avec config.json). Chemins testés:\n" + "\n".join(candidates)
        )

    valid = sorted(valid, key=os.path.getmtime)
    return valid[-1]


MODEL_ID = find_model_path()

TEMPERATURE = 0.0    # greedy — recommandé pour un juge (déterministe et reproductible)
TOP_P       = 1.0
MAX_NEW_TOKENS = 3000


def build_goal_text(candidate_goals: list) -> str:
    """
    Formate les candidats pour le prompt du juge.
    candidate_goals est une liste de dicts {"goal": ..., "justification": ...}
    déjà propres (pas de parsing nécessaire).
    """
    lines = []
    for idx, candidate in enumerate(candidate_goals, 1):
        lines.append(f"Candidate {idx}:\n{candidate['goal']}")
    return "\n\n".join(lines)


def build_prompt(example: dict) -> str:
    goal_text = build_goal_text(example["candidate_goals"])
    return EVALUATE_GOALS_TO_TABLE.format(
        table     = example["table_text"],
        goal_text = goal_text,
    )


def load_examples(path: str) -> list:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_done_tabids(path: str) -> set:
    """Charge les tabids déjà traités pour reprendre un job interrompu."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(str(json.loads(line)["tabid"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true",
        help="Traite seulement le premier exemple (test rapide).",
    )
    args = parser.parse_args()

    os.makedirs("data/scored", exist_ok=True)

    # ── Charger les exemples ──────────────────────────────────────────────────
    print(f"Chargement de {INPUT_PATH} ...")
    examples = load_examples(INPUT_PATH)
    print(f"{len(examples)} exemples chargés.")

    if args.debug:
        examples = examples[:1]
        print("MODE DEBUG : 1 seul exemple.")

    # ── Reprise : ignorer les tabids déjà scorés ──────────────────────────────
    done_tabids = load_done_tabids(OUTPUT_PATH)
    if done_tabids and not args.debug:
        before = len(examples)
        examples = [e for e in examples if str(e["tabid"]) not in done_tabids]
        print(f"Reprise : {before - len(examples)} déjà scorés, "
              f"{len(examples)} restants.")

    if not examples:
        print("Rien à faire.")
        return

    # ── Charger le modèle ─────────────────────────────────────────────────────
    print("Chargement du modèle (peut prendre plusieurs minutes pour 70B)...")
    pipeline = transformers.pipeline(
        "text-generation",
        model       = MODEL_ID,
        model_kwargs= {"torch_dtype": torch.bfloat16},
        device_map  = "auto",
    )
    print("Modèle chargé.")

    # ── Traitement séquentiel avec sauvegarde au fil de l'eau ────────────────
    mode = "a" if (done_tabids and not args.debug) else "w"
    with open(OUTPUT_PATH, mode, encoding="utf-8") as fout:
        for ex in tqdm(examples, desc="Scoring"):
            user_prompt = build_prompt(ex)

            messages = [
                {"role": "system", "content": S2_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens = MAX_NEW_TOKENS,
                do_sample      = (TEMPERATURE > 0.0),
                temperature    = TEMPERATURE if TEMPERATURE > 0.0 else None,
                top_p          = TOP_P,
            )

            # outputs[0]["generated_text"] est la liste complète des messages
            # (system + user + assistant). On prend le dernier.
            generated_text = outputs[0]["generated_text"][-1]["content"].strip()

            row = {
                "tabid"           : ex["tabid"],
                "arxiv_id"        : ex.get("arxiv_id", ""),
                "table_text"      : ex["table_text"],
                "paper_text"      : ex.get("paper_text", ""),
                "candidate_goals" : ex["candidate_goals"],
                "judge_output"    : generated_text,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            if args.debug:
                print("\n" + "=" * 60)
                print("PREVIEW — sortie du juge")
                print("=" * 60)
                print(f"tabid: {ex['tabid']}")
                print(f"\nPrompt envoyé (premiers 500 caractères):")
                print(user_prompt[:500])
                print(f"\nRéponse du juge:")
                print(generated_text)

    print(f"\nTerminé. Résultats dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
