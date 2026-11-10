import gzip
import json
import random
from collections import Counter

import requests

from .common import CATEGORY, ROOT, digest, write_json, write_jsonl


def prompt_key(text):
    return " ".join(text.casefold().split())


def fetch(split, revision):
    path = ROOT / "data" / "raw" / f"{revision}-{split}.jsonl.gz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/PKU-Alignment/BeaverTails/resolve/{revision}/round0/30k/{split}.jsonl.gz"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sample(rows, size, seed, split, blocked_prompts=()):
    if size < 4 or size % 4:
        raise ValueError("Sample sizes must be positive multiples of 4.")
    rng = random.Random(seed)
    rows = list(enumerate(rows))
    rng.shuffle(rows)
    seen = set(blocked_prompts)
    pools = {"positive": [], "safe_control": [], "other_harm_control": []}
    for index, row in rows:
        key = prompt_key(row["prompt"])
        if key in seen:
            continue
        seen.add(key)
        label = bool(row["category"][CATEGORY])
        group = "positive" if label else ("safe_control" if row["is_safe"] else "other_harm_control")
        pools[group].append({
            "id": digest([row["prompt"], row["response"]]),
            "source_split": f"30k_{split}", "source_row": index,
            "prompt": row["prompt"], "response": row["response"],
            "label": int(label), "control_type": group,
            "is_safe": bool(row["is_safe"]), "categories": row["category"],
        })
    selected = []
    for group, count in [("positive", size // 2), ("safe_control", size // 4), ("other_harm_control", size // 4)]:
        if len(pools[group]) < count:
            raise ValueError(f"Only {len(pools[group])} unique prompts in {group}; need {count}. Reduce size.")
        selected.extend(pools[group][:count])
    rng.shuffle(selected)
    return selected


def prepare(config):
    out = ROOT / "data" / "prepared"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Prepared data already exists. Use a new project copy for a different sample.")
    test_raw = fetch("test", config["dataset_revision"])
    train_raw = fetch("train", config["dataset_revision"])
    test = sample(test_raw, config["test_size"], config["seed"], "test")
    # Exclude ALL test prompts from development, not just the selected test sample.
    dev = sample(train_raw, config["dev_size"], config["seed"] + 1, "train",
                 {prompt_key(r["prompt"]) for r in test_raw})
    for name, rows in [("dev", dev), ("test", test)]:
        write_jsonl(out / f"{name}.jsonl", rows)
    write_json(out / "manifest.json", {
        "dataset": "PKU-Alignment/BeaverTails", "revision": config["dataset_revision"],
        "category": CATEGORY, "seed": config["seed"],
        "sampling": "one response per normalized prompt; 50% positive, 25% safe, 25% other harm",
        "splits": {name: {"count": len(rows), "sha256": digest(rows),
                          "groups": dict(Counter(r["control_type"] for r in rows))}
                   for name, rows in [("dev", dev), ("test", test)]},
    })
    print(f"Prepared {len(dev)} development and {len(test)} test examples.")
