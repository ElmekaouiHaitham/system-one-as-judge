import json
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, digest, read_jsonl, write_json
from .providers import Judges


def bind_run(run, config, policy):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    samples = {s: read_jsonl(ROOT / "data" / "prepared" / f"{s}.jsonl") for s in ["dev", "test"]}
    spec = {"config": config, "policy": policy,
            "sample_hashes": {s: digest(rows) for s, rows in samples.items()}}
    signature = digest(spec)
    path = run / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("Config, prompt, or sample changed. Choose a new --run directory.")
    else:
        write_json(path, {**spec, "signature": signature,
                          "created_utc": datetime.now(timezone.utc).isoformat()})
    return run, samples


def load_predictions(run, split, provider):
    path = Path(run) / f"{split}_{provider}.jsonl"
    return {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}


def collect(run, samples, config, policy, splits, providers, limit=None):
    judges = Judges(config, policy)
    for split in splits:
        for provider in providers:
            existing = load_predictions(run, split, provider)
            rows = samples[split][:limit] if limit else samples[split]
            for i, row in enumerate(rows, 1):
                if existing.get(row["id"], {}).get("status") == "ok":
                    continue
                result = {"id": row["id"], "provider": provider,
                          "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                try:
                    result.update(judges.judge(provider, row))
                    result["status"] = "ok"
                except (KeyError, ValueError, RuntimeError) as e:
                    result.update(status="error", error=str(e))
                except Exception as e:
                    # Network errors can embed request URLs; log only the exception type.
                    result.update(status="error", error=type(e).__name__)
                with (Path(run) / f"{split}_{provider}.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    f.flush()
                print(f"{split} {provider} {i}/{len(rows)}: {result['status']}", flush=True)
                if result["status"] == "error":
                    print(f"Stopped after error: {result['error']}. Fix it and rerun to resume.")
                    return False
    return True
