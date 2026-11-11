import json
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, digest, read_jsonl, write_json
from .providers import Judges, ProviderError


def bind_run(run, config, policy):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    samples = {s: read_jsonl(ROOT / "data" / "prepared" / f"{s}.jsonl") for s in ["dev", "test"]}
    spec = {"config": config, "policy": policy, "measurement_version": 2,
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
    latest = {}
    if not path.exists():
        return latest
    for record in read_jsonl(path):
        previous = latest.get(record["id"])
        if previous and previous["status"] == "error" and not terminal_failure(previous):
            # Retain immutable logs, but aggregate transport-retry resources for
            # an outcome completed after restarting the collector.
            current = record
            record = dict(current)
            for key in ("latency_s", "request_latency_s", "retry_wait_s", "list_cost_usd", "cache_adjusted_cost_usd"):
                parts = [previous.get(key), record.get(key)]
                record[key] = sum(parts) if all(p is not None for p in parts) else None
            for key in ("list_cost_usd", "cache_adjusted_cost_usd"):
                record[f"known_{key}"] = (
                    previous.get(f"known_{key}", previous.get(key)) or 0) + (
                    current.get(f"known_{key}", current.get(key)) or 0)
            record["attempts"] = previous.get("attempts", []) + record.get("attempts", [])
            record["retry_count"] = previous.get("retry_count", 0) + record.get("retry_count", 0) + 1
            record["collector_restarts"] = previous.get("collector_restarts", 0) + 1
            record["latency_basis"] = "sum_of_call_durations; collector_restart_gaps_excluded"
        latest[record["id"]] = record
    return latest


def terminal_failure(record):
    # Recognize previously saved Gemini content blocks without rewriting the log.
    return bool(record.get("terminal_error") or
                record.get("raw", {}).get("promptFeedback", {}).get("blockReason"))


def collect(run, samples, config, policy, splits, providers, limit=None):
    judges = Judges(config, policy)
    for split in splits:
        for provider in providers:
            existing = load_predictions(run, split, provider)
            rows = samples[split][:limit] if limit else samples[split]
            for i, row in enumerate(rows, 1):
                previous = existing.get(row["id"], {})
                if previous.get("status") == "ok" or terminal_failure(previous):
                    continue
                result = {"id": row["id"], "provider": provider,
                          "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                try:
                    result.update(judges.judge(provider, row))
                    result["status"] = "ok"
                except ProviderError as e:
                    result.update(e.record)
                    result.update(status="error", error=str(e))
                except (KeyError, ValueError, RuntimeError) as e:
                    result.update(status="error", error=str(e))
                except Exception as e:
                    # Network errors can embed request URLs; log only the exception type.
                    result.update(status="error", error=type(e).__name__)
                with (Path(run) / f"{split}_{provider}.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    f.flush()
                print(f"{split} {provider} {i}/{len(rows)}: {result['status']}", flush=True)
                if result["status"] == "error" and not terminal_failure(result):
                    print(f"Stopped after error: {result['error']}. Fix it and rerun to resume.")
                    return False
    return True
