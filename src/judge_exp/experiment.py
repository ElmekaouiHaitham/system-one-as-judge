import json
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, digest, read_jsonl, write_json
from .providers import Judges, ProviderError
from .rate_limit import RateLimiter


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
            if "provider_attempt_latency_sum_s" in record or "provider_attempt_latency_sum_s" in previous:
                parts = [previous.get("provider_attempt_latency_sum_s"), current.get("provider_attempt_latency_sum_s")]
                record["provider_attempt_latency_sum_s"] = sum(parts) if all(v is not None for v in parts) else None
        latest[record["id"]] = record
    return latest


def terminal_failure(record):
    # Recognize previously saved Gemini content blocks without rewriting the log.
    return bool(record.get("terminal_error") or
                record.get("raw", {}).get("promptFeedback", {}).get("blockReason"))


def reuse_clef(run, source, config, policy):
    """Reuse immutable Clef calls only after verifying compatible inputs/settings."""
    run, source = Path(run), Path(source)
    target = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if original["policy"] != policy or original["sample_hashes"] != target["sample_hashes"]:
        raise ValueError("Clef source has a different policy or dataset sample")
    if original["measurement_version"] != target["measurement_version"]:
        raise ValueError("Clef source has a different measurement version")
    for key in ("clef_model", "clef_request_model", "timeout_seconds", "max_attempts"):
        if original["config"][key] != config[key]:
            raise ValueError(f"Clef source has different {key}")
    if original["config"]["prices_usd_per_million"]["clef_input"] != config["prices_usd_per_million"]["clef_input"]:
        raise ValueError("Clef source has different pricing")
    imports = {}
    for split in ("dev", "test"):
        rows = read_jsonl(ROOT / "data" / "prepared" / f"{split}.jsonl")
        predictions = load_predictions(source, split, "clef")
        if set(predictions) != {r["id"] for r in rows} or any(r["status"] != "ok" for r in predictions.values()):
            raise ValueError(f"Clef source must have successful outcomes for every {split} row")
        content = (source / f"{split}_clef.jsonl").read_bytes()
        destination = run / f"{split}_clef.jsonl"
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError("Target already contains different Clef records")
        imports[split] = content
    for split, content in imports.items():
        (run / f"{split}_clef.jsonl").write_bytes(content)
    write_json(run / "reused_clef_provenance.json", {
        "source_run": str(source.resolve()), "source_signature": original["signature"],
        "source_record_hashes": {s: digest(read_jsonl(source / f"{s}_clef.jsonl")) for s in imports},
        "note": "No new Clef calls. Quality and per-outcome resource estimates reuse earlier calls; timings are not contemporaneous.",
    })


def collect(run, samples, config, policy, splits, providers, limit=None, requests_per_minute=None):
    judges = Judges(config, policy)
    if config.get("fallback_backend") == "bedrock":
        rpm = min(requests_per_minute or config["fallback_requests_per_minute"], config["fallback_requests_per_minute"])
        judges.bedrock_limiter = RateLimiter(rpm, Path(run) / "bedrock_rate_limit.json")
        # Transport changes don't alter the frozen model, rubric or sample.
        # Preserve the immutable original manifest and record each collection session.
        with (Path(run) / "transport_sessions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(),
                "max_requests_per_minute": rpm, "strategy": "persistent_adaptive_v1",
                "splits": splits, "providers": providers}) + "\n")
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
                name = config.get("fallback_display_name", provider) if provider == "gemini" else provider
                print(f"{split} {name} {i}/{len(rows)}: {result['status']}", flush=True)
                if result["status"] == "error" and not terminal_failure(result):
                    print(f"Stopped after error: {result['error']}. Fix it and rerun to resume.")
                    return False
    return True
