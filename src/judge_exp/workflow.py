"""Sequential collection, dev-only threshold freezing, then held-out analysis."""
from datetime import datetime, timezone
import json

from .common import write_json
from .experiment import collect, load_predictions, terminal_failure
from .report import freeze_policies, analyze


def full_experiment(run, samples, config, policy, requests_per_minute=2, max_restarts=6):
    for split in ("dev", "test"):
        write_json(run / "workflow_status.json", {"status": "collecting", "split": split,
                   "updated_utc": datetime.now(timezone.utc).isoformat(),
                   "requests_per_minute_ceiling": requests_per_minute})
        for restart in range(max_restarts + 1):
            if collect(run, samples, config, policy, [split], ["clef", "gemini"],
                       requests_per_minute=requests_per_minute):
                break
            failures = [r for provider in ("clef", "gemini")
                        for r in load_predictions(run, split, provider).values()
                        if r["status"] == "error" and not terminal_failure(r)]
            def transient(record):
                attempts = record.get("attempts", [])
                code = attempts[-1].get("http_status") if attempts else -1
                return code is None or code == 429 or code >= 500
            if restart == max_restarts or not failures or not all(transient(r) for r in failures):
                write_json(run / "workflow_status.json", {"status": "stopped_on_error", "split": split,
                           "updated_utc": datetime.now(timezone.utc).isoformat()})
                raise RuntimeError("Collection stopped; outcomes and rate-limit state are saved")
            print("Transient failure retained; resuming with persistent rate-limit cooldown.", flush=True)
        if split == "dev":
            frozen = freeze_policies(run, samples, config)
            freeze_path = run / "routing_policy_before_test.json"
            if freeze_path.exists():
                previous = json.loads(freeze_path.read_text(encoding="utf-8"))
                if {k: previous[k] for k in ("selection_split", "policies")} != frozen:
                    raise ValueError("Previously frozen policies differ; choose a new run")
            else:
                if load_predictions(run, "test", "gemini"):
                    raise ValueError("Test collection already started without a recorded prior freeze; inspect the run")
                write_json(freeze_path, {**frozen,
                           "frozen_utc": datetime.now(timezone.utc).isoformat()})
            print("Cascade thresholds frozen from development data before test collection.", flush=True)
    analyze(run, samples, config)
    write_json(run / "workflow_status.json", {"status": "complete",
               "updated_utc": datetime.now(timezone.utc).isoformat()})
