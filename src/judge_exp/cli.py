import argparse
import json
import time
import math
from pathlib import Path

from dotenv import load_dotenv

from .common import ROOT, read_jsonl
from .data import prepare
from .experiment import bind_run, collect, reuse_clef
from .providers import Judges, ProviderError
from .accounting import summarize_resources
from .report import analyze
from .routing import route


def main():
    parser = argparse.ArgumentParser(description="Clef / LLM / BeaverTails experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "main")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("collect")
    run.add_argument("--split", choices=["dev", "test", "both"], default="both")
    run.add_argument("--provider", choices=["clef", "gemini", "fallback", "both"], default="both")
    run.add_argument("--limit", type=int, help="Smoke test: first N rows of each split")
    run.add_argument("--requests-per-minute", type=float,
                     help="Bedrock transport ceiling; can lower but never raise the config limit")
    full = sub.add_parser("full", help="Resume collection, freeze thresholds on dev, collect test and analyze")
    full.add_argument("--requests-per-minute", type=float, default=2)
    full.add_argument("--max-restarts", type=int, default=6, help="Bounded resumes for transient failures per split")
    sub.add_parser("analyze")
    sub.add_parser("compare", help="Primary standalone model comparison using all collected examples")
    reuse = sub.add_parser("reuse-clef", help="Import compatible cached Clef calls with provenance")
    reuse.add_argument("--source", type=Path, required=True)
    cascade = sub.add_parser("cascade", help="Live selective calls using the frozen policy")
    cascade.add_argument("--input", type=Path, required=True, help="JSONL with prompt and response")
    cascade.add_argument("--output", type=Path, required=True)
    cascade.add_argument("--policy", choices=["fixed", "class_dependent"], default="fixed")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    policy = (ROOT / "prompts" / "policy.txt").read_text(encoding="utf-8")
    if args.command == "prepare":
        prepare(config)
        return
    run, samples = bind_run(args.run, config, policy)
    if args.command == "full":
        if not math.isfinite(args.requests_per_minute) or args.requests_per_minute <= 0 or args.max_restarts < 0:
            parser.error("Use a finite positive request rate and nonnegative max-restarts")
        from .workflow import full_experiment
        full_experiment(run, samples, config, policy, args.requests_per_minute, args.max_restarts)
    elif args.command == "collect":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.requests_per_minute is not None and (not math.isfinite(args.requests_per_minute) or args.requests_per_minute <= 0):
            parser.error("--requests-per-minute must be finite and positive")
        ok = collect(run, samples, config, policy,
                     ["dev", "test"] if args.split == "both" else [args.split],
                     ["clef", "gemini"] if args.provider == "both" else ["gemini" if args.provider == "fallback" else args.provider], args.limit,
                     requests_per_minute=args.requests_per_minute)
        if not ok:
            raise SystemExit(1)
    elif args.command == "analyze":
        analyze(run, samples, config)
    elif args.command == "compare":
        from .comparison import compare_models
        result = compare_models(run, samples, config)
        print(json.dumps(result, indent=2))
    elif args.command == "reuse-clef":
        reuse_clef(run, args.source, config, policy)
        print("Reused compatible Clef outcomes; no model calls made.")
    else:
        frozen = json.loads((run / "routing_policy.json").read_text(encoding="utf-8"))["policies"][args.policy]
        if args.output.exists():
            raise FileExistsError("Choose a new output file to avoid overwriting prior live calls")
        judges = Judges(config, policy)
        rows = read_jsonl(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for row in rows:
                started = time.perf_counter()
                c, g, escalated, stage = None, None, False, "clef"
                try:
                    c = judges.judge("clef", row)
                    escalated = bool(route([c["label"]], [c["confidence"]], frozen["negative_threshold"], frozen["positive_threshold"])[0])
                    stage = "gemini"
                    g = judges.judge("gemini", row) if escalated else None
                    status, error = "ok", None
                except ProviderError as exc:
                    if stage == "clef":
                        c = exc.record
                    else:
                        g = exc.record
                    status, error = "error", str(exc)
                wall_latency = time.perf_counter() - started
                resources = summarize_resources([c], [g or {}], [escalated])
                f.write(json.dumps({"id": row.get("id"), "clef": c, "gemini": g,
                    "status": status, "error": error, "escalated": escalated,
                    "label": (g["label"] if g else c["label"]) if status == "ok" else None,
                    "wall_latency_s": wall_latency, "list_cost_usd": resources["list_cost_usd"],
                    "request_path_latency_s": resources["mean_latency_s"]}) + "\n")
                f.flush()
                if status == "error":
                    raise SystemExit(f"Live routing stopped; failure saved: {error}")


if __name__ == "__main__":
    main()
