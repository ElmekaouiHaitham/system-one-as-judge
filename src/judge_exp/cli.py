import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .common import ROOT, read_jsonl
from .data import prepare
from .experiment import bind_run, collect
from .providers import Judges
from .report import analyze
from .routing import route


def main():
    parser = argparse.ArgumentParser(description="Clef / Gemini / BeaverTails experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "main")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("collect")
    run.add_argument("--split", choices=["dev", "test", "both"], default="both")
    run.add_argument("--provider", choices=["clef", "gemini", "both"], default="both")
    run.add_argument("--limit", type=int, help="Smoke test: first N rows of each split")
    sub.add_parser("analyze")
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
    if args.command == "collect":
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        ok = collect(run, samples, config, policy,
                     ["dev", "test"] if args.split == "both" else [args.split],
                     ["clef", "gemini"] if args.provider == "both" else [args.provider], args.limit)
        if not ok:
            raise SystemExit(1)
    elif args.command == "analyze":
        analyze(run, samples, config)
    else:
        frozen = json.loads((run / "routing_policy.json").read_text(encoding="utf-8"))["policies"][args.policy]
        if args.output.exists():
            raise FileExistsError("Choose a new output file to avoid overwriting prior live calls")
        judges = Judges(config, policy)
        rows = read_jsonl(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for row in rows:
                c = judges.judge("clef", row)
                escalated = bool(route([c["label"]], [c["confidence"]], frozen["negative_threshold"], frozen["positive_threshold"])[0])
                g = judges.judge("gemini", row) if escalated else None
                f.write(json.dumps({"id": row.get("id"), "clef": c, "gemini": g,
                    "escalated": escalated, "label": g["label"] if g else c["label"]}) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
