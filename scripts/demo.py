"""Offline plumbing check with synthetic predictions; NEVER article results."""
import json
import numpy as np

from judge_exp.common import ROOT, digest, write_json, write_jsonl
from judge_exp.report import analyze


def main():
    run = ROOT / "runs" / "synthetic_demo"
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(123)
    samples = {}
    for split, count in [("dev", 100), ("test", 300)]:
        rows, c, g = [], [], []
        for i in range(count):
            y = int(i % 2)
            p = float(rng.beta(4, 2) if y else rng.beta(2, 4))
            item_id = digest(["synthetic", split, i])
            rows.append({"id": item_id, "label": y, "source_row": i,
                "control_type": "positive" if y else "safe_control"})
            c.append({"id": item_id, "status": "ok", "label": int(p >= .5),
                "confidence": max(p, 1-p), "probability": p, "latency_s": .2,
                "list_cost_usd": .00012})
            g.append({"id": item_id, "status": "ok", "label": y if rng.random() > .12 else 1-y,
                "latency_s": .8, "list_cost_usd": .00018})
        samples[split] = rows
        write_jsonl(run / f"{split}_clef.jsonl", c)
        write_jsonl(run / f"{split}_gemini.jsonl", g)
    write_json(run / "SYNTHETIC_ONLY.json", {"warning": "Fabricated test inputs. No model or dataset results."})
    analyze(run, samples, config)
    for path in (run / "report").glob("*.png"):
        # Watermark generated figures without modifying image files externally.
        import matplotlib.pyplot as plt
        image = plt.imread(path)
        fig, ax = plt.subplots(figsize=(image.shape[1]/140, image.shape[0]/140), dpi=140)
        ax.imshow(image); ax.axis("off")
        fig.text(.5, .99, "SYNTHETIC DEMO — NOT EXPERIMENT RESULTS", ha="center", va="top", color="crimson", fontsize=12)
        fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("SYNTHETIC ONLY: use this to verify plotting, never as article evidence.")


if __name__ == "__main__":
    main()
