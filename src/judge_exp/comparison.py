"""Primary fixed-judge comparison across every collected example."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .common import write_json
from .report import arrays, labels, metrics, score_metrics, paired_first_attempt_latency
from .accounting import summarize_resources


def compare_models(run, samples, config):
    rows, calls = [], {"clef": [], "gemini": []}
    for split in ("dev", "test"):
        results = arrays(run, split, samples[split])
        rows.extend(samples[split])
        for provider in calls:
            calls[provider].extend(results[provider])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Standalone comparison requires distinct examples across both groups")
    y = np.array([r["label"] for r in rows])
    predictions = {p: labels(r) for p, r in calls.items()}
    names = {"clef": "Clef", "gemini": config.get("fallback_display_name", "Gemini")}
    rng = np.random.default_rng(config["seed"])
    bootstrap = rng.integers(0, len(y), (2000, len(y)))
    summary = []
    for provider, pred in predictions.items():
        correct = (pred == y)
        accuracy = correct[bootstrap].mean(axis=1)
        delta = (correct.astype(int) - (predictions["gemini"] == y).astype(int))[bootstrap].mean(axis=1)
        scores, _ = score_metrics(y, calls[provider])
        summary.append({"system": names[provider], "n": len(y), "correct": int(correct.sum()),
            **metrics(y, pred), **scores,
            "accuracy_ci_low": float(np.quantile(accuracy, .025)),
            "accuracy_ci_high": float(np.quantile(accuracy, .975)),
            "accuracy_difference_vs_fallback_ci_low": float(np.quantile(delta, .025)),
            "accuracy_difference_vs_fallback_ci_high": float(np.quantile(delta, .975)),
            **summarize_resources(calls["clef"], calls["gemini"], gemini_only=provider == "gemini")})
    out = run / "report"
    out.mkdir(exist_ok=True)
    write_json(out / "standalone_metrics.json", summary)
    pd.DataFrame(summary).to_csv(out / "standalone_metrics.csv", index=False)
    write_json(out / "standalone_design.json", {
        "primary_question": "Compare the fixed standalone judges against existing human labels",
        "evaluation_n": len(y), "group_counts": {s: len(r) for s, r in samples.items()},
        "positive_n": int(y.sum()), "negative_n": int((y == 0).sum()),
        "model_training_or_finetuning": False,
        "basis": "All collected examples, one frozen rubric and model configuration; cached calls reused",
        "cascade_analysis": "Separate secondary analysis: calibrate router on dev, evaluate it on test. Do not pool in-sample router decisions into its headline score."})
    write_json(out / "standalone_paired_latency.json",
               paired_first_attempt_latency(calls["clef"], calls["gemini"], names["gemini"]))
    pd.DataFrame([{"id": r["id"], "human_label": r["label"],
        "clef": int(predictions["clef"][i]), "fallback": int(predictions["gemini"][i])}
        for i, r in enumerate(rows)]).to_csv(out / "standalone_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    x = np.arange(2)
    acc = np.array([r["accuracy"] for r in summary])
    axes[0].bar(x, acc, color=["#277da8", "#f2a541"], yerr=[
        acc - [r["accuracy_ci_low"] for r in summary],
        np.array([r["accuracy_ci_high"] for r in summary]) - acc], capsize=5)
    axes[0].set(xticks=x, xticklabels=list(names.values()), ylim=(0, 1),
                title="Accuracy with 95% bootstrap intervals", ylabel="Score")
    for i, r in enumerate(summary):
        axes[0].text(i, .04, f"{r['correct']}/{r['n']}\n{r['accuracy']:.1%}", ha="center", color="white")
    for shift, key in [(-.25, "precision"), (0, "recall"), (.25, "f1")]:
        axes[1].bar(x + shift, [r[key] for r in summary], width=.25, label=key.title())
    axes[1].set(xticks=x, xticklabels=list(names.values()), ylim=(0, 1), title="Category-positive performance")
    axes[1].legend()
    fig.suptitle(f"Primary standalone comparison: all {len(y)} examples")
    fig.savefig(out / "standalone_comparison.png", dpi=160)
    plt.close(fig)
    return summary
