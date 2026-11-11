import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .common import write_json
from .experiment import load_predictions
from .routing import route, select_thresholds


def arrays(run, split, rows):
    all_results = {}
    for provider in ("clef", "gemini"):
        predictions = load_predictions(run, split, provider)
        missing = [r["id"] for r in rows if r["id"] not in predictions]
        if missing:
            raise ValueError(f"{split}/{provider}: {len(missing)} unattempted examples. Finish collect first.")
        all_results[provider] = [predictions[r["id"]] for r in rows]
    return all_results


def labels(results):
    return np.array([r["label"] if r["status"] == "ok" else -1 for r in results])


def metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred != 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": float((y == pred).mean()), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_positive_rate": fp / max(1, int((y == 0).sum())),
            "miss_rate_including_failures": fn / max(1, int((y == 1).sum())),
            "failure_rate": float((pred < 0).mean())}


def score_metrics(y, results):
    """AUROC on valid Clef scores only; publish the denominator explicitly."""
    y = np.asarray(y)
    probabilities = np.array([r.get("probability", np.nan) if r["status"] == "ok" else np.nan
                              for r in results], dtype=float)
    valid = np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)
    target, p = y[valid], probabilities[valid]
    errors = (target != (p >= .5)).astype(int)
    uncertainty = 1 - np.maximum(p, 1 - p)
    classification = float(roc_auc_score(target, p)) if len(np.unique(target)) == 2 else None
    error_detection = float(roc_auc_score(errors, uncertainty)) if len(np.unique(errors)) == 2 else None
    return {"classification_auroc": classification, "error_detection_auroc": error_detection,
            "auroc_scored_n": int(valid.sum()), "auroc_excluded_n": int((~valid).sum())}, (target, p, errors, uncertainty)


def analyze(run, samples, config):
    dev = arrays(run, "dev", samples["dev"])
    dc, dg = labels(dev["clef"]), labels(dev["gemini"])
    if (dc < 0).any() or (dg < 0).any():
        raise ValueError("Resolve development API failures before tuning thresholds.")
    dy = np.array([r["label"] for r in samples["dev"]])
    conf_dev = np.array([r["confidence"] for r in dev["clef"]])
    policies = {name: select_thresholds(dy, dc, dg, conf_dev, config["threshold_grid"],
                config["max_dev_accuracy_drop"], adaptive)
                for name, adaptive in [("fixed", False), ("class_dependent", True)]}
    write_json(run / "routing_policy.json", {"selection_split": "dev", "policies": policies})
    results = arrays(run, "test", samples["test"])
    y = np.array([r["label"] for r in samples["test"]])
    clef, gemini = labels(results["clef"]), labels(results["gemini"])
    confidence = np.array([r.get("confidence", 0) for r in results["clef"]])
    predictions = {"clef": clef, "gemini": gemini}
    masks = {"clef": np.zeros(len(y), dtype=bool), "gemini": np.ones(len(y), dtype=bool)}
    for name, p in policies.items():
        masks[name] = route(clef, confidence, p["negative_threshold"], p["positive_threshold"])
        predictions[name] = np.where(masks[name], gemini, clef)

    out = run / "report"
    out.mkdir(exist_ok=True)
    def values(provider, key):
        return np.array([r.get(key) if r.get(key) is not None else np.nan for r in results[provider]], dtype=float)
    rows = []
    clef_scores, roc_inputs = score_metrics(y, results["clef"])
    rng = np.random.default_rng(config["seed"])
    bootstrap = rng.integers(0, len(y), (2000, len(y)))
    for name, pred in predictions.items():
        mask = masks[name]
        cost = values("gemini", "list_cost_usd") if name == "gemini" else values("clef", "list_cost_usd").copy()
        latency = values("gemini", "latency_s") if name == "gemini" else values("clef", "latency_s").copy()
        if name in policies:
            cost += np.where(mask, values("gemini", "list_cost_usd"), 0)
            latency += np.where(mask, values("gemini", "latency_s"), 0)
        accuracy_samples = (y == pred)[bootstrap].mean(axis=1)
        difference = ((y == pred).astype(int) - (y == gemini).astype(int))[bootstrap].mean(axis=1)
        rows.append({"system": name, **metrics(y, pred),
                     **(clef_scores if name == "clef" else {
                         "classification_auroc": None, "error_detection_auroc": None,
                         "auroc_scored_n": 0, "auroc_excluded_n": len(y)}),
                     "accuracy_ci_low": float(np.quantile(accuracy_samples, .025)),
                     "accuracy_ci_high": float(np.quantile(accuracy_samples, .975)),
                     "accuracy_difference_vs_gemini_ci_low": float(np.quantile(difference, .025)),
                     "accuracy_difference_vs_gemini_ci_high": float(np.quantile(difference, .975)),
                     "escalation_rate": float(mask.mean()) if name != "gemini" else 1.0,
                     "list_cost_usd": float(cost.sum()) if np.isfinite(cost).all() else None,
                     "mean_latency_s": float(latency.mean()) if np.isfinite(latency).all() else None})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "metrics.csv", index=False)
    write_json(out / "metrics.json", rows)
    details = pd.DataFrame([{k: r[k] for k in ("id", "label", "control_type", "source_row")} for r in samples["test"]])
    for name, pred in predictions.items():
        details[name] = pred
    details["clef_confidence"] = confidence
    details["clef_probability"] = values("clef", "probability")
    details.to_csv(out / "predictions.csv", index=False)
    slices = []
    for group in details.control_type.unique():
        selected = (details.control_type == group).to_numpy()
        for name, pred in predictions.items():
            slices.append({"group": group, "system": name, "n": int(selected.sum()), **metrics(y[selected], pred[selected])})
    pd.DataFrame(slices).to_csv(out / "control_slices.csv", index=False)
    diagnostics = {}
    for name in policies:
        mask = masks[name]
        diagnostics[name] = {"escalated_count": int(mask.sum()),
            "fallback_corrections": int((mask & (clef != y) & (gemini == y)).sum()),
            "fallback_regressions": int((mask & (clef == y) & (gemini != y)).sum()),
            "shared_errors": int(((clef != y) & (gemini != y)).sum())}
    write_json(out / "cascade_diagnostics.json", diagnostics)

    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    target, p, errors, uncertainty = roc_inputs
    for ax, truth, scores, title, key in [
        (axes[0], target, p, "Clef category classification", "classification_auroc"),
        (axes[1], errors, uncertainty, "Clef error detection using uncertainty", "error_detection_auroc"),
    ]:
        auc = clef_scores[key]
        ax.plot([0, 1], [0, 1], "--", color="gray")
        if auc is not None:
            fpr, tpr, _ = roc_curve(truth, scores)
            ax.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
            ax.legend()
        else:
            ax.text(.5, .5, "Undefined: both outcome classes required", ha="center", fontsize=9)
        ax.set(title=title, xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(f"Valid Clef scores: {clef_scores['auroc_scored_n']}; excluded: {clef_scores['auroc_excluded_n']}")
    fig.savefig(out / "clef_roc.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)
    for ax, (name, pred) in zip(axes, predictions.items()):
        matrix = np.array([[int(((y == a) & (pred == b)).sum()) for b in (0, 1, -1)] for a in (0, 1)])
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(matrix):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set(xticks=[0, 1, 2], xticklabels=["Negative", "Positive", "Failure"],
               yticks=[0, 1], yticklabels=["Negative", "Positive"], title=name,
               xlabel="Predicted", ylabel="Human label")
    fig.savefig(out / "confusion_matrices.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, col, title in zip(axes, ["list_cost_usd", "mean_latency_s"],
                              ["Estimated list cost (USD, test sample)", "Mean request latency (seconds)"]):
        ax.bar(summary.system, pd.to_numeric(summary[col]), color=["#277da8", "#f2a541", "#43aa8b", "#9467bd"])
        ax.set_title(title); ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Cascade totals are simulated from cached calls; no free-tier discount")
    fig.savefig(out / "cost_latency.png"); plt.close(fig)

    curve = []
    for t in config["threshold_grid"]:
        mask = route(clef, confidence, t, t)
        accepted = ~mask
        curve.append({"threshold": t, "clef_coverage": float(accepted.mean()),
                      "cascade_accuracy": float((np.where(mask, gemini, clef) == y).mean()),
                      "accepted_error_rate": float((clef[accepted] != y[accepted]).mean()) if accepted.any() else None})
    pd.DataFrame(curve).to_csv(out / "threshold_sweep_diagnostic.csv", index=False)
    curve_df = pd.DataFrame(curve)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(curve_df.clef_coverage, curve_df.cascade_accuracy, "o-", label="Fixed-threshold sweep")
    axes[0].axhline((gemini == y).mean(), color="#f2a541", linestyle="--", label="Gemini alone")
    for name in policies:
        axes[0].scatter(1 - masks[name].mean(), (predictions[name] == y).mean(), s=100, marker="*", label=f"{name}: frozen on dev")
    axes[0].set(xlabel="Fraction handled by Clef", ylabel="Cascade accuracy")
    axes[0].legend(fontsize=8)
    axes[1].plot(curve_df.clef_coverage, curve_df.accepted_error_rate, "o-")
    axes[1].set(xlabel="Fraction handled by Clef", ylabel="Error rate among accepted Clef verdicts")
    fig.suptitle("Test curves describe trade-offs; do not select thresholds from them")
    fig.savefig(out / "coverage_tradeoff.png"); plt.close(fig)
    print(summary.to_string(index=False))
    print(f"Saved report to {out}")
