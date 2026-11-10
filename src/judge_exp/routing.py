import itertools

import numpy as np


def route(clef_labels, confidence, negative_threshold, positive_threshold):
    labels = np.asarray(clef_labels)
    thresholds = np.where(labels == 1, positive_threshold, negative_threshold)
    return (labels < 0) | (np.asarray(confidence) < thresholds)


def select_thresholds(y, clef, gemini, confidence, grid, max_drop, adaptive=False):
    """Select on development labels only; minimize escalations subject to accuracy."""
    y, clef, gemini = map(np.asarray, (y, clef, gemini))
    target = float(np.mean(y == gemini)) - max_drop
    candidates = itertools.product(grid, repeat=2) if adaptive else ((t, t) for t in grid)
    options = []
    for negative, positive in candidates:
        mask = route(clef, confidence, negative, positive)
        accuracy = float(np.mean(y == np.where(mask, gemini, clef)))
        if accuracy + 1e-12 >= target:
            options.append((float(mask.mean()), -accuracy, negative, positive))
    if not options:
        raise ValueError("No feasible threshold; include a threshold > 1 to allow all-Gemini routing")
    escalation, neg_accuracy, negative, positive = min(options)
    return {"negative_threshold": negative, "positive_threshold": positive,
            "dev_accuracy": -neg_accuracy, "dev_escalation_rate": escalation,
            "dev_target_accuracy": target}
