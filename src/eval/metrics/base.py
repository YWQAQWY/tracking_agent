"""Small shared arithmetic with explicit missing-data semantics."""

import math
import statistics


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def mean(values) -> float | None:
    observed = [value for value in values if value is not None]
    return statistics.mean(observed) if observed else None


def p95(values: list[float]) -> float | None:
    """Nearest-rank percentile: sorted[ceil(0.95 * n) - 1]."""
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None


def classification(expected: list[bool], predicted: list[bool]) -> dict:
    pairs = list(zip(expected, predicted, strict=True))
    tp = sum(label and pred for label, pred in pairs)
    tn = sum(not label and not pred for label, pred in pairs)
    fp = sum(not label and pred for label, pred in pairs)
    fn = sum(label and not pred for label, pred in pairs)
    return {
        "labelled_count": len(pairs), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": ratio(tp + tn, len(pairs)), "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
    }
