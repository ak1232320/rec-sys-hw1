"""MAP@K exactly as Kaggle scores this competition (the ml_metrics definition)."""
import numpy as np


def apk(actual, predicted, k=5):
    """Average precision at k for one user; `actual` is the set that counts as relevant."""
    if not actual:
        return 0.0
    predicted = list(predicted)[:k]
    actual = set(actual)
    hits, score = 0, 0.0
    for i, item in enumerate(predicted):
        if item in actual and item not in predicted[:i]:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(truth, predictions, k=5, users=None):
    """Mean of apk over `users` (default: every user that has a truth list)."""
    users = list(truth) if users is None else list(users)
    if not users:
        return float("nan")
    return float(np.mean([apk(truth[u], predictions.get(u, []), k) for u in users]))


def hit_rate(truth, predictions, k=5, users=None):
    """Share of truth items that appear anywhere in the top-k (order ignored)."""
    users = list(truth) if users is None else list(users)
    found = total = 0
    for u in users:
        actual = set(truth[u])
        found += len(actual & set(list(predictions.get(u, []))[:k]))
        total += min(len(actual), k)
    return found / total if total else float("nan")


def _self_check():
    assert apk([1, 2, 3], [1, 2, 3]) == 1.0
    assert apk([1, 2, 3], [4, 5, 6]) == 0.0
    # Kaggle reference examples (ml_metrics test suite).
    assert abs(apk(range(1, 6), [6, 4, 7, 1, 2], 2) - 0.25) < 1e-12
    assert abs(apk(range(1, 6), [1, 1, 1, 1, 1], 5) - 0.2) < 1e-12
    assert abs(apk(range(1, 101), list(range(1, 21)) + list(range(200, 600)), 20) - 1.0) < 1e-12
    assert abs(apk([1, 3], [1, 2, 3, 4, 5], 3) - (1 + 2 / 3) / 2) < 1e-12
    assert apk([], [1, 2]) == 0.0
    assert abs(mapk({"a": [1], "b": [2]}, {"a": [1], "b": [3]}) - 0.5) < 1e-12
    assert hit_rate({"a": [1, 2]}, {"a": [2, 9]}) == 0.5
    print("metrics.py self-check: ok")


if __name__ == "__main__":
    _self_check()
