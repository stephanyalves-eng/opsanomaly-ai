"""CPU-only Isolation Forest comparator."""

from typing import Any
from sklearn.ensemble import IsolationForest


def fit(x: Any, seed: int = 42) -> IsolationForest:
    return IsolationForest(n_estimators=200, random_state=seed, n_jobs=1).fit(x)


def score(model: IsolationForest, x: Any) -> Any:
    return -model.score_samples(x)
