"""Descriptive evidence, never causal or exact model attribution."""

from typing import Any
import numpy as np


def evidence(deviations: Any, sensors: list[str]) -> list[str]:
    return [
        f"{sensors[int(i)]}: absolute robust deviation={row[int(i)]:.3f}; "
        "descriptive evidence, not causality or model attribution"
        for row, i in zip(deviations, np.argmax(deviations, axis=1), strict=True)
    ]
