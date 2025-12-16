"""Feature transformation helpers for fate regression."""

from __future__ import annotations

import numpy as np


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stable logit transform that clips probabilities away from 0/1."""
    p_clip = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))
