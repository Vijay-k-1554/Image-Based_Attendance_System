# utils.py

import numpy as np

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a numpy vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
