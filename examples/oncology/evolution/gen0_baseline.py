"""Gen 0: Baseline — max-of-3 enrichment (current v2 approach)."""
import numpy as np

def score_genes(X: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(X[:, 6], X[:, 7]), X[:, 8])
