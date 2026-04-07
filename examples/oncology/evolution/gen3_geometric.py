"""Gen 3b: Geometric mean of enrichments — naturally rewards multi-modality convergence."""
import numpy as np

def score_genes(X: np.ndarray) -> np.ndarray:
    # Add 1 to enrichments (so 0 enrichment = 1, baseline = 1)
    e = np.maximum(np.stack([X[:, 6], X[:, 7], X[:, 8]], axis=1), 0) + 1

    # Geometric mean = (e1 * e2 * e3)^(1/3)
    # This naturally rewards genes that are high in MULTIPLE modalities
    geo_mean = np.power(e[:, 0] * e[:, 1] * e[:, 2], 1/3)

    # But also keep the max for single-modality stars
    max_e = np.maximum(np.maximum(X[:, 6], X[:, 7]), X[:, 8])

    convergence = 0.5 * geo_mean + 0.5 * np.log1p(max_e)

    # Evidence + constraint
    evidence = X[:, 10] * 2.0 + X[:, 16] * 0.6 + X[:, 15] * 0.3

    # Reliability
    reliability = np.clip(X[:, 18] / 10.0, 0.1, 1)

    return (convergence + evidence) * reliability
