"""Gen 9v4: Unified scoring — same formula for oncology and non-oncology.

Hypothesis: maybe routing is premature. The original gen7 works ~70% on oncology
without any routing. What if we just add mutation_frequency as another "evidence gate"
alongside gwas_gate? A gene with EITHER gwas OR mutation evidence gets boosted.
This avoids the risk of the routing boundary degrading edge cases.
"""
import numpy as np

def score_genes(X: np.ndarray) -> np.ndarray:
    # Per-modality reliability gates
    g_esm2 = 1.0 / (1.0 + np.exp(-0.5 * (X[:, 1] - 5)))
    g_gf = 1.0 / (1.0 + np.exp(-0.5 * (X[:, 3] - 5)))
    g_pt = 1.0 / (1.0 + np.exp(-0.5 * (X[:, 5] - 5)))
    gated = np.stack([X[:, 6]*g_esm2, X[:, 7]*g_gf, X[:, 8]*g_pt], axis=1)

    e = np.maximum(gated, 0) + 1
    geo = np.power(e[:, 0] * e[:, 1] * e[:, 2], 1/3)
    max_g = gated.max(axis=1)

    n_sg = X[:, 19]
    scale = np.where(n_sg < 1000, 1.8, np.where(n_sg > 5000, 0.6, 1.0))
    conv = (0.5 * geo + 0.5 * np.log1p(max_g)) * scale

    # Evidence gates — GWAS and mutation are parallel evidence channels
    gwas_gate = 1.0 / (1.0 + np.exp(-10 * (X[:, 10] - 0.1)))
    mut_freq = X[:, 20]
    mut_gate = 1.0 / (1.0 + np.exp(-20 * (mut_freq - 0.05)))

    # Combined evidence: max of GWAS and mutation signal
    evidence = np.maximum(gwas_gate, mut_gate)

    score = conv * (1 + evidence * 3) + X[:, 16] * 0.9 + X[:, 15] * 0.4 + X[:, 13] * 1.0
    return score
