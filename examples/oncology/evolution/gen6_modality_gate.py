"""Gen 6c: Per-modality sigmoid gates — only trust modalities with enough neighbors."""
import numpy as np
def score_genes(X: np.ndarray) -> np.ndarray:
    # Gate each modality by its neighbor count (reliable only with enough neighbors)
    gate_esm2 = 1.0 / (1.0 + np.exp(-0.5 * (X[:, 1] - 5)))
    gate_gf = 1.0 / (1.0 + np.exp(-0.5 * (X[:, 3] - 5)))
    gate_pt = 1.0 / (1.0 + np.exp(-0.5 * (X[:, 5] - 5)))
    # Gated enrichments
    gated_esm2 = X[:, 6] * gate_esm2
    gated_gf = X[:, 7] * gate_gf
    gated_pt = X[:, 8] * gate_pt
    max_gated = np.maximum(np.maximum(gated_esm2, gated_gf), gated_pt)
    sum_gated = (gated_esm2 + gated_gf + gated_pt) / 3
    conv = np.log1p(0.5 * max_gated + 0.5 * sum_gated)
    gwas_gate = 1.0 / (1.0 + np.exp(-10 * (X[:, 10] - 0.1)))
    score = conv * (1 + gwas_gate * 3) + X[:, 16] * 0.8 + X[:, 15] * 0.4
    n_sg = X[:, 19]
    onc = (X[:, 13] + X[:, 14]) * np.where(n_sg > 5000, 1.0, 0.1)
    return score + onc
