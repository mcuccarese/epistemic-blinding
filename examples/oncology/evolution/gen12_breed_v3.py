"""Gen 12 breed v3: Hierarchical scoring.

Insight: the two approaches capture different *types* of drug targets:
- Evidence-first captures targets that ARE the mutated/GWAS gene (BRAF, EGFR, KRAS)
- Convergence captures targets in the PATHWAY of mutated genes (MEK, RAF1)

The hierarchical approach: score by evidence first, then within evidence tiers
use convergence to separate.

Tier 1: Direct evidence gene (is_mutated OR is_gwas) — start high
Tier 2: Convergent with evidence genes — medium
Tier 3: Neither — low

Within each tier, convergence and constraint break ties.
"""
import numpy as np

def score_genes(X: np.ndarray) -> np.ndarray:
    is_onc = X[:, 21] > 0.5

    # Per-modality
    g_esm2 = 1.0 / (1.0 + np.exp(-0.5 * (X[:, 1] - 5)))
    g_gf = 1.0 / (1.0 + np.exp(-0.5 * (X[:, 3] - 5)))
    g_pt = 1.0 / (1.0 + np.exp(-0.5 * (X[:, 5] - 5)))
    gated = np.stack([X[:, 6]*g_esm2, X[:, 7]*g_gf, X[:, 8]*g_pt], axis=1)

    e = np.maximum(gated, 0) + 1
    geo = np.power(e[:, 0] * e[:, 1] * e[:, 2], 1/3)
    max_g = gated.max(axis=1)
    conv = 0.5 * geo + 0.5 * np.log1p(max_g)

    n_sg = X[:, 19]
    pli = X[:, 16]
    mut_freq = X[:, 20]

    # ── Non-oncology: hierarchical GWAS ──
    gwas_gate = 1.0 / (1.0 + np.exp(-10 * (X[:, 10] - 0.1)))
    is_gwas = X[:, 9]
    scale = np.where(n_sg < 1000, 1.8, np.where(n_sg > 5000, 0.6, 1.0))

    # Tier assignment (soft)
    tier1_non = is_gwas * gwas_gate  # direct GWAS hit with strong score
    tier2_non = (1 - is_gwas) * conv * scale * 0.3  # convergent but not direct hit

    non_onc_score = (
        tier1_non * 8.0                        # tier 1 base: massive head start
        + tier1_non * conv * scale * 2.0       # tier 1 tiebreak: convergence within GWAS hits
        + tier1_non * pli * 1.5                # tier 1 bonus: constrained GWAS hits
        + tier2_non * 3.0                      # tier 2 base
        + conv * scale * 0.8                   # residual convergence signal
        + pli * 0.7
        + X[:, 11] * 1.2 + X[:, 12] * 0.8    # clinvar + l2g
        + X[:, 15] * 0.3
    )

    # ── Oncology: hierarchical mutation ──
    mut_gate = 1.0 / (1.0 + np.exp(-25 * (mut_freq - 0.02)))
    is_mut = X[:, 13]

    tier1_onc = is_mut * mut_gate  # significantly mutated (soft: weighted by frequency)
    tier2_onc = (1 - is_mut) * conv * 0.3  # convergent but not mutated

    onc_score = (
        tier1_onc * 8.0                                    # tier 1 base
        + tier1_onc * np.log1p(mut_freq * 20) * 3.0       # tier 1: higher frequency = better
        + tier1_onc * conv * 2.0                           # tier 1: convergent mutated genes
        + tier1_onc * pli * 2.0                            # tier 1: constrained mutated genes
        + tier2_onc * 3.0                                  # tier 2 base
        + conv * 0.5                                       # residual convergence
        + pli * 0.4
    )

    score = np.where(is_onc, onc_score, non_onc_score)
    return score
