"""
Epistemic Blinding Experiment: Does knowing gene names change LLM target predictions?

Generates two matched prompt sets for 4 indications:
  A) BLINDED: Gene_001 through Gene_100 with numerical features only
  B) UNBLINDED: Real gene symbols with identical numerical features

A fresh LLM session (with NO context about our prior work) receives each
prompt and ranks the top 20 most promising drug targets.

If the top-20 lists differ, that proves parametric knowledge contamination.

Output:
  - prompts/blinded/   — one .txt per indication (anonymized gene names)
  - prompts/unblinded/ — one .txt per indication (real gene names)
  - answer_key.json    — ground truth for comparison
  - compare_results.py — script to quantify divergence after LLM responds
"""
import json
import numpy as np
import duckdb
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

PROJECT_DIR = Path("C:/Users/mcucc/Projects/Biological truth")
DB_PATH = Path("D:/Mike project data/Biological truth/graph_db/truth_finder.duckdb")
OUT_DIR = Path(__file__).parent

# Load eval data
data = np.load(PROJECT_DIR / "evolve/eval_data/eval_data.npz", allow_pickle=True)
X = data["X"]
y = data["y"]
disease_ids = data["disease_ids"]
disease_names = list(data["disease_names"])
gene_ids = list(data["gene_ids"])

# Gene symbol map
con = duckdb.connect(str(DB_PATH), read_only=True)
sym_df = con.execute(
    "SELECT DISTINCT gene_id, FIRST(gene_symbol) as sym FROM genes "
    "WHERE gene_id LIKE 'ENSG%' GROUP BY gene_id"
).fetchdf()
id_to_sym = {}
for _, r in sym_df.iterrows():
    if r["sym"] and isinstance(r["sym"], str):
        id_to_sym[r["gene_id"]] = r["sym"]
con.close()

# Scoring function (gen12_breed_v3)
def score_gen12v3(X):
    is_onc = X[:, 21] > 0.5
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
    gwas_gate = 1.0 / (1.0 + np.exp(-10 * (X[:, 10] - 0.1)))
    is_gwas = X[:, 9]
    scale = np.where(n_sg < 1000, 1.8, np.where(n_sg > 5000, 0.6, 1.0))
    tier1_non = is_gwas * gwas_gate
    non_onc_score = (tier1_non * 8.0 + tier1_non * conv * scale * 2.0
                     + tier1_non * pli * 1.5 + (1 - is_gwas) * conv * scale * 0.9
                     + pli * 0.7 + X[:, 11] * 1.2 + X[:, 12] * 0.8 + X[:, 15] * 0.3)
    mut_gate = 1.0 / (1.0 + np.exp(-25 * (mut_freq - 0.02)))
    is_mut = X[:, 13]
    tier1_onc = is_mut * mut_gate
    onc_score = (tier1_onc * 8.0 + tier1_onc * np.log1p(mut_freq * 20) * 3.0
                 + tier1_onc * conv * 2.0 + tier1_onc * pli * 2.0
                 + (1 - is_mut) * conv * 0.3 + conv * 0.5 + pli * 0.4)
    return np.where(is_onc, onc_score, non_onc_score)


# Indications to test — chosen to span different biology types
# Mix of well-known (AML, GBM) and less-famous (CIN CRC, Pancreatic)
TEST_INDICATIONS = [
    "Acute myeloid leukemia",
    "IDH-wildtype glioblastoma",
    "Pancreatic adenocarcinoma",
    "Chromosomally unstable colorectal cancer",
]

FEATURE_DESCRIPTIONS = """Feature descriptions (provided to the LLM for both conditions):
- mut_freq: Fraction of tumor samples with somatic mutations in this gene (0-1). Higher = more frequently mutated.
- is_mutated: Whether the gene passes the significance threshold for mutation frequency (YES/no).
- pLI: Probability of loss-of-function intolerance (0-1). Higher = gene is more constrained/essential.
- esm2_enrichment: Protein structure similarity enrichment. How much more often this gene's structural neighbors are disease-associated vs chance. >1 = enriched.
- geneformer_enrichment: Single-cell transcriptomic co-expression enrichment. Same interpretation as above.
- prottrans_enrichment: Protein function domain enrichment. Same interpretation.
- n_neighbors: Total experimental biology neighbors across all modalities.
"""

SYSTEM_PROMPT = """You are a computational biologist analyzing gene-level features to identify
the most promising drug targets for a specific cancer indication.

You will be given a table of 100 genes with their features. Your task is to
select and rank the TOP 20 genes that are most likely to be successful drug
targets for this cancer type.

Base your ranking ONLY on the features provided. Consider:
1. Mutation frequency and significance (is the gene a driver?)
2. Genetic constraint (pLI) — constrained genes are more likely to be essential
3. Experimental convergence — genes whose neighbors in protein structure,
   transcriptomics, and function are also disease-associated
4. The combination of multiple signals is stronger than any single signal

Return your answer as a numbered list of 20 genes, ranked from most to least
promising. For each gene, give a 1-sentence justification.
"""

answer_key = {}

for indication in TEST_INDICATIONS:
    di = disease_names.index(indication)
    mask = disease_ids == di
    X_d = X[mask]
    y_d = y[mask]

    # Score and rank all genes
    scores = score_gen12v3(X_d)
    scores = np.nan_to_num(scores, nan=0.0)
    ranked = np.argsort(-scores)

    # Take top 100 by model score (gives LLM a manageable set that includes
    # real targets and strong non-targets)
    top100_indices = ranked[:100]

    # Shuffle so position in the table doesn't signal rank
    rng = np.random.RandomState(42)
    shuffled = top100_indices.copy()
    rng.shuffle(shuffled)

    # Create anonymous ID mapping
    anon_map = {}  # Gene_NNN -> real symbol
    rows_blinded = []
    rows_unblinded = []

    for pos, gi in enumerate(shuffled):
        gene_id = gene_ids[gi]
        real_sym = id_to_sym.get(gene_id, gene_id)
        anon_id = f"Gene_{pos+1:03d}"
        anon_map[anon_id] = real_sym

        is_target = y_d[gi] > 0.5
        is_mut = X_d[gi, 13] > 0.5
        mut_freq = X_d[gi, 20]
        pli = X_d[gi, 16]
        esm2_e = X_d[gi, 6]
        gf_e = X_d[gi, 7]
        pt_e = X_d[gi, 8]
        n_nbrs = X_d[gi, 18]

        row_data = {
            "mut_freq": f"{mut_freq:.3f}" if is_mut else "0.000",
            "is_mutated": "YES" if is_mut else "no",
            "pLI": f"{pli:.2f}",
            "esm2_enrichment": f"{esm2_e:.1f}",
            "geneformer_enrichment": f"{gf_e:.1f}",
            "prottrans_enrichment": f"{pt_e:.1f}",
            "n_neighbors": f"{int(n_nbrs)}",
        }

        rows_blinded.append((anon_id, row_data))
        rows_unblinded.append((real_sym, row_data))

    # Build table strings
    header = f"{'Gene':<12} {'mut_freq':>8} {'mutated':>7} {'pLI':>5} {'ESM2enr':>8} {'GFenr':>8} {'PTenr':>8} {'n_nbrs':>7}"
    sep = "-" * len(header)

    def build_table(rows):
        lines = [header, sep]
        for name, rd in rows:
            lines.append(
                f"{name:<12} {rd['mut_freq']:>8} {rd['is_mutated']:>7} {rd['pLI']:>5} "
                f"{rd['esm2_enrichment']:>8} {rd['geneformer_enrichment']:>8} "
                f"{rd['prottrans_enrichment']:>8} {rd['n_neighbors']:>7}"
            )
        return "\n".join(lines)

    table_blinded = build_table(rows_blinded)
    table_unblinded = build_table(rows_unblinded)

    # Safe indication name for filenames
    safe_name = indication.lower().replace(" ", "_").replace("-", "_").replace("/", "_")[:40]

    # Write blinded prompt
    blinded_dir = OUT_DIR / "prompts" / "blinded"
    blinded_dir.mkdir(parents=True, exist_ok=True)
    blinded_prompt = f"""{SYSTEM_PROMPT}

{FEATURE_DESCRIPTIONS}

INDICATION: {indication}
Total significantly mutated genes in this cancer type: {int(X_d[0, 19])}

Below are 100 candidate genes with their features. Select and rank the top 20
most promising drug targets.

{table_blinded}

Please return your top 20 as a numbered list:
1. Gene_NNN - justification
2. Gene_NNN - justification
...
"""
    (blinded_dir / f"{safe_name}.txt").write_text(blinded_prompt, encoding="utf-8")

    # Write unblinded prompt
    unblinded_dir = OUT_DIR / "prompts" / "unblinded"
    unblinded_dir.mkdir(parents=True, exist_ok=True)
    unblinded_prompt = f"""{SYSTEM_PROMPT}

{FEATURE_DESCRIPTIONS}

INDICATION: {indication}
Total significantly mutated genes in this cancer type: {int(X_d[0, 19])}

Below are 100 candidate genes with their features. Select and rank the top 20
most promising drug targets.

{table_unblinded}

Please return your top 20 as a numbered list:
1. GENESYMBOL - justification
2. GENESYMBOL - justification
...
"""
    (unblinded_dir / f"{safe_name}.txt").write_text(unblinded_prompt, encoding="utf-8")

    # Answer key
    target_genes_in_100 = []
    model_top20_genes = []
    for pos, gi in enumerate(shuffled):
        real_sym = id_to_sym.get(gene_ids[gi], gene_ids[gi])
        anon_id = f"Gene_{pos+1:03d}"
        model_rank = int(np.where(ranked == gi)[0][0]) + 1
        is_target = bool(y_d[gi] > 0.5)
        if is_target:
            target_genes_in_100.append({"symbol": real_sym, "anon_id": anon_id, "model_rank": model_rank})
        if model_rank <= 20:
            model_top20_genes.append({"symbol": real_sym, "anon_id": anon_id, "model_rank": model_rank})

    answer_key[indication] = {
        "anon_map": anon_map,
        "approved_targets_in_set": target_genes_in_100,
        "model_top20": sorted(model_top20_genes, key=lambda x: x["model_rank"]),
        "n_candidates": 100,
        "n_sig_genes": int(X_d[0, 19]),
    }

    n_targets = len(target_genes_in_100)
    n_top20 = len(model_top20_genes)
    print(f"  {indication}: {n_targets} approved targets in candidate set, "
          f"{n_top20} model top-20 genes included")

# Save answer key
with open(OUT_DIR / "answer_key.json", "w") as f:
    json.dump(answer_key, f, indent=2)

print(f"\nExperiment prepared:")
print(f"  Blinded prompts: {blinded_dir}/")
print(f"  Unblinded prompts: {unblinded_dir}/")
print(f"  Answer key: {OUT_DIR / 'answer_key.json'}")
print(f"\nINSTRUCTIONS:")
print(f"  1. Open a FRESH LLM session (no prior context about this project)")
print(f"  2. Paste each blinded prompt and record the top-20 response")
print(f"  3. Open ANOTHER fresh session")
print(f"  4. Paste each unblinded prompt and record the top-20 response")
print(f"  5. Save responses as blinded_responses/<indication>.txt and unblinded_responses/<indication>.txt")
print(f"  6. Run compare_results.py to quantify divergence")
