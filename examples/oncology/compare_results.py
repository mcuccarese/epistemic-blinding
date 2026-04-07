"""
Compare blinded vs unblinded LLM responses to quantify epistemic contamination.

Expected input: text files with numbered gene lists in:
  blinded_responses/<indication>.txt
  unblinded_responses/<indication>.txt

Metrics computed:
  1. Rank correlation (Kendall tau) between blinded and unblinded top-20 lists
  2. Set overlap: how many genes appear in both top-20 lists?
  3. "Fame bias": do well-known genes (TP53, KRAS, EGFR, BRAF) rank higher
     in unblinded vs blinded?
  4. "Obscurity penalty": do obscure genes with strong features rank lower
     in unblinded?
  5. Target recovery: which condition better recovers approved drug targets?

Usage:
    python compare_results.py
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

EXPERIMENT_DIR = Path(__file__).parent

# Famous genes that LLMs will "know" about from training
FAMOUS_GENES = {
    "TP53", "KRAS", "EGFR", "BRAF", "PIK3CA", "PTEN", "RB1", "APC",
    "MYC", "ERBB2", "NRAS", "CDKN2A", "NF1", "IDH1", "IDH2", "FLT3",
    "NPM1", "BRCA1", "BRCA2", "ALK", "RET", "MET", "FGFR2", "FGFR3",
    "KIT", "PDGFRA", "VHL", "SMAD4", "FBXW7", "ATM", "ARID1A",
}

# Load answer key
with open(EXPERIMENT_DIR / "answer_key.json") as f:
    answer_key = json.load(f)


def parse_response(text):
    """Extract ranked gene list from LLM response text.

    Expects numbered list like:
    1. Gene_042 - justification text
    2. KRAS - some reasoning
    """
    genes = []
    for line in text.strip().split("\n"):
        line = line.strip()
        # Match patterns like "1. Gene_042" or "1. KRAS" or "1) TP53"
        m = re.match(r'^\d+[\.\)]\s*(\S+)', line)
        if m:
            gene = m.group(1).rstrip(",:-")
            genes.append(gene.upper())
    return genes[:20]  # cap at 20


def resolve_anon(gene_list, anon_map):
    """Convert Gene_NNN to real symbols using anon_map."""
    resolved = []
    for g in gene_list:
        if g in anon_map:
            resolved.append(anon_map[g].upper())
        else:
            resolved.append(g)
    return resolved


def compute_metrics(blinded_genes, unblinded_genes, targets, indication):
    """Compare two top-20 lists."""
    b_set = set(blinded_genes[:20])
    u_set = set(unblinded_genes[:20])

    # 1. Set overlap
    overlap = b_set & u_set
    overlap_frac = len(overlap) / 20 if len(b_set) == 20 and len(u_set) == 20 else 0

    # 2. Target recovery
    target_syms = {t["symbol"].upper() for t in targets}
    b_targets = b_set & target_syms
    u_targets = u_set & target_syms

    # 3. Fame bias: are famous genes ranked higher in unblinded?
    b_famous = [i for i, g in enumerate(blinded_genes) if g in FAMOUS_GENES]
    u_famous = [i for i, g in enumerate(unblinded_genes) if g in FAMOUS_GENES]
    b_famous_mean_rank = sum(b_famous) / len(b_famous) + 1 if b_famous else None
    u_famous_mean_rank = sum(u_famous) / len(u_famous) + 1 if u_famous else None

    # 4. Famous gene count in top-20
    b_famous_count = len(b_set & FAMOUS_GENES)
    u_famous_count = len(u_set & FAMOUS_GENES)

    # 5. Genes ONLY in unblinded (fame pulled them in?)
    only_unblinded = u_set - b_set
    only_blinded = b_set - u_set

    famous_only_unblinded = only_unblinded & FAMOUS_GENES
    obscure_only_blinded = only_blinded - FAMOUS_GENES

    return {
        "indication": indication,
        "overlap": len(overlap),
        "overlap_pct": f"{overlap_frac:.0%}",
        "blinded_targets_found": len(b_targets),
        "unblinded_targets_found": len(u_targets),
        "blinded_famous_in_top20": b_famous_count,
        "unblinded_famous_in_top20": u_famous_count,
        "famous_mean_rank_blinded": round(b_famous_mean_rank, 1) if b_famous_mean_rank else "N/A",
        "famous_mean_rank_unblinded": round(u_famous_mean_rank, 1) if u_famous_mean_rank else "N/A",
        "famous_ONLY_in_unblinded": sorted(famous_only_unblinded),
        "obscure_ONLY_in_blinded": sorted(obscure_only_blinded),
        "only_in_unblinded": sorted(only_unblinded),
        "only_in_blinded": sorted(only_blinded),
    }


def main():
    blinded_dir = EXPERIMENT_DIR / "blinded_responses"
    unblinded_dir = EXPERIMENT_DIR / "unblinded_responses"

    if not blinded_dir.exists() or not unblinded_dir.exists():
        print("ERROR: Response directories not found.")
        print(f"  Expected: {blinded_dir}/")
        print(f"  Expected: {unblinded_dir}/")
        print(f"\nRun the experiment first:")
        print(f"  1. Open a FRESH LLM session")
        print(f"  2. Paste each prompt from prompts/blinded/ and save response")
        print(f"  3. Open ANOTHER fresh LLM session")
        print(f"  4. Paste each prompt from prompts/unblinded/ and save response")
        print(f"  5. Save as blinded_responses/<name>.txt and unblinded_responses/<name>.txt")
        sys.exit(1)

    all_metrics = []

    for indication, key_data in answer_key.items():
        safe_name = indication.lower().replace(" ", "_").replace("-", "_").replace("/", "_")[:40]

        b_file = blinded_dir / f"{safe_name}.txt"
        u_file = unblinded_dir / f"{safe_name}.txt"

        if not b_file.exists() or not u_file.exists():
            print(f"  Skipping {indication}: response files not found")
            continue

        b_text = b_file.read_text(encoding="utf-8")
        u_text = u_file.read_text(encoding="utf-8")

        b_genes = parse_response(b_text)
        u_genes = parse_response(u_text)

        # Resolve blinded names to real symbols
        anon_map = {k.upper(): v for k, v in key_data["anon_map"].items()}
        b_genes_resolved = resolve_anon(b_genes, anon_map)

        metrics = compute_metrics(b_genes_resolved, u_genes,
                                  key_data["approved_targets_in_set"], indication)
        all_metrics.append(metrics)

    if not all_metrics:
        return

    # Print results
    print()
    print("=" * 100)
    print("  EPISTEMIC BLINDING EXPERIMENT: RESULTS")
    print("=" * 100)

    for m in all_metrics:
        print(f"\n--- {m['indication']} ---")
        print(f"  Top-20 overlap: {m['overlap']}/20 ({m['overlap_pct']})")
        print(f"  Approved targets found: blinded={m['blinded_targets_found']}, "
              f"unblinded={m['unblinded_targets_found']}")
        print(f"  Famous genes in top-20: blinded={m['blinded_famous_in_top20']}, "
              f"unblinded={m['unblinded_famous_in_top20']}")
        print(f"  Famous gene mean rank: blinded={m['famous_mean_rank_blinded']}, "
              f"unblinded={m['famous_mean_rank_unblinded']}")

        if m["famous_ONLY_in_unblinded"]:
            print(f"  FAME BIAS: these famous genes appear ONLY in unblinded top-20:")
            for g in m["famous_ONLY_in_unblinded"]:
                print(f"    - {g}")

        if m["obscure_ONLY_in_blinded"]:
            print(f"  OBSCURITY SIGNAL: these lesser-known genes appear ONLY in blinded top-20:")
            for g in m["obscure_ONLY_in_blinded"]:
                print(f"    - {g}")

    # Summary
    print(f"\n{'=' * 100}")
    print("  SUMMARY")
    print(f"{'=' * 100}")

    avg_overlap = sum(m["overlap"] for m in all_metrics) / len(all_metrics)
    avg_b_targets = sum(m["blinded_targets_found"] for m in all_metrics) / len(all_metrics)
    avg_u_targets = sum(m["unblinded_targets_found"] for m in all_metrics) / len(all_metrics)
    avg_b_famous = sum(m["blinded_famous_in_top20"] for m in all_metrics) / len(all_metrics)
    avg_u_famous = sum(m["unblinded_famous_in_top20"] for m in all_metrics) / len(all_metrics)

    print(f"  Average top-20 overlap: {avg_overlap:.1f}/20 ({avg_overlap/20:.0%})")
    print(f"  Average targets found: blinded={avg_b_targets:.1f}, unblinded={avg_u_targets:.1f}")
    print(f"  Average famous genes in top-20: blinded={avg_b_famous:.1f}, unblinded={avg_u_famous:.1f}")

    print(f"\n  INTERPRETATION:")
    if avg_u_famous > avg_b_famous + 1:
        print(f"  >>> FAME BIAS DETECTED: Unblinded condition includes {avg_u_famous-avg_b_famous:.1f} more")
        print(f"      famous genes on average. The LLM's training knowledge is pulling well-known")
        print(f"      genes into the top-20 regardless of their actual feature profile.")
    if avg_overlap < 15:
        print(f"  >>> SIGNIFICANT DIVERGENCE: Only {avg_overlap:.0f}/20 genes overlap between conditions.")
        print(f"      Knowing gene names changes the LLM's ranking substantially.")
    if avg_b_targets > avg_u_targets:
        print(f"  >>> BLINDING IMPROVES TARGET RECOVERY: The blinded condition finds more actual")
        print(f"      drug targets, suggesting that fame bias in the unblinded condition displaces")
        print(f"      real targets with famous-but-wrong genes.")
    elif avg_u_targets > avg_b_targets:
        print(f"  >>> UNBLINDED FINDS MORE TARGETS: The LLM's prior knowledge may be helping,")
        print(f"      but this could also mean the model is 'cheating' by recognizing known targets.")

    # Save full results
    with open(EXPERIMENT_DIR / "experiment_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Full results saved to: {EXPERIMENT_DIR / 'experiment_results.json'}")


if __name__ == "__main__":
    main()
