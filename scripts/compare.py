"""
A/B comparison between blinded and unblinded LLM responses.

Given two response files (each a ranked list) and the mapping file, this
script quantifies parametric-knowledge contamination. Metrics:

    1. Set overlap   — how many entities appear in both top-K lists
    2. Jaccard index — |A ∩ B| / |A ∪ B|
    3. Rank delta    — mean absolute rank change for shared entities
    4. Kendall tau   — rank correlation for the set intersection
    5. Promoted      — entities in unblinded top-K but not blinded
    6. Demoted       — entities in blinded top-K but not unblinded
    7. (optional) Fame bias — if a `famous_entities` list is provided,
                    reports how many famous entities appear in each list

Usage:
    python compare.py \
        --mapping output/mapping.json \
        --blinded blinded_response.txt \
        --unblinded unblinded_response.txt \
        [--top-k 20] \
        [--famous famous.txt]

Response file format: a numbered list like
    1. Company_042 - some justification
    2. Company_017 - another justification
Lines that don't match the "<number>. <identifier>" pattern are ignored.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from deblind import load_reverse_map, deblind_text


RANK_LINE = re.compile(r"^\s*(\d+)[\.\)]\s*([^\s,;:\-]+)")


def _clean_identifier(raw: str) -> str:
    """Strip markdown formatting (bold, backticks) from a parsed identifier."""
    s = raw.rstrip(",:-")
    s = s.strip("*`_")          # **AAPL** → AAPL, `AAPL` → AAPL
    return s


def parse_ranked_list(text: str, top_k: int) -> list[str]:
    """Extract the first identifier from each numbered line."""
    items: list[tuple[int, str]] = []
    for line in text.splitlines():
        m = RANK_LINE.match(line)
        if m:
            items.append((int(m.group(1)), _clean_identifier(m.group(2))))
    items.sort(key=lambda x: x[0])
    return [tok for _, tok in items[:top_k]]


def kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Kendall tau over the intersection of two ranked lists.

    Returns 1.0 if the shared entities appear in the same relative order
    in both lists, -1.0 if fully reversed, 0.0 for independent orderings.
    """
    shared = [x for x in order_a if x in order_b]
    if len(shared) < 2:
        return float("nan")
    idx_b = {x: i for i, x in enumerate(order_b)}
    concordant = discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            a_order = 1 if i < j else -1
            b_order = 1 if idx_b[shared[i]] < idx_b[shared[j]] else -1
            if a_order == b_order:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else float("nan")


def compare(
    blinded_ranked: list[str],
    unblinded_ranked: list[str],
    famous: set[str] | None = None,
) -> dict:
    set_a = set(blinded_ranked)
    set_b = set(unblinded_ranked)
    shared = set_a & set_b
    union = set_a | set_b
    jaccard = len(shared) / len(union) if union else float("nan")

    rank_a = {x: i + 1 for i, x in enumerate(blinded_ranked)}
    rank_b = {x: i + 1 for i, x in enumerate(unblinded_ranked)}
    if shared:
        deltas = [abs(rank_a[x] - rank_b[x]) for x in shared]
        mean_delta = sum(deltas) / len(deltas)
    else:
        mean_delta = float("nan")

    tau = kendall_tau(blinded_ranked, unblinded_ranked)

    promoted = [x for x in unblinded_ranked if x not in set_a]   # appear only in unblinded
    demoted = [x for x in blinded_ranked if x not in set_b]      # appear only in blinded

    report = {
        "top_k": max(len(blinded_ranked), len(unblinded_ranked)),
        "blinded_count": len(blinded_ranked),
        "unblinded_count": len(unblinded_ranked),
        "overlap": len(shared),
        "jaccard": jaccard,
        "mean_rank_delta_shared": mean_delta,
        "kendall_tau": tau,
        "promoted_in_unblinded": promoted,
        "demoted_in_unblinded": demoted,
    }

    if famous is not None:
        fam_b = sorted(set_a & famous)
        fam_u = sorted(set_b & famous)
        report["famous_in_blinded"] = fam_b
        report["famous_in_unblinded"] = fam_u
        report["fame_bias_delta"] = len(fam_u) - len(fam_b)

    return report


def format_report(report: dict) -> str:
    lines = [
        "Epistemic Blinding -- A/B comparison",
        "=" * 40,
        f"Top-K:                    {report['top_k']}",
        f"Blinded list size:        {report['blinded_count']}",
        f"Unblinded list size:      {report['unblinded_count']}",
        f"Set overlap:              {report['overlap']} / {report['top_k']}",
        f"Jaccard index:            {report['jaccard']:.3f}",
        f"Mean rank delta (shared): {report['mean_rank_delta_shared']:.2f}",
        f"Kendall tau (shared):     {report['kendall_tau']:.3f}",
        "",
        f"Promoted when unblinded ({len(report['promoted_in_unblinded'])}):",
    ]
    for x in report["promoted_in_unblinded"]:
        lines.append(f"  + {x}")
    lines.append("")
    lines.append(f"Demoted when unblinded ({len(report['demoted_in_unblinded'])}):")
    for x in report["demoted_in_unblinded"]:
        lines.append(f"  - {x}")

    if "fame_bias_delta" in report:
        lines.append("")
        lines.append(f"Famous entities in blinded list:   {len(report['famous_in_blinded'])}")
        lines.append(f"Famous entities in unblinded list: {len(report['famous_in_unblinded'])}")
        lines.append(f"Fame bias delta:                   {report['fame_bias_delta']:+d}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare blinded vs unblinded LLM responses")
    ap.add_argument("--mapping", required=True, help="Path to mapping.json from blind.py")
    ap.add_argument("--blinded", required=True, help="LLM response to the blinded prompt")
    ap.add_argument("--unblinded", required=True, help="LLM response to the unblinded prompt")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--famous", help="Optional file of famous entity names (one per line)")
    ap.add_argument("--json", action="store_true", help="Emit JSON report instead of text")
    args = ap.parse_args()

    reverse = load_reverse_map(args.mapping)
    blinded_text = Path(args.blinded).read_text(encoding="utf-8")
    unblinded_text = Path(args.unblinded).read_text(encoding="utf-8")

    # Resolve the blinded response into real identifiers so both lists
    # are in the same namespace for comparison.
    blinded_resolved = deblind_text(blinded_text, reverse)

    blinded_ranked = parse_ranked_list(blinded_resolved, args.top_k)
    unblinded_ranked = parse_ranked_list(unblinded_text, args.top_k)

    famous: set[str] | None = None
    if args.famous:
        famous = {
            line.strip()
            for line in Path(args.famous).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }

    report = compare(blinded_ranked, unblinded_ranked, famous)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
