"""
Automated A/B experiment runner for epistemic blinding.

Uses `claude -p` (Claude Code headless mode) to send each prompt in a fresh
session — no API key needed, works with Claude Max subscription.

Each seed generates a different row shuffle, then both blinded and unblinded
prompts are sent to Claude. Responses are saved, compared, and an aggregate
report is produced.

Usage:
    python run_experiment.py                   # 5 seeds, default model
    python run_experiment.py --seeds 3         # fewer seeds for a quick test
    python run_experiment.py --model sonnet    # use sonnet (faster/cheaper)
    python run_experiment.py --dry-run         # generate prompts only, don't call Claude
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _find_claude() -> str:
    """Locate the claude executable, handling Windows .cmd wrappers."""
    # shutil.which handles Windows PATH + PATHEXT resolution
    found = shutil.which("claude")
    if found:
        return found
    # Fallback: known npm global install location on Windows
    fallback = Path.home() / "AppData/Roaming/npm/claude.cmd"
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("Cannot find 'claude' CLI. Is it installed and on PATH?")

SEEDS = [42, 137, 256, 389, 501]
SCRIPT_DIR = Path(__file__).resolve().parent
BLIND_SCRIPT = SCRIPT_DIR.parents[1] / "scripts" / "blind.py"
COMPARE_SCRIPT = SCRIPT_DIR.parents[1] / "scripts" / "compare.py"
CONFIG_TEMPLATE = SCRIPT_DIR / "config.yaml"
FAMOUS_FILE = SCRIPT_DIR / "famous.txt"


def update_config_seed(config_path: Path, seed: int) -> None:
    """Rewrite the config YAML with a new shuffle seed."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ds in cfg.get("datasets", []):
        ds["shuffle_seed"] = seed
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def run_blinding(config_path: Path) -> None:
    """Run blind.py to regenerate prompts for current seed."""
    result = subprocess.run(
        [sys.executable, str(BLIND_SCRIPT), str(config_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  [ERROR] blind.py failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("blind.py failed")


def send_to_claude(prompt_path: Path, model: str | None = None) -> str:
    """Send a prompt file to claude -p via stdin and return the response text.

    Uses stdin piping instead of passing the prompt as a CLI argument,
    because the prompt is too large for Windows' command-line length limit.
    """
    prompt_text = prompt_path.read_text(encoding="utf-8")

    cmd = [_find_claude(), "-p", "-"]
    if model:
        cmd.extend(["--model", model])

    print(f"    Sending to Claude ({prompt_path.name})...")
    start = time.time()
    result = subprocess.run(
        cmd, input=prompt_text, capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - start
    print(f"    Done ({elapsed:.0f}s)")

    if result.returncode != 0:
        print(f"  [ERROR] claude -p failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("claude -p failed")

    return result.stdout


def run_comparison(
    mapping_path: Path,
    blinded_response_path: Path,
    unblinded_response_path: Path,
    top_k: int,
) -> dict:
    """Run compare.py and return the JSON report."""
    cmd = [
        sys.executable, str(COMPARE_SCRIPT),
        "--mapping", str(mapping_path),
        "--blinded", str(blinded_response_path),
        "--unblinded", str(unblinded_response_path),
        "--top-k", str(top_k),
        "--json",
    ]
    if FAMOUS_FILE.exists():
        cmd.extend(["--famous", str(FAMOUS_FILE)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"  [ERROR] compare.py failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("compare.py failed")

    return json.loads(result.stdout)


def aggregate_reports(reports: list[dict]) -> dict:
    """Compute aggregate statistics across multiple seed runs."""
    import statistics

    n = len(reports)
    numeric_keys = ["overlap", "jaccard", "mean_rank_delta_shared", "kendall_tau"]
    agg: dict = {"n_runs": n}

    for key in numeric_keys:
        vals = [r[key] for r in reports if not (isinstance(r[key], float) and r[key] != r[key])]
        if vals:
            agg[key] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
            }

    # Fame bias
    if "fame_bias_delta" in reports[0]:
        deltas = [r["fame_bias_delta"] for r in reports]
        agg["fame_bias_delta"] = {
            "mean": statistics.mean(deltas),
            "std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "values": deltas,
        }

    # Frequency of promoted/demoted names across runs
    from collections import Counter
    promoted_counts = Counter()
    demoted_counts = Counter()
    for r in reports:
        promoted_counts.update(r.get("promoted_in_unblinded", []))
        demoted_counts.update(r.get("demoted_in_unblinded", []))

    agg["consistently_promoted"] = {k: v for k, v in promoted_counts.most_common() if v >= 2}
    agg["consistently_demoted"] = {k: v for k, v in demoted_counts.most_common() if v >= 2}

    return agg


def format_aggregate(agg: dict) -> str:
    """Format aggregate report as human-readable text."""
    lines = [
        "",
        "=" * 60,
        f"AGGREGATE RESULTS ({agg['n_runs']} runs)",
        "=" * 60,
    ]

    for key in ["overlap", "jaccard", "mean_rank_delta_shared", "kendall_tau"]:
        if key in agg:
            s = agg[key]
            lines.append(f"  {key}: {s['mean']:.3f} +/- {s['std']:.3f}  (range: {s['min']:.3f} - {s['max']:.3f})")

    if "fame_bias_delta" in agg:
        fd = agg["fame_bias_delta"]
        lines.append(f"  fame_bias_delta: {fd['mean']:+.1f} +/- {fd['std']:.1f}  (per-run: {fd['values']})")

    if agg.get("consistently_promoted"):
        lines.append("")
        lines.append("  Consistently promoted when unblinded (appeared in 2+ runs):")
        for name, count in sorted(agg["consistently_promoted"].items(), key=lambda x: -x[1]):
            lines.append(f"    + {name} ({count}/{agg['n_runs']} runs)")

    if agg.get("consistently_demoted"):
        lines.append("")
        lines.append("  Consistently demoted when unblinded (appeared in 2+ runs):")
        for name, count in sorted(agg["consistently_demoted"].items(), key=lambda x: -x[1]):
            lines.append(f"    - {name} ({count}/{agg['n_runs']} runs)")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run epistemic blinding A/B experiment")
    ap.add_argument("--seeds", type=int, default=5, help="Number of seeds to run (default: 5)")
    ap.add_argument("--model", default=None, help="Claude model to use (e.g., 'sonnet', 'opus')")
    ap.add_argument("--top-k", type=int, default=20, help="Top-K list size (default: 20)")
    ap.add_argument("--dry-run", action="store_true", help="Generate prompts only, don't call Claude")
    args = ap.parse_args()

    seeds = SEEDS[:args.seeds]
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # Work on a copy of the config so we don't mutate the template
    work_config = SCRIPT_DIR / "config_run.yaml"
    work_config.write_text(CONFIG_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")

    reports: list[dict] = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"Run {i}/{len(seeds)} — seed {seed}")
        print(f"{'='*60}")

        # 1. Update seed and regenerate prompts
        update_config_seed(work_config, seed)
        run_blinding(work_config)

        output_dir = SCRIPT_DIR / "output"
        blinded_prompt = output_dir / "blinded_prompt.txt"
        unblinded_prompt = output_dir / "unblinded_prompt.txt"
        mapping = output_dir / "mapping.json"

        if args.dry_run:
            # Save prompts for manual review
            for src, label in [(blinded_prompt, "blinded"), (unblinded_prompt, "unblinded")]:
                dst = results_dir / f"seed{seed}_{label}_prompt.txt"
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  [dry-run] Prompts saved to results/seed{seed}_*.txt")
            continue

        # 2. Send both prompts to Claude (sequential — fresh session each)
        blinded_response_text = send_to_claude(blinded_prompt, model=args.model)
        unblinded_response_text = send_to_claude(unblinded_prompt, model=args.model)

        # 3. Save responses
        blinded_resp_path = results_dir / f"seed{seed}_blinded_response.txt"
        unblinded_resp_path = results_dir / f"seed{seed}_unblinded_response.txt"
        blinded_resp_path.write_text(blinded_response_text, encoding="utf-8")
        unblinded_resp_path.write_text(unblinded_response_text, encoding="utf-8")

        # Also save the mapping for this seed (codes change per shuffle)
        mapping_copy = results_dir / f"seed{seed}_mapping.json"
        mapping_copy.write_text(mapping.read_text(encoding="utf-8"), encoding="utf-8")

        # 4. Compare
        print(f"  Comparing...")
        report = run_comparison(mapping, blinded_resp_path, unblinded_resp_path, args.top_k)
        report["seed"] = seed
        reports.append(report)

        # Save per-run report
        report_path = results_dir / f"seed{seed}_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Print per-run summary
        print(f"  Overlap: {report['overlap']}/{args.top_k}")
        print(f"  Jaccard: {report['jaccard']:.3f}")
        print(f"  Promoted when unblinded: {report.get('promoted_in_unblinded', [])}")
        if "fame_bias_delta" in report:
            print(f"  Fame bias delta: {report['fame_bias_delta']:+d}")

    if args.dry_run:
        print(f"\n[dry-run] All prompts saved to {results_dir}/")
        print("Review them, then remove --dry-run to send to Claude.")
        return 0

    # 5. Aggregate
    if len(reports) >= 2:
        agg = aggregate_reports(reports)
        agg_text = format_aggregate(agg)
        print(agg_text)

        # Save aggregate
        agg_path = results_dir / "aggregate_report.json"
        agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")

        summary_path = results_dir / "aggregate_report.txt"
        summary_path.write_text(agg_text, encoding="utf-8")
        print(f"\nFull results saved to {results_dir}/")

    # Restore original config
    work_config.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
