"""
De-blinding helper.

Takes an LLM response that was produced from a blinded prompt and a
mapping.json file, and rewrites all anonymous entity codes back into
their real identifiers.

Usage:
    python deblind.py --mapping output/mapping.json --input response.txt

    # or pipe stdin → stdout
    cat response.txt | python deblind.py --mapping output/mapping.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def load_reverse_map(mapping_path: str | Path) -> dict[str, str]:
    """Flatten reverse mappings from all entity types into one lookup dict.

    Anonymous codes are constructed as `{prefix}_{number}` and prefixes
    are chosen by the user. Collisions across entity types would only
    happen if two types shared a prefix AND a number — unlikely in
    practice, but if it happens we warn and keep the first one.
    """
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if "reverse" not in mapping:
        raise ValueError(f"{mapping_path} does not contain a 'reverse' section")
    flat: dict[str, str] = {}
    for et, table in mapping["reverse"].items():
        for anon, real in table.items():
            if anon in flat and flat[anon] != real:
                print(
                    f"[warn] anonymous code '{anon}' appears under multiple entity_types "
                    f"with different values; keeping first",
                    file=sys.stderr,
                )
                continue
            flat[anon] = real
    return flat


def deblind_text(text: str, reverse: dict[str, str]) -> str:
    """Replace all anonymous codes in a text with their real values.

    We match tokens of the form WORD_NUMBER (e.g., Company_001, Gene_042).
    The regex is intentionally permissive about the prefix shape so users
    can choose any alphanumeric prefix in their config.
    """
    if not reverse:
        return text
    # Sort keys by length descending so longer codes match before prefixes
    # of shorter codes (defensive; unlikely to matter given the _NNN suffix).
    keys = sorted(reverse.keys(), key=len, reverse=True)
    # Build one alternation regex; word-boundary on both sides so we don't
    # chew into neighboring identifiers.
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    return pattern.sub(lambda m: reverse[m.group(1)], text)


def main() -> int:
    ap = argparse.ArgumentParser(description="De-blind an LLM response using a mapping file")
    ap.add_argument("--mapping", required=True, help="Path to mapping.json produced by blind.py")
    ap.add_argument("--input", help="Input file (default: stdin)")
    ap.add_argument("--output", help="Output file (default: stdout)")
    args = ap.parse_args()

    reverse = load_reverse_map(args.mapping)

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    resolved = deblind_text(text, reverse)

    if args.output:
        Path(args.output).write_text(resolved, encoding="utf-8")
    else:
        sys.stdout.write(resolved)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
