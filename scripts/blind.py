"""
Epistemic blinding engine.

Takes a YAML config describing one or more datasets with named entities,
produces a blinded version of the data (and optionally an unblinded A/B
control), renders them as LLM-ready prompts, and writes a mapping table
that is kept out of prompt context.

Usage:
    python blind.py path/to/config.yaml

See examples/stocks_demo/config.yaml for a reference config.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ----------------------------- config schema ----------------------------- #

@dataclass
class EntityColumn:
    column: str                      # column name in the source dataframe
    entity_type: str                 # shared key for cross-dataset consistency
    prefix: str = "E"                # anonymous code prefix, e.g. "Company"
    preserve: list[str] = field(default_factory=list)  # values to leave un-blinded
    blind: bool = True               # if False, column is pass-through (A/B control helper)


@dataclass
class DatasetConfig:
    name: str
    path: str
    entity_columns: list[EntityColumn]
    columns: list[str] | None = None  # column whitelist; None = all columns
    shuffle: bool = True
    shuffle_seed: int = 42
    label: str | None = None          # human label rendered above the table


@dataclass
class TaskConfig:
    system_prompt: str = ""
    instructions: str = ""
    output_format: str = ""


@dataclass
class OutputConfig:
    dir: str = "output"
    produce_blinded: bool = True
    produce_unblinded: bool = True
    mapping_file: str = "mapping.json"
    table_format: str = "markdown"    # 'markdown' or 'fixed'


@dataclass
class Config:
    task: TaskConfig
    datasets: list[DatasetConfig]
    output: OutputConfig
    source_path: Path = Path(".")     # directory holding config.yaml, used for relative paths


def _parse_entity_column(d: dict) -> EntityColumn:
    return EntityColumn(
        column=d["column"],
        entity_type=d["entity_type"],
        prefix=d.get("prefix", "E"),
        preserve=list(d.get("preserve", []) or []),
        blind=bool(d.get("blind", True)),
    )


def _parse_dataset(d: dict) -> DatasetConfig:
    return DatasetConfig(
        name=d["name"],
        path=d["path"],
        entity_columns=[_parse_entity_column(ec) for ec in d.get("entity_columns", [])],
        columns=d.get("columns"),
        shuffle=bool(d.get("shuffle", True)),
        shuffle_seed=int(d.get("shuffle_seed", 42)),
        label=d.get("label"),
    )


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    task_d = raw.get("task", {}) or {}
    out_d = raw.get("output", {}) or {}

    cfg = Config(
        task=TaskConfig(
            system_prompt=task_d.get("system_prompt", ""),
            instructions=task_d.get("instructions", ""),
            output_format=task_d.get("output_format", ""),
        ),
        datasets=[_parse_dataset(d) for d in raw.get("datasets", [])],
        output=OutputConfig(
            dir=out_d.get("dir", "output"),
            produce_blinded=bool(out_d.get("produce_blinded", True)),
            produce_unblinded=bool(out_d.get("produce_unblinded", True)),
            mapping_file=out_d.get("mapping_file", "mapping.json"),
            table_format=out_d.get("table_format", "markdown"),
        ),
        source_path=path.parent,
    )

    if not cfg.datasets:
        raise ValueError("config must define at least one dataset under 'datasets:'")
    return cfg


# ----------------------------- blinding core ---------------------------- #

def _load_dataset(ds: DatasetConfig, base: Path) -> pd.DataFrame:
    p = Path(ds.path)
    if not p.is_absolute():
        p = base / p
    if not p.exists():
        raise FileNotFoundError(f"dataset '{ds.name}' file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in {".csv"}:
        df = pd.read_csv(p)
    elif suffix in {".tsv", ".txt"}:
        df = pd.read_csv(p, sep="\t")
    elif suffix in {".parquet"}:
        df = pd.read_parquet(p)
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(p, lines=(suffix == ".jsonl"))
    else:
        raise ValueError(f"unsupported file type for dataset '{ds.name}': {suffix}")
    if ds.columns is not None:
        missing = [c for c in ds.columns if c not in df.columns]
        if missing:
            raise ValueError(f"dataset '{ds.name}' missing columns: {missing}")
        df = df[ds.columns].copy()
    return df


def build_mapping_table(cfg: Config, frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    """Construct a shared mapping table keyed by entity_type.

    For each entity_type we collect every distinct value seen across ANY
    dataset that declares a column of that type, then assign anonymous
    codes in a deterministic-but-shuffled order. This guarantees that the
    same real entity maps to the same anonymous code everywhere it
    appears — which is what makes cross-dataset reasoning work after
    blinding.

    Returns: {entity_type: {real_value: anonymous_code}}
    """
    # Collect per-type state
    type_to_prefix: dict[str, str] = {}
    type_to_values: dict[str, set[str]] = {}
    type_to_preserve: dict[str, set[str]] = {}

    for ds in cfg.datasets:
        df = frames[ds.name]
        for ec in ds.entity_columns:
            if ec.column not in df.columns:
                raise ValueError(
                    f"dataset '{ds.name}': entity column '{ec.column}' not in dataframe"
                )
            if ec.entity_type not in type_to_prefix:
                type_to_prefix[ec.entity_type] = ec.prefix
                type_to_values[ec.entity_type] = set()
                type_to_preserve[ec.entity_type] = set()
            elif type_to_prefix[ec.entity_type] != ec.prefix:
                # Be forgiving — warn but keep the first prefix encountered
                print(
                    f"[warn] entity_type '{ec.entity_type}' used with conflicting prefixes "
                    f"('{type_to_prefix[ec.entity_type]}' vs '{ec.prefix}'); "
                    f"keeping '{type_to_prefix[ec.entity_type]}'",
                    file=sys.stderr,
                )
            if ec.blind:
                vals = df[ec.column].dropna().astype(str).unique().tolist()
                type_to_values[ec.entity_type].update(vals)
            type_to_preserve[ec.entity_type].update(ec.preserve)

    # Assign codes — deterministic shuffle per entity_type using a stable seed
    mapping: dict[str, dict[str, str]] = {}
    for et, values in type_to_values.items():
        preserve = type_to_preserve[et]
        to_blind = sorted(v for v in values if v not in preserve)
        # Deterministic shuffle so order in the table doesn't leak entity identity
        rng = random.Random(f"epistemic-blinding::{et}")
        rng.shuffle(to_blind)
        width = max(3, len(str(len(to_blind))))
        prefix = type_to_prefix[et]
        mapping[et] = {}
        for i, val in enumerate(to_blind, start=1):
            mapping[et][val] = f"{prefix}_{i:0{width}d}"
        # Preserved values map to themselves (documented in the mapping for clarity)
        for val in preserve:
            if val in values:
                mapping[et][val] = val
    return mapping


def apply_blinding(
    df: pd.DataFrame,
    ds: DatasetConfig,
    mapping: dict[str, dict[str, str]],
    blinded: bool,
) -> pd.DataFrame:
    out = df.copy()
    for ec in ds.entity_columns:
        if not blinded or not ec.blind:
            continue
        m = mapping.get(ec.entity_type, {})
        out[ec.column] = out[ec.column].astype(str).map(lambda v: m.get(v, v))
    if ds.shuffle:
        out = out.sample(frac=1.0, random_state=ds.shuffle_seed).reset_index(drop=True)
    return out


# ----------------------------- rendering ------------------------------- #

def render_table(df: pd.DataFrame, fmt: str) -> str:
    if fmt == "markdown":
        # Keep it lightweight — don't require the `tabulate` package
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, r in df.iterrows():
            rows.append("| " + " | ".join(_fmt_cell(r[c]) for c in cols) + " |")
        return "\n".join([header, sep] + rows)
    elif fmt == "fixed":
        widths = {c: max(len(str(c)), df[c].astype(str).map(len).max() if len(df) else 0) for c in df.columns}
        header = "  ".join(f"{c:<{widths[c]}}" for c in df.columns)
        sep = "-" * len(header)
        rows = [header, sep]
        for _, r in df.iterrows():
            rows.append("  ".join(f"{_fmt_cell(r[c]):<{widths[c]}}" for c in df.columns))
        return "\n".join(rows)
    else:
        raise ValueError(f"unknown table_format: {fmt}")


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def assemble_prompt(cfg: Config, rendered_datasets: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    if cfg.task.system_prompt.strip():
        parts.append(cfg.task.system_prompt.strip())
        parts.append("")
    if cfg.task.instructions.strip():
        parts.append(cfg.task.instructions.strip())
        parts.append("")
    for label, table in rendered_datasets:
        if label:
            parts.append(f"## {label}")
            parts.append("")
        parts.append(table)
        parts.append("")
    if cfg.task.output_format.strip():
        parts.append("## Response format")
        parts.append("")
        parts.append(cfg.task.output_format.strip())
    return "\n".join(parts).rstrip() + "\n"


# ----------------------------- driver --------------------------------- #

def run(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    out_dir = cfg.source_path / cfg.output.dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {ds.name: _load_dataset(ds, cfg.source_path) for ds in cfg.datasets}
    mapping = build_mapping_table(cfg, frames)

    results: dict[str, Any] = {"blinded_prompt": None, "unblinded_prompt": None, "mapping": None}

    if cfg.output.produce_blinded:
        blinded_tables = []
        for ds in cfg.datasets:
            df = apply_blinding(frames[ds.name], ds, mapping, blinded=True)
            label = ds.label or ds.name
            blinded_tables.append((label, render_table(df, cfg.output.table_format)))
        prompt = assemble_prompt(cfg, blinded_tables)
        p = out_dir / "blinded_prompt.txt"
        p.write_text(prompt, encoding="utf-8")
        results["blinded_prompt"] = str(p)

    if cfg.output.produce_unblinded:
        unblinded_tables = []
        for ds in cfg.datasets:
            df = apply_blinding(frames[ds.name], ds, mapping, blinded=False)
            label = ds.label or ds.name
            unblinded_tables.append((label, render_table(df, cfg.output.table_format)))
        prompt = assemble_prompt(cfg, unblinded_tables)
        p = out_dir / "unblinded_prompt.txt"
        p.write_text(prompt, encoding="utf-8")
        results["unblinded_prompt"] = str(p)

    # Persist mapping in a single file with both forward and reverse directions
    mapping_out = {
        "forward": mapping,  # {entity_type: {real: anon}}
        "reverse": {         # {entity_type: {anon: real}}
            et: {anon: real for real, anon in m.items()} for et, m in mapping.items()
        },
        "config_source": str(Path(config_path).resolve()),
    }
    mp = out_dir / cfg.output.mapping_file
    mp.write_text(json.dumps(mapping_out, indent=2), encoding="utf-8")
    results["mapping"] = str(mp)

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Epistemic blinding engine")
    ap.add_argument("config", help="Path to YAML config file")
    args = ap.parse_args()
    r = run(args.config)
    print("Wrote:")
    for k, v in r.items():
        if v:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
