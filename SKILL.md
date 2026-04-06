---
name: epistemic-blinding
description: Use this skill when the user wants to prevent an LLM's training-data priors from contaminating data-driven analysis over named entities. Typical triggers - "blind this analysis", "I don't want the model to recognize these names", "compare blinded vs unblinded ranking", "parametric contamination", "run an epistemic blinding experiment", or any task where the user is asking an LLM to rank, score, or prioritize named entities (genes, drugs, tickers, companies, cases, papers, authors, people, places) from a feature table and is worried that entity recognition is biasing the result.
version: 0.1.0
license: MIT
---

# Epistemic Blinding

## What this skill does

Epistemic blinding is an inference-time intervention: before asking an LLM to reason over data that contains named entities, replace each entity identifier with an anonymous code (e.g., `TP53` → `Gene_042`, `AAPL` → `Company_017`). The LLM reasons from the *features* of each entity instead of from its *identity*, so outputs reflect the data rather than the model's memorized priors. After the LLM responds, the anonymous codes are resolved back to real names.

This skill wraps that workflow in a config-driven pipeline that supports:

- **One or multiple datasets** in a single analysis
- **Consistent cross-dataset blinding** — the same real entity gets the same anonymous code everywhere it appears, so multi-source reasoning still works
- **Per-column control** — blind some entity columns, leave others visible (e.g., blind gene names but keep tissue types visible as anchoring context)
- **Preservation lists** — keep specific named values un-blinded as controls or anchors
- **A/B comparison** — produce a matched unblinded prompt and quantify how much the ranking shifts

## When to use this skill

Activate whenever the user is asking an LLM to do any of:

- Rank or prioritize items from a feature table where the items have names (drug targets, stocks, legal cases, research papers, candidate catalysts, etc.)
- Score, grade, or evaluate entities where training-data familiarity could bias the result
- Compare ranking outputs with and without entity identities (A/B contamination experiment)
- Harmonize multiple data sources where entity recognition could override numerical evidence

Also activate if the user mentions "blind the LLM," "parametric contamination," "my model keeps boosting famous X," "we need to show the model can't just use its priors," or asks for the epistemic-blinding skill by name.

## Workflow

1. **Understand the task.** Ask the user (or infer from context) three things:
   - What is the *data source(s)* — file path(s), format, relevant columns
   - What are the *entity columns* — the column(s) whose values should be blinded
   - What is the *analysis task* — what the LLM is supposed to do with the data

2. **Write a config.yaml** following the schema below. Put it next to the data, or in a dedicated working directory. See `examples/stocks_demo/config.yaml` for a full reference.

3. **Run the engine:**
   ```bash
   python scripts/blind.py path/to/config.yaml
   ```
   This writes `blinded_prompt.txt`, `unblinded_prompt.txt` (if enabled), and `mapping.json` to the output directory.

4. **Send the blinded prompt to an LLM.** The skill is model-agnostic. The user can paste `blinded_prompt.txt` into Claude, GPT, Gemini, a local model, or an API call. Save the response as `blinded_response.txt`.

5. **De-blind the response:**
   ```bash
   python scripts/deblind.py --mapping output/mapping.json --input blinded_response.txt --output resolved_response.txt
   ```
   Anonymous codes get rewritten back to real entity names.

6. **Optional A/B comparison.** If `produce_unblinded: true` is in the config, the user also runs the unblinded prompt through the *same model in a fresh session* and saves `unblinded_response.txt`. Then:
   ```bash
   python scripts/compare.py --mapping output/mapping.json \
       --blinded blinded_response.txt \
       --unblinded unblinded_response.txt \
       --top-k 20 \
       [--famous famous_entities.txt]
   ```
   This emits overlap, Jaccard, mean rank delta, Kendall tau, promoted/demoted entities, and optional fame-bias stats.

**CRITICAL: Fresh sessions for A/B.** The two prompts must be run in *independent* LLM sessions with no shared context. If the model sees the blinded version first, its internal mapping will contaminate the unblinded run. Instruct the user to use separate conversations, separate API calls, or session IDs.

## Config schema

```yaml
task:
  system_prompt: |
    <role definition — tell the model what kind of analyst it is>
  instructions: |
    <what to do with the data; reference columns by their feature names>
  output_format: |
    <how the model should structure its response — use anonymous-code-style examples>

datasets:
  - name: primary                # arbitrary label used internally
    path: data.csv               # relative to config.yaml, or absolute
    label: "Primary data"        # human-readable heading above the table in the prompt
    columns: [ticker, pe, growth, margin, debt]   # optional column whitelist; omit for all
    entity_columns:
      - column: ticker           # column whose values are blinded
        entity_type: company     # shared key — same entity_type = same mapping across datasets
        prefix: Company          # anonymous code prefix → Company_001, Company_002, ...
        preserve: []             # values to leave un-blinded (optional; acts as anchors)
        blind: true              # set false to pass-through (useful for multi-stage pipelines)
    shuffle: true                # shuffle row order to prevent positional leakage
    shuffle_seed: 42

  # Optional: a second dataset covering the same entities
  - name: analyst_coverage
    path: coverage.csv
    entity_columns:
      - column: stock_symbol
        entity_type: company     # SAME entity_type as above → consistent mapping
        prefix: Company          # prefix is taken from the first declaration
    shuffle: true

output:
  dir: output                    # where to write prompts and mapping.json
  produce_blinded: true
  produce_unblinded: true        # A/B control
  mapping_file: mapping.json
  table_format: markdown         # 'markdown' or 'fixed'
```

## Multi-dataset semantics

The `entity_type` field is the glue for multi-source reasoning. If dataset A has a column `ticker` and dataset B has a column `stock_symbol`, and both declare `entity_type: company`, then:

- A single shared mapping is built from the union of values across both datasets
- Every occurrence of `AAPL` in either dataset becomes the *same* anonymous code
- The LLM can cross-reference entities between tables even though it doesn't know their names

Use different `entity_type` values for different kinds of entities in the same analysis (e.g., `company` for tickers, `sector` for industry labels, `analyst` for coverage authors). Only blind what could contaminate the reasoning — preserve units, feature column names, dates, and anything else that carries information the model legitimately needs.

## Partial blinding (advanced)

Sometimes the goal is not "blind everything" but "blind the things with strong priors, keep the things that anchor the task." Examples:

- Blind gene names but keep disease names visible (the user wants the model to know *what disease* to reason about, but not get distracted by famous genes)
- Blind company tickers but keep sector labels visible
- Blind paper titles and authors but keep the venue and year

To do this, declare one entity column as `blind: true` and another as `blind: false`, or simply omit the entity column spec entirely for columns the user wants to preserve.

## When NOT to blind

Epistemic blinding is overkill or counterproductive when:

- The task *requires* the model to use its training knowledge (e.g., "what does TP53 do?")
- Entity names encode information that isn't in the feature table (e.g., drug suffixes indicating mechanism — `-mab` for monoclonal antibodies)
- The dataset is so small that blinding every value leaves nothing meaningful to reason over
- The user wants a literature review, not a data-driven analysis

If the user is asking a knowledge-retrieval question, suggest they skip blinding entirely. If they're asking a data-driven-ranking question, blinding is almost always worth running at least as an A/B control.

## Example invocations

**Minimal single dataset:**
> "I have a CSV of 200 companies with fundamentals and I want Claude to rank the top 20 undervalued ones. Make sure the model can't cheat by recognizing famous tickers."

Action: write config pointing at the CSV, `entity_columns: [{column: ticker, entity_type: company, prefix: Company}]`, enable unblinded A/B, run, show delta.

**Multi-dataset:**
> "I have patient mutation data in one table and knockout screen data in another, both indexed by gene symbol. I want the LLM to identify targets where both sources agree, but without its training biases about well-known oncogenes."

Action: two datasets in config, both declaring `entity_type: gene`, shared mapping ensures the same gene gets the same code in both tables.

**Partial blinding:**
> "Blind the gene names but keep the disease names visible — I want the model to know we're looking at breast cancer but not get distracted by BRCA1."

Action: entity column for genes with `blind: true`; disease column not listed (so it passes through as a feature).

## Dependencies

- Python 3.10+
- `pandas`
- `pyyaml`

Install: `pip install pandas pyyaml` (or `uv pip install pandas pyyaml`).

## Reference files

- `scripts/blind.py` — engine
- `scripts/deblind.py` — reverse lookup
- `scripts/compare.py` — A/B metrics
- `references/method.md` — technical description of the blinding method and its limits
- `references/when_to_apply.md` — decision criteria for when blinding pays off
- `examples/stocks_demo/` — minimal runnable example
