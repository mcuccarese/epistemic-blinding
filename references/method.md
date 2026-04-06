# Method: How Epistemic Blinding Works

Epistemic blinding is an inference-time intervention on the prompt. It requires no model modification, no retraining, and no access to model internals. The full procedure is:

1. **Identify entity columns.** For each dataset, decide which columns contain named entities whose identities could leak training-data priors (gene symbols, tickers, drug names, case captions, author names, etc.).

2. **Build a mapping table.** For each distinct entity type, collect every distinct value that appears in any dataset. Shuffle deterministically with a stable seed (so the position of an entity in the code sequence carries no information). Assign anonymous codes of the form `{prefix}_{number}`, zero-padded to a width that fits the entity count.

3. **Apply the mapping.** Replace every occurrence of a real entity value with its anonymous code. The same real value must always map to the same anonymous code — this is what enables cross-dataset reasoning to survive blinding.

4. **Shuffle row order.** Randomize the order of rows in each table so that position in the presented data carries no information about original rank or importance.

5. **Render as prompt.** Format the blinded table(s) into a prompt along with the task instructions, system role, and desired output format. Preserve column headers, units, and feature descriptions — these are legitimate information the model needs.

6. **Send to LLM in a fresh session.** Context from other conversations can leak identity information back in. A/B comparison runs must use independent sessions.

7. **De-anonymize the response.** Parse the model's output and rewrite the anonymous codes back to real entity names.

## What gets blinded and what doesn't

The goal is to blind the **things that carry training priors** and preserve the **things that carry legitimate task information**. In practice:

| Blind                      | Preserve                                  |
|---------------------------|-------------------------------------------|
| Entity identifiers        | Feature column names                      |
| Famous names              | Units (mg/kg, USD, fold-change)           |
| Proper nouns              | Categorical labels (tissue type, sector)  |
| Anything with a Wikipedia article | Numeric values                    |
|                           | Edge/relationship types in graphs         |
|                           | Data source names                         |
|                           | Dates and time windows                    |

The decision is not absolute. Some categorical labels (e.g., disease names in a biology analysis) sit in the middle: the model needs to know what disease is being studied, but famous disease names carry their own priors. The right answer depends on the task. Use `preserve:` in the config to keep specific values un-blinded, or set `blind: false` on an entity column to pass it through while still registering it as an entity type.

## Why cross-dataset consistency matters

If you blind two datasets independently, the same entity may end up with different anonymous codes in each dataset. The model then cannot cross-reference entities across tables, and multi-source reasoning breaks. The engine in this repository uses a shared mapping per `entity_type` — a single sorted-and-shuffled assignment drawn from the union of all values across all datasets that declare that type. This preserves every cross-dataset linkage while removing identity information.

## Limits and failure modes

**Structural leakage.** Some features reveal identity even without the name. A table row with `market_cap: $3.0T, operating_margin: 0.30, product_category: smartphones` is clearly Apple even if the ticker is blinded. When structural features are highly identifying, blinding only reduces rather than eliminates contamination. Mitigations: drop the identifying feature, bucket numeric values into coarser bins, or accept partial blinding.

**Ordering leakage.** If you forget to shuffle, original row order may encode meaningful information (e.g., pre-sorted by importance). The engine shuffles by default.

**Model memory of the code format.** If an LLM has seen many papers that use `Gene_001` style anonymization in examples of *known* analyses, it may attempt to pattern-match. Mitigation: pick a prefix that's not a common convention in your domain, or use a seed that produces different assignments each run.

**Entity count explosion.** With very large entity sets, the codes become long and the table harder to read. This is a usability issue, not a correctness one. Break large sets into smaller chunks and run the analysis per chunk.

**Semantically meaningful names.** Some domains use names that contain information (e.g., IUPAC chemical names encode structure; drug names ending in `-mab` indicate monoclonal antibodies). Blinding destroys that information. If the task requires it, either keep those columns visible or pre-extract the encoded information as separate features.

**Task mismatch.** Blinding is the wrong tool if the user actually wants the model to use its training knowledge — e.g., "tell me what TP53 does" or "summarize the literature on KRAS inhibitors." Blinding is for *data-driven* reasoning, not *knowledge-retrieval* reasoning.

## Relationship to other techniques

- **Clinical trial blinding.** The direct analog: prevent the analyst from accessing information that could bias the analysis. Epistemic blinding is blinding applied to the LLM-as-analyst.
- **Double-blind peer review.** Anonymizes author identities to reduce prestige bias in human review. Epistemic blinding extends this from authors to the objects of reasoning.
- **Counterfactual data augmentation.** Perturbs entity names during training to reduce memorization. Operates at the training-data level; epistemic blinding operates at the inference-time prompt level and requires no retraining.
- **Data contamination detection.** A different problem: detecting whether a benchmark was in the training set. Epistemic blinding addresses general knowledge leaking into reasoning, not test-set memorization inflating metrics.
