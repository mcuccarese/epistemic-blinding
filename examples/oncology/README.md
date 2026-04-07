# Epistemic Blinding Experiment

## Question
Does knowing gene names change an LLM's drug target predictions from the same data?

## Design
**A/B test** with matched prompts across 4 cancer indications:

- **Condition A (Blinded):** 100 genes labeled Gene_001 through Gene_100 with numerical features only
- **Condition B (Unblinded):** Same 100 genes with real symbols (KRAS, TP53, etc.) and identical features

Each condition is presented to a **fresh LLM session with no prior context**.
The LLM ranks its top 20 most promising drug targets based on the features.

## Predictions (what contamination would look like)
If the LLM's training knowledge contaminates its reasoning:

1. **Fame bias:** Well-known genes (TP53, KRAS, EGFR) rank higher in unblinded even when their features don't warrant it
2. **Obscurity penalty:** Genes with strong features but low name recognition rank lower in unblinded
3. **Set divergence:** The two top-20 lists will differ substantially (< 75% overlap)
4. **False knowledge:** The LLM may justify unblinded picks with reasoning from its training data rather than from the provided features

## How to run

### Step 1: Generate prompts
```bash
conda run -n truth-finder python prepare_experiment.py
```

### Step 2: Run blinded condition
1. Open a **fresh** Claude/GPT session (no prior conversation)
2. Paste each prompt from `prompts/blinded/` one at a time
3. Save the response to `blinded_responses/<indication>.txt`

### Step 3: Run unblinded condition
1. Open **another fresh** session (must not see blinded results)
2. Paste each prompt from `prompts/unblinded/`
3. Save to `unblinded_responses/<indication>.txt`

### Step 4: Compare
```bash
conda run -n truth-finder python compare_results.py
```

## Key controls
- **Same features:** Both conditions see identical numerical data
- **Same prompt structure:** Only gene names differ
- **Shuffled order:** Genes are randomly ordered so position doesn't signal importance
- **Fresh sessions:** Each condition uses a separate session to prevent cross-contamination
- **No project context:** Neither session knows about our convergence model or prior results

## Output metrics
- Top-20 set overlap between conditions
- Famous gene count in each top-20
- Approved target recovery rate per condition
- Specific genes that appear only when names are visible (fame bias evidence)
- Specific genes that appear only when blinded (obscurity signal evidence)
