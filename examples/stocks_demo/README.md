# Stocks demo — does knowing the ticker change the ranking?

A minimal A/B test for the epistemic-blinding skill. The question:

> Given the same table of fundamentals, does an LLM rank companies
> differently when it knows their tickers vs. when it doesn't?

## The setup

`data.csv` contains 20 companies with six synthetic fundamental metrics.
Half the tickers are real large-caps the LLM will recognize from
training (AAPL, TSLA, NVDA, MSFT, META, AMZN, GOOGL, JPM, BRK, V). The
other half are fictional (NVTX, BRDA, QLMX, SVKA, HLRA, GRTL, KPMN,
OXIR, TRVL, ZNPA) and have **no** training-data footprint.

> **Important:** these fundamentals are synthetic and designed for a
> demonstration. They do not represent actual financial data for any
> real company and should not be used for any investment decision.

The numbers are rigged so that the fictional tickers look systematically
cheaper on P/E, higher on FCF yield, and less leveraged than the famous
names — while the famous names still have better margins and occasional
standout growth (NVDA, for example, has the best operating margin in the
set). A value-oriented ranker should favor the fictional names; a
parametric-knowledge-contaminated ranker should boost the famous ones
even when their numbers don't justify it.

## Running the demo

From the repository root:

```bash
# 1. Produce the blinded and unblinded prompts
python scripts/blind.py examples/stocks_demo/config.yaml

# This writes:
#   examples/stocks_demo/output/blinded_prompt.txt
#   examples/stocks_demo/output/unblinded_prompt.txt
#   examples/stocks_demo/output/mapping.json
```

Next, run **each** prompt through an LLM in a **fresh session** (do not
use the same conversation — the model will carry context from the
blinded run into the unblinded run and contaminate the A/B). Save the
responses as:

- `examples/stocks_demo/output/blinded_response.txt`
- `examples/stocks_demo/output/unblinded_response.txt`

Then compute the delta:

```bash
# 2. Resolve anonymous codes in the blinded response
python scripts/deblind.py \
    --mapping examples/stocks_demo/output/mapping.json \
    --input examples/stocks_demo/output/blinded_response.txt \
    --output examples/stocks_demo/output/resolved_response.txt

# 3. A/B comparison
python scripts/compare.py \
    --mapping examples/stocks_demo/output/mapping.json \
    --blinded examples/stocks_demo/output/blinded_response.txt \
    --unblinded examples/stocks_demo/output/unblinded_response.txt \
    --top-k 10 \
    --famous examples/stocks_demo/famous.txt
```

## What to look for

- **Set overlap** between the blinded and unblinded top-10 lists. If
  blinding has no effect, overlap should be ~10/10. Typical result with
  current-generation LLMs: 5-8/10.
- **Fame bias delta** — how many of the 10 famous tickers appear in each
  list. If the unblinded list has more famous tickers than the blinded
  list, parametric knowledge is leaking into the ranking.
- **Justifications.** In the unblinded response, look for phrases like
  "strong brand," "market leader," "durable moat" — these are training
  priors, not data-driven reasoning. They should be absent from the
  blinded response.

## Adapting the demo

- **Swap in real data.** Replace `data.csv` with your own CSV. Update
  the feature descriptions in `config.yaml` to match your columns. The
  `entity_columns` list tells the engine which column to blind.
- **Change the task.** Edit `task.instructions` to ask for a different
  kind of ranking (growth, quality, ESG, whatever).
- **Try a different top-K.** Edit the `--top-k` flag on `compare.py`.
- **Multi-dataset.** Add a second entry under `datasets:` with the same
  `entity_type: company` — both tables will share one consistent
  blinding mapping so cross-dataset reasoning still works.
