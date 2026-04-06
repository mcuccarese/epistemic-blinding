# Epistemic Blinding

**Prevent an LLM's training-data priors from contaminating data-driven analysis.**

When you ask a language model to rank, score, or prioritize named entities from a feature table — drug targets, stocks, legal cases, research papers, candidate catalysts, job applicants — the model's output is a mixture of two signals:

1. What the data actually says
2. What the model learned about those entities during pretraining

If the entities have uneven representation in the training corpus (and they almost always do), the second signal will systematically bias the first. Famous entities get promoted. Obscure entities get demoted. The model's justifications will cite "proven clinical utility" or "market-leading brand" — facts from its training, not your data.

**Epistemic blinding** is a one-step fix: replace entity identifiers with anonymous codes before sending the data to the model. The model reasons from the features alone. After it responds, you resolve the codes back to real names.

This repository is a config-driven skill that implements that workflow.

## Quick start

```bash
pip install pandas pyyaml

# Run the demo — produces blinded + unblinded prompts from a synthetic stocks dataset
python scripts/blind.py examples/stocks_demo/config.yaml

# Send the two prompts to any LLM in separate fresh sessions, save responses,
# then compute the A/B delta:
python scripts/compare.py \
    --mapping examples/stocks_demo/output/mapping.json \
    --blinded examples/stocks_demo/output/blinded_response.txt \
    --unblinded examples/stocks_demo/output/unblinded_response.txt \
    --top-k 10 \
    --famous examples/stocks_demo/famous.txt
```

See [examples/stocks_demo/README.md](examples/stocks_demo/README.md) for the full walkthrough.

## What this supports

- **Single or multi-dataset analyses.** The same entity in two tables gets the same anonymous code, so cross-dataset reasoning still works.
- **Per-column blinding control.** Blind gene names, keep disease names visible. Blind tickers, keep sectors visible.
- **Preservation lists.** Anchor specific entities un-blinded as controls.
- **Row shuffling.** Prevents positional leakage.
- **A/B comparison.** Produce a matched unblinded prompt and quantify the delta (overlap, rank shift, Kendall tau, fame bias).
- **Model-agnostic.** The skill emits prompts; you run them through any LLM you choose.

## How to use it as a Claude Code skill

Copy or symlink this directory into your Claude Code skills path:

```bash
# macOS / Linux
ln -s "$(pwd)" ~/.claude/skills/epistemic-blinding

# Windows
mklink /D "%USERPROFILE%\.claude\skills\epistemic-blinding" "%CD%"
```

Claude Code will load `SKILL.md` automatically. When you ask a question like "rank these candidates from the CSV but make sure you're not using training priors," Claude will recognize the skill applies and walk you through building the config.

## When to use it

Any time you're asking a model to reason over data containing named entities where training priors could bias the answer. In practice that covers most prioritization and ranking tasks across finance, law, medicine, scientific research, hiring, policy analysis, peer review, and recommendation systems. See [references/when_to_apply.md](references/when_to_apply.md) for the decision criteria.

## Origin

This tool grew out of work on oncology drug target prioritization, where the contamination effect is particularly acute: LLMs have absorbed very strong priors about which genes are "important" in which cancers. In a controlled four-indication experiment, knowing gene names changed ~16% of top-20 predictions and the direction of change was systematic — famous genes promoted, obscure genes demoted, regardless of feature strength. The method generalizes to any domain with the same structure.

A preprint describing the original experiment is in preparation.

## License

MIT. See [LICENSE](LICENSE).
