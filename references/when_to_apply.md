# When to apply epistemic blinding

Blinding is worthwhile when **all three** of these conditions are true:

1. **The data actually contains decision-relevant signal.** You're not just decorating a gut call with numbers — the features you're giving the model carry real information that should drive the ranking.

2. **The named entities have uneven representation in the model's training corpus.** Some are famous, some are obscure. Famous entities will activate strong priors; obscure ones will not. The asymmetry is what produces contamination.

3. **Someone will act on the output.** Allocate capital, allocate trial slots, cite a case, fund a grant, schedule an interview. If the ranking has consequences, contamination has consequences.

When all three hold, blinding converts a partially-parametric ranker into a data-driven one and the delta is measurable.

## Domains where this typically pays off

- **Finance.** Equity ranking, portfolio construction, credit analysis. Ticker/brand asymmetry is extreme and stakes are high.
- **Biomedical research.** Drug target prioritization, differential diagnosis, biomarker ranking, drug repurposing. LLMs have absorbed very strong gene/disease/drug priors.
- **Legal.** Case importance, citation recommendation, prior art search. Famous cases and famous inventors dominate.
- **Research evaluation.** Paper ranking, reviewer scoring, grant prioritization, conference program selection. Strong author/institution priors.
- **Hiring and admissions.** Resume screening, candidate ranking. Named universities, named employers, named references all carry priors.
- **Policy and development economics.** Country rankings, intervention evaluation. Western-corpus training skew creates acute bias on non-Western subjects.
- **Materials science and chemistry.** Catalyst selection, candidate theories, named reactions. Citation-count priors are large.
- **Journalism.** Source credibility weighting, fact prioritization, pundit evaluation.

## Domains where blinding is the wrong tool

- **Knowledge retrieval.** "What does TP53 do?" "Summarize the Miranda ruling." Here you *want* training knowledge.
- **Name-encoded information.** IUPAC chemical names, drug suffixes (`-mab`, `-tinib`), catalog numbers that encode structure. Blinding destroys the signal.
- **Tiny datasets.** If you have five entities and they're all famous, blinding them all just makes the task incoherent.
- **Literature review.** The task is to consolidate what has been said about named entities. Blinding defeats the purpose.
- **Recommendation where popularity is a legitimate feature.** "Recommend popular restaurants" is a question where fame is signal, not noise.

## Signs that contamination is happening in your current workflow (without blinding)

- The model's justifications reference facts not present in your data ("proven clinical utility," "strong brand moat," "well-known therapeutic target")
- Famous entities appear at the top of rankings regardless of which features you emphasize
- Rerunning the same query produces the same ranked list of famous entities with different feature sets
- The model is reluctant to rank a famous entity low even when its features are weak
- The model promotes entities you know are wrong for the task, but that match its training associations

If you see these patterns, run an A/B blinded/unblinded comparison. The delta is the contamination.

## How to decide between full blinding and partial blinding

- **Full blinding** (blind everything with a training footprint) is the strongest form. Use it when you want maximal confidence that the output is data-driven.
- **Partial blinding** (blind some entity columns, keep others) is the right choice when some named context is needed to anchor the task. Example: in a disease-specific drug target analysis, blind the gene names but keep the disease name visible so the model knows *what* it's ranking for.
- **A/B blinded + unblinded** is what you want for *measuring* contamination — for making the case in a paper, a blog post, or a regulatory submission.

## How to decide top-K

The sensitivity of the comparison depends on K:

- **Small K (5-10)** — more sensitive to contamination, but more noise-prone on repeat runs
- **Medium K (20)** — a good default for most ranking tasks
- **Large K (50+)** — captures broader shifts in the ranking distribution but dilutes the "top pick" signal

Start at K=20 for the initial A/B. If you want to stress-test, also report K=10.
