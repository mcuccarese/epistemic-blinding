# Social Media Posts — Epistemic Blinding Launch

---

## LinkedIn

**Title suggestion:** (LinkedIn doesn't have titles, but this is the hook line)

---

I built a tool to catch LLMs cheating on data analysis.

When you ask an LLM to rank entities from a feature table — drug targets, stocks, job candidates — it blends two things: what your data says and what it memorized during training. The blend is invisible. There's no way to tell, from a single output, how much came from your numbers and how much came from the model's memory.

Here's a concrete example. I asked Claude to rank 100 genes as drug targets for colorectal cancer, using identical quantitative features. When gene names were visible, KRAS was ranked #1 with the justification "proven therapeutic tractability via covalent RAS inhibitors." When I replaced names with anonymous codes, the same gene dropped to #5. That phrase isn't in the data. It's parametric knowledge, injected into what was supposed to be data-driven analysis.

This isn't a model bug — it's an experimental design problem. The model was asked to reason from data but simultaneously had access to its training priors about the entities. The two signals are entangled.

I built epistemic blinding to fix this. It's dead simple: replace entity names with anonymous codes before prompting, run the analysis, then compare against an unblinded control. The delta tells you exactly how much influence training priors had on the output.

Key findings:
- In oncology target prioritization (4 cancer types): 16% of top-20 predictions change when names are revealed. Famous genes get promoted, obscure genes with strong data profiles get demoted. Validated target recovery is identical either way.
- In S&P 500 value screening (5 random seeds): 35% of top-20 rankings change when tickers are revealed. The bias is systematic across seeds — same tickers promoted/demoted every time.

The important thing: I'm not claiming blinded results are better. Training priors are sometimes genuinely helpful. The point is that without blinding, you have no way to know to what degree the agent is following the analytical process you designed.

This started as a practical need — I was building an agentic system for multi-dataset biological reasoning and needed to verify the agent was actually reasoning from the data I gave it. It turned out to be a general problem across any domain where LLMs reason over named entities.

The tool is open-source (works with any LLM), and a preprint describing the full experiments is available.

GitHub: https://github.com/mcuccarese/epistemic-blinding
Preprint: [link when on arXiv]

#AI #LLM #MachineLearning #DrugDiscovery #AIEvaluation

---

## X (Twitter) — Thread

---

**Tweet 1 (hook):**
I asked an LLM to rank drug targets for colorectal cancer using the same data twice — once with gene names, once with anonymous codes.

KRAS went from #1 to #5. The model's justification cited "proven therapeutic tractability via covalent RAS inhibitors." That phrase isn't in the data. It's training memory.

**Tweet 2 (the problem):**
This isn't a bug. When you ask an LLM to reason over named entities, it silently blends your data with everything it memorized about those names.

Famous entities get promoted. Obscure entities with strong data get demoted. And there's no way to tell from a single output.

**Tweet 3 (the fix):**
Epistemic blinding: replace entity names with anonymous codes before prompting. Run the analysis. Compare against an unblinded control.

The delta tells you how much came from your data vs. the model's memory. It's string replacement — no model modification, works with any LLM.

**Tweet 4 (results):**
Tested across two domains:

Oncology (4 cancer types): 16% of top-20 predictions change. PTEN jumps from #15 to #3 when the model knows its name.

S&P 500 (5 random seeds): 35% of top-20 change. Same tickers promoted/demoted every seed — systematic, not noise.

**Tweet 5 (the thesis):**
I'm NOT claiming blinded results are better. Training priors are sometimes genuinely helpful.

The point: when we write traditional code, every step is auditable. LLMs break that. Epistemic blinding brings back one axis of traceability — did the output come from my data, or from somewhere else?

**Tweet 6 (CTA):**
Open-source tool + preprint:

GitHub: https://github.com/mcuccarese/epistemic-blinding
Paper: [arXiv link]

Built this while developing agentic systems for drug target discovery. Turns out the problem generalizes to any domain where LLMs reason over named entities.
