# Signed MaxSim - testing a theory paper's idea in practice

An independent implementation and test of **Signed MaxSim**
(Killingback, Ingale, Zamani, Musco — *Quantifying and Expanding the
Theoretical Capacity of Late-Interaction Retrieval Models*, arXiv:2607.05803,
July 2026). No official code was public at the time of this project.

*Not affiliated with the paper's authors.*

#### **What the paper is about**

This theory paper does two things:

1. Proves a limit: Standard MaxSim (the scoring method behind ColBERT
   and similar models) can't represent every possible signed similarity
   score, for a fixed embedding size. This is a structural limit, not a
   training problem.
2. Proposes a fix: Signed MaxSim splits each token into a "magnitude"
   part and a "sign" part. Important detail: the sign is applied after
   picking the best-matching token pair, not before,   the matching itself
   still only depends on magnitude, just like standard MaxSim. The sign
   only flips or scales the score of the match that was already chosen.

The paper's own experiment (nDCG@10 jumping from 0.597 to 1.000 on a
negation task) is there to show the theoretical limit has a real
effect.

## What this project adds

The paper doesn't test whether the idea still works with limited data, different base models, 
and controls to make sure
any gain really comes from the sign mechanism and not just from
fine-tuning in general. That's the gap this project tries to measure:

- **4 different base models**, to see if any gain holds across models or
  is a one-off (`colbertv2.0`, `answerai-colbert-small-v1`,
  `GTE-ModernColBERT-v1`, `LateOn`)
- **Ablation table.** 4 conditions per model, to separate the sign mechanism's
  effect from the effect of fine-tuning itself:

| Condition        | Backbone fine-tuned? | Sign active? | What it isolates                          |
|-------------------|:--------------------:|:------------:|--------------------------------------------|
| `baseline`         | No        | -            | Reference point, untouched model            |
| `magnitude_only`   | Yes                   | No           | Standard MaxSim, effect of fine-tuning alone |
| `signed_frozen`    | No (frozen)          | Yes          | Effect of the sign alone |
| `signed`           | Yes                   | Yes          | Full mechanism                              |

- **The standard NevIR benchmark** (pairwise accuracy, random guessing =
  25%) instead of a custom task, less favorable test than the
  paper's own setup
- **A check on general search quality** (nDCG@10 on scifact and
  nfcorpus) to ensure fine tuning doesn't degrade quality


## Setup

- **Models**: LateOn, ColBERTv2, ColBERT-small (Answer.AI), ModernColBERT 
- **Metrics**: NevIR pairwise accuracy (test set, chance = 25%) and nDCG@10
  on BEIR/scifact + BEIR/nfcorpus (exact scan, no approximate index).
- **Training data**: NevIR train (real negation pairs, oversampled) + a
  slice of MS MARCO (general matching, keeps quality from collapsing).
- **No ANN index (FastPLAID/Voyager) support**: those indexes are built
  around standard MaxSim's geometry. Everything here runs as an exact scan,
  baseline included, to keep comparisons fair.

## Layout

```
src/signed_maxsim/
  config.py    # models, variants, hyperparameters
  data.py      # NevIR / MS MARCO / BEIR loading
  model.py     # SignedColBERT, signed_maxsim
  train.py     # step-based training, early stopping
  eval.py      # NevIR pairwise accuracy, BEIR nDCG@10
  io_utils.py  # atomic writes, crash-safe resume
  runner.py    # runs one model end to end
scripts/run_backbone.py   # CLI
notebooks/experiments.ipynb  # tuning sandbox, plots
```

## Run

```bash
uv sync
```
Run each backbone separately (`lateon`, `colbert-v2`, `colbert-small`, `moderncolbert`):

```bash
uv run python scripts/run_backbone.py lateon
```

Results save incrementally and atomically to
`results/signed_maxsim_results.json`

## Results

| Model | Variant | NevIR pairwise acc | scifact nDCG@10 | nfcorpus nDCG@10 |
|---|---|---:|---:|---:|
| LateOn | baseline | 0.306 | 0.763 | 0.382 |
| LateOn | magnitude_only | 0.257 | 0.682 | 0.343 |
| LateOn | signed_frozen | 0.252 | 0.720 | 0.357 |
| LateOn | signed | 0.283 | 0.719 | 0.358 |
| ColBERTv2 | baseline | 0.164 | 0.646 | 0.331 |
| ColBERTv2 | magnitude_only | 0.179 | 0.659 | 0.321 |
| ColBERTv2 | signed_frozen | 0.171 | 0.668 | 0.324 |
| ColBERTv2 | signed | 0.154 | 0.666 | 0.324 |
| ColBERT-small | baseline | 0.217 | 0.746 | 0.370 |
| ColBERT-small | magnitude_only | 0.221 | 0.733 | 0.362 |
| ColBERT-small | signed_frozen | 0.228 | 0.717 | 0.357 |
| ColBERT-small | signed | 0.250 | 0.720 | 0.362 |
| ModernColBERT | baseline | 0.275 | 0.760 | 0.380 |
| ModernColBERT | magnitude_only | 0.253 | 0.741 | 0.369 |
| ModernColBERT | signed_frozen | 0.273 | 0.728 | 0.368 |
| ModernColBERT | signed | 0.271 | 0.731 | 0.367 |

**Sign vs magnitude-only, at equal fine-tuning effort** the comparison
that isolates what the sign actually contributes:

| Model | signed | magnitude_only | Sign helps? |
|---|---:|---:|:---:|
| LateOn | 0.283 | 0.257 | Yes (+2.6 pts) |
| ColBERTv2 | 0.154 | 0.179 | No (-2.5 pts) |
| ColBERT-small | 0.250 | 0.221 | Yes (+3.0 pts) |
| ModernColBERT | 0.271 | 0.253 | Yes (+1.8 pts) |


Adding the sign makes fine-tuning a bit better than magnitude-only, on 3
of 4 models. ColBERTv2 (the oldest model) is the exception. But
every fine-tuned version (sign or not) scores worse on general search
(BEIR) than the untouched model. That cost shows up on every model, every
variant, and isn't solved yet.

## Known limitations

- No ANN index support yet
- Training budget is capped by NevIR train's small size (948 pairs); no
  full hyperparameter search, choices were made once and applied the same
  way to every model.
- Only two BEIR tasks (scifact, nfcorpus) as a safety check (due to computation cost), 
  not a full coverage of general search quality.
- The paper's own experimental setup (their exact "out-of-domain" data and
  protocol) isn't reproduced here, this project tests the mechanism under
  different conditions
