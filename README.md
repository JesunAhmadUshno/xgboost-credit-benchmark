# xgboost-credit-benchmark

Under one identical leakage-safe credit protocol, do
XGBoost and LightGBM actually beat the sklearn
gradient-boosting champion they would replace, on
discrimination and on calibration, when every arm gets
the same data, the same splits, and the same tuning
budget?

The protocol was frozen in PREREGISTRATION.md (the first
content commit of this repository) before any measured
run. Verdicts are computed from the recorded artifacts by
scripts/make_results.py, never assigned by hand.

## Headline verdicts

- **H1 INCONCLUSIVE.** Tuned XGBoost edges the champion
  on AUC, 0.7822 against 0.7810, a delta of +0.0012, but
  the frozen margin for "beats" was +0.005. The beat is
  real and smaller than the named margin.
- **H2 INCONCLUSIVE.** Tuned LightGBM edges the champion
  on KS, 0.4239 against 0.4196, a delta of +0.0043, but
  the frozen margin was +0.010. Same shape: a beat below
  the line that would have made it a finding.
- **H3 CONFIRMED.** The calibration ranking is not the
  discrimination ranking. By AUC: XGBoost, LightGBM,
  champion. By Brier: LightGBM, XGBoost, champion. The
  AUC winner is not the calibration winner.

The control arm reproduces credit-risk-lab's published
champion exactly (AUC 0.7810, KS 0.4196, Brier 0.1355,
delta 0.0000 on all three), so the table below measures
the libraries, not the harness. Full tables and the
tuning record: RESULTS.md. The honest self-grade, with
itemized deductions: AUDIT.md (8,100 of 10,000).

## The table (frozen test split, 9,000 rows)

| Arm | AUC | KS | Brier |
| --- | --- | --- | --- |
| Control: sklearn champion (published config) | 0.7810 | 0.4196 | 0.1355 |
| XGBoost (library defaults) | 0.7598 | 0.3957 | 0.1420 |
| LightGBM (library defaults) | 0.7769 | 0.4237 | 0.1361 |
| XGBoost (tuned, 24 trials) | 0.7822 | 0.4196 | 0.1351 |
| LightGBM (tuned, 24 trials) | 0.7816 | 0.4239 | 0.1348 |
| sklearn HGB (tuned, 24 trials, reference) | 0.7812 | 0.4218 | 0.1352 |

## How to read it

Three facts carry the study. First, with an equal
24-trial tuning budget, all three libraries land within
0.0010 AUC of each other (0.7812 to 0.7822): on this
protocol the champion's small deficit to its tuned
rivals is a tuning deficit, not a library deficit, and
the tuned sklearn reference arm is what proves it
(DECISIONS.md D2). Second, defaults tell the opposite
story to tuning: XGBoost at library defaults trails the
champion badly (0.7598), while LightGBM at defaults
already beats the champion on KS. "Which library wins"
has no answer until the tuning budget is named. Third,
the AUC winner (XGBoost) is not the calibration winner
(LightGBM, Brier 0.1348): a swap chosen on the ranking
metric alone would have picked the worse-calibrated of
the two challengers.

## Why both metrics are measured

Credit models decide who gets loans, and at what price.
A library swap that lifts a ranking metric while
worsening calibration quietly misprices risk for real
borrowers: the ranking says the order of applicants
improved while the probabilities underneath, the numbers
prices and provisions are set from, drifted further from
the truth. That is why this benchmark freezes a
calibration hypothesis beside its discrimination
hypotheses, and why the honest headline is a split
verdict rather than a winner.

## Protocol in brief

- Subject: JesunAhmadUshno/credit-risk-lab at commit
  fee580d0, cloned read-only; its UCI credit-card data
  file is SHA-256 verified before use. The subject's
  leakage-safe protocol is imported exactly: stratified
  70/30 split, seed 42, preprocessing fit on train only.
- Every arm receives the identical preprocessed matrix
  (the subject's pipeline with the model step factored
  out; DECISIONS.md D1). The test split is scored exactly
  once per final model.
- Tuning fairness: exactly 24 random-search trials per
  tuned arm, stratified 3-fold CV on the training rows
  only, selection by mean CV AUC, grids and margins
  frozen in PREREGISTRATION.md.

## Reproduce

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sh scripts/fetch_credit_data.sh        # verifies the data SHA-256
.venv/bin/python -m pytest -q          # unit tests (synthetic data)
.venv/bin/python src/benchmark.py      # the measured runs
.venv/bin/python scripts/make_results.py   # regenerates RESULTS.md
```

Artifacts: results/benchmark_metrics.json (every arm's
metrics and timings), results/tuning_<arm>.json (every
trial), results/run_manifest.json (machine, versions,
data hash). Design choices and their reasons:
DECISIONS.md.
