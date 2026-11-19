# Amendment A1: uncertainty intervals for the headline deltas

Post-registration analysis, registered in DECISIONS.md D6
on 2026-11-17 before anything ran. It changes no frozen
verdict: H1 and H2 remain INCONCLUSIVE, H3 remains
CONFIRMED, exactly as scripts/make_results.py computed
them. Machine-readable output: results/amendment_a1_bootstrap.json.
Script: scripts/amendment_a1_bootstrap.py.

## Method (part a, completed): paired bootstrap on the locked test set

The three final models (control-champion, xgboost-tuned,
lightgbm-tuned) were refit on the frozen training split
with the exact final configurations recorded in
results/benchmark_metrics.json, built through
benchmark.make_model, and scored once on the frozen
9,000-row test split. Refit check first: the recomputed
unrounded AUC, KS, and Brier of all three arms match the
recorded values, with a maximum absolute difference of
5.0e-7 across all nine figures (the recorded file stores
6 decimals, so this is its rounding, not drift). Then
5,000 paired resamples of the test rows (sampling with
replacement, numpy RandomState seed 20261117) gave
percentile 95% intervals for the AUC and KS deltas of
each tuned challenger against the champion, computed on
unrounded values.

One convention note: the frozen headline deltas (+0.0012
AUC, +0.0043 KS) come from the preregistration's
rounded-value arithmetic. The unrounded point estimates
below are +0.001136 and +0.004252. Same deltas, two
precisions; the amendment registered the unrounded
computation in advance.

## Intervals

| Delta | Point estimate | 95% CI | Includes zero | Resamples at or below zero |
| --- | --- | --- | --- | --- |
| XGBoost AUC delta (H1 headline) | +0.001136 | [-0.000611, +0.002912] | yes | 10.26% |
| LightGBM KS delta (H2 headline) | +0.004252 | [-0.002503, +0.012353] | yes | 10.58% |
| XGBoost KS delta | +0.000029 | [-0.006821, +0.007498] | yes | 45.30% |
| LightGBM AUC delta | +0.000605 | [-0.001769, +0.002982] | yes | 30.66% |

Every interval includes zero. On the frozen test split,
neither headline beat is distinguishable from no
difference at the 95% level, which is the interval form
of the frozen INCONCLUSIVE verdicts: the deltas are
positive point estimates whose sampling uncertainty
comfortably covers zero. The bootstrap standard
deviations are 0.000905 (XGBoost AUC delta) and 0.003806
(LightGBM KS delta), so both deltas are also smaller
than one and a quarter of their own standard errors,
far below the frozen margins of +0.005 and +0.010.

## Part (b), repeated stratified splits: not completed on this machine

The registered second part (10 stratified 70/30 splits,
seeds 1001 to 1010, each re-running the frozen 24-trial
tuning protocol for both challengers via
benchmark.tune_arm) was started and stopped without a
single split completing, for a measured reason. The
original study's own record puts the two tuning runs at
260.7 s and 349.3 s of wall time on the 4-vCPU machine of
record. This run executed on a VM throttled to about 1.5
shared CPUs: the worker held roughly 82% of one effective
core, and after 15 minutes (12.5 CPU-minutes consumed)
the first split's first tuning arm had still not finished
its 72 fits. At that rate one split costs roughly 45
minutes and the ten splits roughly 7 hours, which is not
a reasonable run for this analysis. No split completed,
so results/amendment_a1_bootstrap.json carries no
repeated-splits section and no split figure is reported
anywhere. The script skips split seeds already recorded
in the JSON, so the run can be resumed on a machine with
the machine-of-record's CPU headroom.

## Tests

The repository's test suite (pytest, scoped to tests/ by
pytest.ini) passes after the amendment: 7 passed.
