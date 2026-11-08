# Audit

Self-audit of xgboost-credit-benchmark against its own
frozen protocol (PREREGISTRATION.md) and the standing
project standard. Written after the runs it grades.
Every number below is in RESULTS.md or in a results/
artifact.

## What the evidence supports

- The reproduction gate passed exactly. The control arm,
  the subject's published champion configuration run
  through this harness, reproduces the published figures
  to the last printed digit: AUC 0.7810, KS 0.4196,
  Brier 0.1355, delta 0.0000 on all three. The factored
  preprocessing (DECISIONS.md D1) changed nothing, and
  the three timing repetitions of every arm returned
  identical metrics, so the comparisons are between
  libraries, not between runs.
- The tuning budget was equal and is fully visible:
  24 trials per tuned arm under one validation scheme,
  every trial published in results/tuning_<arm>.json,
  and the test split scored exactly once per final
  model. No arm was re-tuned after any test result.
- The verdicts are what the frozen lines produce, and
  they are not the verdicts a launch post would choose:
  H1 and H2 are INCONCLUSIVE because both challengers
  beat the champion by less than the margins frozen
  before the runs. The margins were not renegotiated
  afterwards, and the sub-margin deltas (+0.0012 AUC,
  +0.0043 KS) are published beside the verdicts.
- The reference arm does its job: tuned sklearn
  (AUC 0.7812) sits between the champion and the tuned
  challengers, which is the evidence that the gap is a
  tuning gap. Without it, the table would invite a
  library conclusion the data does not support.

## Incidents and corrections, on the record

- The first dependency install failed on a transient
  "no space left on device" error from the package
  installer; disk checks immediately after showed ample
  space on every filesystem, and the retry with the
  download cache disabled succeeded. No run was affected;
  the incident is recorded because the environment log
  should not have a silent gap in it.
- An unscoped local pytest run collided with the
  read-only subject clone's same-named test module
  (data/ref/credit-risk-lab/tests/test_units.py).
  pytest.ini scopes collection to this repo's tests/
  (DECISIONS.md D5). This was a harness fix made before
  any measured run, not a protocol change.
- Wall times were measured on a shared 4-vCPU VM, and it
  shows: LightGBM defaults' three fit repetitions span
  0.616 to 2.306 seconds. Medians are reported as
  registered, but the fit and score times in RESULTS.md
  are indicative magnitudes, not precise measurements.

## Jev claim check

The headline verdicts (exact control reproduction, H1
INCONCLUSIVE at +0.0012 against the +0.005 margin, H2
INCONCLUSIVE at +0.0043 against the +0.010 margin, H3
CONFIRMED with the two rankings, and the reference arm's
figures) were checked with the Jev claim_check gate:
model jev-1.13.0, noul 0.64, route REVIEW (house bands:
clear at 0.85 or above, block at 0.10 or below). Spec and
raw response: results/jev-claim-spec.json and
results/jev-claim-check.json. On a REVIEW route the
figures were hand-verified against the artifacts: the
control, XGBoost, LightGBM, and reference-arm AUC, KS,
and Brier values against results/benchmark_metrics.json;
both deltas and both ranking lists against
results/verdicts.json as computed by
scripts/make_results.py; the published figures against
the subject's results/results.json at the pinned commit.
Every figure in the claim matches its artifact. Jev
gates claims; it authorizes nothing, and nothing here
needed authorizing.

## Grade: 8,100 of 10,000

Deductions, itemized:

- One dataset, one split, one seed. The headline deltas
  (+0.0012, +0.0043) are smaller than split-to-split
  variation would plausibly be, and the study carries no
  repeated-split or bootstrap interval for them. The
  frozen INCONCLUSIVE band is the honest verdict under
  that design, but a stronger design would put an
  interval on the deltas instead of only a band on the
  verdict. Minus 800.
- The search grids are library-native and unequal in
  dimension (5 parameters against 7 and 7), a limitation
  registered in advance: the equal claim is about trial
  count, validation, and selection metric, not about
  search-space coverage. Minus 400.
- Selection used CV AUC alone, while H2 is a KS claim:
  LightGBM was tuned for the metric of H1's family, not
  its own. Frozen and disclosed, but a KS-aware selection
  would have given H2 a fairer test. Minus 300.
- Timing precision on a shared VM (see incidents): the
  timing columns are the weakest numbers in the study.
  Minus 300.
- CI covers one interpreter (3.12) where the house
  pattern is two; the pinned stack cannot install on
  3.11. Minus 100.

What would raise the grade: repeated stratified splits
(or a bootstrap over the frozen split) with intervals on
the two headline deltas, a second selection metric for
the KS hypothesis, and a quiet machine for the timing
table. The protocol's own structure (frozen arms, one
test scoring per model, script-computed verdicts) needs
no change for any of that.
