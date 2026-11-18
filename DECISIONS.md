# Decisions

## D1: the benchmark re-implements the subject protocol, with the preprocessor factored out

The benchmark re-implements credit-risk-lab's training in
src/credit_common.py from its published source at the
pinned commit, following the line-faithful re-implementation
the sibling study mlops-evidence-lab verified on
2026-11-08. Importing the subject repo as a package would
couple this lab to a repo whose own runs write into its
results directory; the benchmark must not touch the subject.

One structural change is made, and it is frozen in
PREREGISTRATION.md: the subject's Pipeline (preprocessing,
then model) is factored so the fitted ColumnTransformer is
applied once and every arm receives the identical dense
matrix. Without the factoring, "the same features" for
three libraries would be three separate pipeline fits, and
any difference could be preprocessing noise. The control
arm's reproduction gate is the empirical test that the
factoring changed nothing for the champion configuration.

## D2: the sklearn-tuned reference arm

The published champion configuration (300 iterations,
learning rate 0.06, depth 3) was a hand-set configuration,
never the product of a search. Comparing tuned XGBoost and
LightGBM against it alone would confound two claims: that
the library is better, and that tuning is better. A sixth
arm, sklearn-tuned, gives the champion's own library the
same 24-trial budget under the same validation scheme. It
is a reference arm only: no hypothesis is scored on it,
and it appears in every table so the confound stays
visible in print.

## D3: the tuning grids are library-native

The three search grids differ in dimension (5 parameters
for sklearn, 7 each for XGBoost and LightGBM) because the
libraries expose different controls; forcing a common
parameterization would have meant tuning a least common
denominator that none of the libraries would use in
practice. What is held equal is the budget that decides
fairness in this study's question: 24 trials, the same
stratified 3-fold validation on the same training rows,
the same selection metric (mean CV AUC), and one test
scoring per final model. The asymmetry is stated in
PREREGISTRATION.md as a limitation, not discovered after
the fact.

## D4: CI runs Python 3.12 only, and never fetches the corpus

The pinned numerical stack requires Python 3.12 or newer
(numpy 2.5.x declares Requires-Python >= 3.12), so CI runs
the machine-of-record interpreter only, as in the sibling
study. Unit tests run on synthetic frames, so CI needs no
data fetch and the corpus never enters a CI artifact.

## D5: pytest collection is scoped to tests/

The read-only subject clone under data/ref/ carries its
own tests/test_units.py; an unscoped pytest run collides
on the duplicate module name. pytest.ini sets testpaths
to this repo's tests/ and excludes data/, which also keeps
the subject's own test suite out of this repo's CI signal.

## D6 (2026-11-17): Amendment A1, uncertainty intervals for the two headline deltas (post-registration analysis)

AUDIT.md's largest deduction (800 points) is that the two
headline deltas, XGBoost tuned AUC +0.0012 and LightGBM
tuned KS +0.0043 over the champion, rest on the one frozen
split and one seed with no uncertainty interval. This
entry registers, before any of it is run, a
post-registration analysis that adds uncertainty intervals
to those existing deltas. It changes no frozen verdict:
H1 and H2 remain INCONCLUSIVE and H3 remains CONFIRMED,
exactly as scripts/make_results.py computed them from the
frozen pass lines, and no frozen results file (RESULTS.md,
results/benchmark_metrics.json, results/verdicts.json, the
tuning records) is edited or regenerated. The analysis
describes the sampling uncertainty of the recorded
deltas; it issues no new verdict.

What will be computed, by the new script
scripts/amendment_a1_bootstrap.py, with outputs only in
the new files results/amendment_a1_bootstrap.json and
results/amendment_a1_bootstrap.md:

1. Paired bootstrap on the locked test set. The three
   final models (control-champion, xgboost-tuned,
   lightgbm-tuned) are refit on the frozen training split
   with the exact final configurations recorded in
   results/benchmark_metrics.json, built through
   benchmark.make_model, and scored once on the frozen
   9,000-row test split. The refit must first reproduce
   the recorded unrounded metrics in that file; if it
   does not, the analysis stops and reports the mismatch
   instead of computing intervals. Then 5,000 paired
   resamples of the test rows (sampling with replacement,
   numpy RandomState seed 20261117) give percentile 95
   percent intervals for four deltas against the champion,
   computed on unrounded values: the XGBoost AUC delta
   and KS delta, and the LightGBM AUC delta and KS delta.
   The two headline deltas are the XGBoost AUC delta and
   the LightGBM KS delta.
2. Repeated stratified splits. Ten further stratified
   70/30 splits of the full data (split seeds 1001 to
   1010) each re-run the frozen protocol end to end:
   preprocessor fit on that split's training rows only,
   the control at its published configuration, and for
   xgboost-tuned and lightgbm-tuned the frozen 24-trial
   random search exactly as benchmark.tune_arm runs it
   (same grids, same arm streams seeded 20261108 plus arm
   index, same stratified 3-fold CV at random_state 42,
   selection by mean CV AUC, winner refit on the full
   training rows, that split's test rows scored once).
   The sklearn-tuned reference arm is not re-run: no
   headline delta involves it. Per-split trial records
   are written to a scratch directory, not to results/,
   so the frozen tuning files stay untouched. The output
   is the split-to-split distribution (mean, standard
   deviation, median, range, and sign count) of the same
   four deltas.
