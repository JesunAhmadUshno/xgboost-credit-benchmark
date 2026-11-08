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
