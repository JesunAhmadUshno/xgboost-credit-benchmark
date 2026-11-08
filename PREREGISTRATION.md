# Pre-registration: XGBoost and LightGBM against the sklearn champion under one leakage-safe credit protocol

Locked before any measured run exists. The commit that adds
this file precedes every result in this repository; git
history proves the order. Any change to this file after
locking is an amendment and must be recorded in DECISIONS.md
with its reason, before the runs it affects.

## Research question

Under one identical leakage-safe credit protocol, do XGBoost
and LightGBM actually beat the sklearn gradient-boosting
champion they would replace, on discrimination and on
calibration, when every arm gets the same data, the same
splits, and the same tuning budget?

## Machine of record

All runs happen on one named machine: the Muse VM, hostname
htch-runtime (Linux 7.0.0-43-generic, AMD EPYC 9D25,
4 vCPUs, 7.7 GiB RAM), Python 3.12.3. No number from any
other machine is reported in this study. Exact library
versions are recorded in the run manifest
(results/run_manifest.json).

## Subject and data (frozen)

- Subject repository: JesunAhmadUshno/credit-risk-lab,
  pinned at commit fee580d0beee43c76dc57c6c366c211533054956
  (the commit its own results record, and the pin the
  sibling study mlops-evidence-lab verified on 2026-11-08).
  The subject is cloned read-only by
  scripts/fetch_credit_data.sh and is never modified.
- Data file: data/default of credit card clients.xls from
  that commit (UCI default of credit card clients,
  30,000 accounts, default rate 0.2212), SHA-256
  30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933,
  verified by the fetch script before any run uses it.
- Published champion figures under test (from the subject
  repo's results/results.json at the pinned commit):
  gradient boosting AUC 0.7810, KS 0.4196, Brier 0.1355.
  The logistic baseline (AUC 0.7518) is context only; it is
  not an arm of this study.

## The protocol (frozen)

The benchmark re-implements the subject's training protocol
exactly as its src/credit_risk_lab.py specifies:

- Load with pandas read_excel, header=1; rename the target
  to default; EDUCATION values 0, 5, 6 mapped to 4;
  MARRIAGE 0 mapped to 3.
- Features, exactly as the subject: engineered utilization,
  pay_ratio (filled 0, clipped to [-1, 5]), delinq_mean,
  delinq_max, bill_trend (dpd_months is computed by the
  subject for staging only and is not a model input);
  model inputs are LIMIT_BAL, AGE, BILL_AMT1, PAY_AMT1,
  utilization, pay_ratio, delinq_mean, delinq_max,
  bill_trend, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
  (numeric) and SEX, EDUCATION, MARRIAGE (categorical).
- Split: train_test_split, test_size 0.30, stratify on the
  target, random_state 42. Training rows 21,000, test rows
  9,000.
- Preprocessing: the subject's ColumnTransformer (median
  imputation plus standard scaling for numerics, one-hot
  encoding with handle_unknown=ignore for categoricals),
  fit on the training split only. The fitted transformer is
  applied once and every arm receives the identical dense
  float64 matrix. This factors the model step out of the
  subject's pipeline so that all arms see bit-identical
  inputs; the reproduction gate below is the test that the
  factoring changed nothing for the control.
- No statistic is fit on test rows. That is the subject's
  leakage-safe discipline, and this study inherits it
  unchanged.

## Metrics and timing (frozen)

- Per arm, on the frozen test split: AUC
  (roc_auc_score), KS (the subject's ks_stat), Brier
  (brier_score_loss), and Gini (2 x AUC - 1) as a courtesy
  column. Metrics are reported rounded to 4 decimals, the
  subject's convention, and verdict arithmetic uses the
  rounded values.
- Timing: fit time is the wall time of the final full-train
  fit; score time is the wall time of predict_proba on the
  9,000 test rows. Each is measured over 3 repetitions of
  the frozen final configuration and the median is
  reported, in seconds to 3 decimals. Timing excludes
  preprocessing (identical for all arms) and excludes
  tuning; tuning wall time is reported separately per
  tuned arm.
- Threads: the runner sets OMP_NUM_THREADS=4; XGBoost runs
  with n_jobs=4 and LightGBM with num_threads=4. The
  machine has 4 vCPUs.

## Reproduction gate (frozen)

The control arm is the subject's published champion
configuration: HistGradientBoostingClassifier
(max_iter 300, learning_rate 0.06, max_depth 3,
random_state 42) on the preprocessed matrix.

If the control's test AUC differs from the published
0.7810 by more than 0.005 in absolute value, the harness
is declared broken: no challenger verdicts are issued,
RESULTS.md reports the harness failure with the AUC, KS,
and Brier deltas, and the run stops for repair. KS and
Brier deltas are always reported beside the gate outcome
but do not decide it. A benchmark whose control is off
measures the harness, not the libraries.

## Arms (frozen)

1. control-champion: the published configuration above.
2. xgboost-default: XGBClassifier with library defaults
   except random_state 42, eval_metric logloss,
   tree_method hist, n_jobs 4.
3. lightgbm-default: LGBMClassifier with library defaults
   except random_state 42, num_threads 4, verbose -1.
4. xgboost-tuned: random search, budget below.
5. lightgbm-tuned: random search, budget below.
6. sklearn-tuned: HistGradientBoostingClassifier under
   the same random-search budget. This is a reference arm,
   not a hypothesis subject: it separates "the library is
   better" from "tuning is better", because the published
   champion configuration was never given this budget.

## Tuning protocol (frozen)

- Budget: exactly 24 random-search trials per tuned arm.
  No arm receives more trials than another, and no arm is
  re-tuned after any test result is seen. The test split
  is scored exactly once per final model.
- Validation: stratified 3-fold cross-validation on the
  21,000 training rows only (StratifiedKFold, 3 splits,
  shuffle True, random_state 42). Selection metric: mean
  CV AUC. Ties are broken by earlier trial order. The
  winner is refit on the full training split.
- Sampling: numpy RandomState draws from the grids below,
  parameters drawn in the order listed. Arm streams are
  seeded 20261108 + arm index (sklearn-tuned 0,
  xgboost-tuned 1, lightgbm-tuned 2), so each arm's draws
  are independent and reproducible.
- Grids:
  - sklearn-tuned (HistGradientBoostingClassifier,
    random_state 42 fixed, max_depth unset):
    max_iter in {200, 300, 500, 800};
    learning_rate in {0.02, 0.04, 0.06, 0.10};
    max_leaf_nodes in {15, 31, 63};
    min_samples_leaf in {20, 40, 80};
    l2_regularization in {0.0, 0.1, 1.0}.
  - xgboost-tuned (XGBClassifier, random_state 42,
    eval_metric logloss, tree_method hist, n_jobs 4
    fixed): n_estimators in {200, 300, 500, 800};
    learning_rate in {0.02, 0.04, 0.06, 0.10};
    max_depth in {3, 4, 5, 6};
    min_child_weight in {1, 5, 10};
    subsample in {0.8, 1.0};
    colsample_bytree in {0.8, 1.0};
    reg_lambda in {0.1, 1.0, 5.0}.
  - lightgbm-tuned (LGBMClassifier, random_state 42,
    num_threads 4, verbose -1 fixed):
    n_estimators in {200, 300, 500, 800};
    learning_rate in {0.02, 0.04, 0.06, 0.10};
    num_leaves in {15, 31, 63};
    min_child_samples in {20, 40, 80};
    subsample in {0.8, 1.0} with subsample_freq 1;
    colsample_bytree in {0.8, 1.0};
    reg_lambda in {0.1, 1.0, 5.0}.
- The grids are library-native, not identical in
  dimension (5 for sklearn, 7 for the other two). The
  budget that is held equal is the trial count, the
  validation scheme, and the selection metric. This
  asymmetry is a stated limitation, recorded here rather
  than discovered later.

## Hypotheses (frozen, with pass lines)

Verdict vocabulary: CONFIRMED, REFUTED, INCONCLUSIVE.
If the reproduction gate fails, H1 to H3 are NOT RUN.

H1 (XGBoost on discrimination): let d be
AUC(xgboost-tuned) minus AUC(control-champion) on the
rounded values.
- CONFIRMED if d is at least +0.005.
- REFUTED if d is at most 0.0000.
- INCONCLUSIVE if d is positive but below +0.005 (a real
  beat smaller than the named margin).

H2 (LightGBM on KS): let d be KS(lightgbm-tuned) minus
KS(control-champion) on the rounded values.
- CONFIRMED if d is at least +0.010.
- REFUTED if d is at most 0.0000.
- INCONCLUSIVE if d is positive but below +0.010.

H3 (calibration ranking differs from discrimination
ranking): score the three arms control-champion,
xgboost-tuned, lightgbm-tuned. Ordering A ranks them by
AUC descending; ordering B ranks them by Brier
ascending; both use unrounded values.
- CONFIRMED if ordering A and ordering B are not
  identical (at least one arm changes position).
- REFUTED if the two orderings are identical.
- INCONCLUSIVE only if an exact tie in the unrounded
  values makes either ordering ambiguous.

All arms' AUC, KS, and Brier are published beside the
verdicts, whichever way they fall. A win on one metric
and a loss on another is the expected honest shape of
this table, not a problem to be edited away.

## Results generation (frozen)

RESULTS.md is generated by scripts/make_results.py from
the recorded run artifacts (results/benchmark_metrics.json
and results/verdicts.json); verdicts are computed by that
script from the frozen pass lines above, never typed by
hand. With no runs, the generator produces an
awaiting-runs RESULTS.md that cannot be mistaken for
results. README headline numbers are copied from the
generated RESULTS.md, and AUDIT.md records the Jev
claim_check of the headline verdicts with model, noul,
and route.
