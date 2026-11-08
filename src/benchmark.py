"""Benchmark runner: the six frozen arms of PREREGISTRATION.md.

Run order is part of the protocol: the control arm runs
first and the reproduction gate is checked before any
challenger is scored. Writes results/benchmark_metrics.json,
results/tuning_<arm>.json for the tuned arms, and
results/run_manifest.json. Verdicts are not computed here;
scripts/make_results.py computes them from these artifacts
under the frozen pass lines.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import credit_common as cc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PUBLISHED = {"auc": 0.7810, "ks": 0.4196, "brier": 0.1355}
GATE_TOLERANCE = 0.005
N_TRIALS = 24
SEARCH_SEED = 20261108
TIMING_REPS = 3

# Frozen grids, in the frozen draw order (PREREGISTRATION.md).
GRIDS = {
    "sklearn-tuned": {
        "max_iter": [200, 300, 500, 800],
        "learning_rate": [0.02, 0.04, 0.06, 0.10],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [20, 40, 80],
        "l2_regularization": [0.0, 0.1, 1.0],
    },
    "xgboost-tuned": {
        "n_estimators": [200, 300, 500, 800],
        "learning_rate": [0.02, 0.04, 0.06, 0.10],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0],
    },
    "lightgbm-tuned": {
        "n_estimators": [200, 300, 500, 800],
        "learning_rate": [0.02, 0.04, 0.06, 0.10],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [20, 40, 80],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0],
    },
}
ARM_INDEX = {"sklearn-tuned": 0, "xgboost-tuned": 1, "lightgbm-tuned": 2}


def sample_params(rng: np.random.RandomState, grid: dict) -> dict:
    """One random draw: one uniform choice per parameter, in order."""
    return {name: values[int(rng.randint(len(values)))]
            for name, values in grid.items()}


def make_model(arm: str, params: dict):
    if arm.startswith("sklearn") or arm == "control-champion":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=42, **params)
    if arm.startswith("xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=42, eval_metric="logloss",
                             tree_method="hist", n_jobs=4, **params)
    if arm.startswith("lightgbm"):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=42, num_threads=4, verbose=-1,
                              **params)
    raise ValueError(arm)


def timed_fit_score(arm: str, params: dict, Xtr, ytr, Xte, yte) -> dict:
    """Fit and score the frozen final configuration, 3 repetitions.

    Metrics come from repetition 1; all repetitions' metrics
    are recorded so nondeterminism, if any, is visible rather
    than hidden.
    """
    fit_times, score_times, rep_metrics = [], [], []
    for _ in range(TIMING_REPS):
        model = make_model(arm, params)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        t1 = time.perf_counter()
        p = model.predict_proba(Xte)[:, 1]
        t2 = time.perf_counter()
        fit_times.append(round(t1 - t0, 3))
        score_times.append(round(t2 - t1, 3))
        rep_metrics.append(cc.metrics_for(yte, p))
    full = cc.metrics_unrounded(yte, model.predict_proba(Xte)[:, 1])
    return {
        "params": params,
        "metrics": rep_metrics[0],
        "metrics_full": {k: round(v, 6) for k, v in full.items()},
        "metrics_by_rep": rep_metrics,
        "fit_times_s": fit_times,
        "score_times_s": score_times,
        "fit_time_median_s": round(float(np.median(fit_times)), 3),
        "score_time_median_s": round(float(np.median(score_times)), 3),
        "tuning": None,
    }


def tune_arm(arm: str, Xtr, ytr) -> dict:
    """The frozen random search: 24 trials, stratified 3-fold CV
    on the training rows only, selection by mean CV AUC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    grid = GRIDS[arm]
    rng = np.random.RandomState(SEARCH_SEED + ARM_INDEX[arm])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42)
                 .split(Xtr, ytr))
    trials = []
    t_start = time.perf_counter()
    for i in range(N_TRIALS):
        params = sample_params(rng, grid)
        if arm == "lightgbm-tuned":
            params["subsample_freq"] = 1
        fold_aucs = []
        for tr_idx, va_idx in folds:
            model = make_model(arm, params)
            model.fit(Xtr[tr_idx], ytr[tr_idx])
            p = model.predict_proba(Xtr[va_idx])[:, 1]
            fold_aucs.append(round(float(roc_auc_score(ytr[va_idx], p)), 4))
        trials.append({"trial": i + 1, "params": params,
                       "fold_aucs": fold_aucs,
                       "mean_cv_auc": round(float(np.mean(fold_aucs)), 4)})
    wall = round(time.perf_counter() - t_start, 3)
    best = max(trials, key=lambda t: t["mean_cv_auc"])
    cc.dump_json(trials, RESULTS / f"tuning_{arm}.json")
    return {"best_params": best["params"], "best_cv_auc": best["mean_cv_auc"],
            "n_trials": N_TRIALS, "wall_time_s": wall}


def library_versions() -> dict:
    import lightgbm
    import numpy
    import pandas
    import sklearn
    import xgboost
    return {"python": platform.python_version(),
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    Xtr, Xte, ytr, yte = cc.load_matrices()
    data_sha = hashlib.sha256(cc.DATA.read_bytes()).hexdigest()
    manifest = {
        "machine": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": library_versions(),
        "data_file": "data/credit_card.xls",
        "data_sha256": data_sha,
        "train_rows": int(len(ytr)), "test_rows": int(len(yte)),
        "n_features_after_preprocessing": int(Xtr.shape[1]),
        "published_champion": PUBLISHED,
    }
    cc.dump_json(manifest, RESULTS / "run_manifest.json")
    print(f"train {len(ytr)} test {len(yte)} features {Xtr.shape[1]}")

    out = {}
    control = timed_fit_score(
        "control-champion",
        {"max_iter": 300, "learning_rate": 0.06, "max_depth": 3},
        Xtr, ytr, Xte, yte)
    out["control-champion"] = control
    cc.dump_json(out, RESULTS / "benchmark_metrics.json")
    delta = round(control["metrics"]["auc"] - PUBLISHED["auc"], 4)
    print(f"control AUC {control['metrics']['auc']} delta {delta}")
    if abs(delta) > GATE_TOLERANCE:
        print("REPRODUCTION GATE FAILED: harness broken, stopping.")
        manifest["gate_failed"] = True
        cc.dump_json(manifest, RESULTS / "run_manifest.json")
        return
    manifest["gate_failed"] = False
    cc.dump_json(manifest, RESULTS / "run_manifest.json")

    out["xgboost-default"] = timed_fit_score(
        "xgboost-default", {}, Xtr, ytr, Xte, yte)
    out["lightgbm-default"] = timed_fit_score(
        "lightgbm-default", {}, Xtr, ytr, Xte, yte)
    cc.dump_json(out, RESULTS / "benchmark_metrics.json")

    for arm in ["xgboost-tuned", "lightgbm-tuned", "sklearn-tuned"]:
        print(f"tuning {arm} ...")
        tuned = tune_arm(arm, Xtr, ytr)
        result = timed_fit_score(arm, tuned["best_params"],
                                 Xtr, ytr, Xte, yte)
        result["params"] = {k: v for k, v in tuned["best_params"].items()}
        result["tuning"] = {"n_trials": tuned["n_trials"],
                            "best_cv_auc": tuned["best_cv_auc"],
                            "wall_time_s": tuned["wall_time_s"]}
        out[arm] = result
        cc.dump_json(out, RESULTS / "benchmark_metrics.json")
        print(f"{arm}: best CV AUC {tuned['best_cv_auc']}, "
              f"test AUC {result['metrics']['auc']}")

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    cc.dump_json(manifest, RESULTS / "run_manifest.json")
    print("done")


if __name__ == "__main__":
    main()
