"""Amendment A1 (DECISIONS.md D6, registered 2026-11-17 before
any of this ran): uncertainty intervals for the two headline
deltas of the frozen study.

Post-registration analysis. It changes no frozen verdict and
writes only results/amendment_a1_bootstrap.json. Two parts:

  bootstrap  Paired bootstrap on the locked test set: the three
             final models are refit with the exact final
             configurations recorded in
             results/benchmark_metrics.json (built through
             benchmark.make_model), scored once on the frozen
             9,000-row test split, and the refit must reproduce
             the recorded unrounded metrics before any interval
             is computed. Then 5,000 paired resamples of the
             test rows (RandomState seed 20261117) give
             percentile 95% intervals for the AUC and KS deltas
             of each tuned challenger against the champion.
  splits     Ten repeated stratified 70/30 splits (seeds 1001
             to 1010), each re-running the frozen training and
             selection protocol (benchmark.tune_arm, unchanged)
             for xgboost-tuned and lightgbm-tuned, with the
             control at its published configuration. Per-split
             trial records go to a scratch directory so the
             frozen results/tuning_<arm>.json files are never
             overwritten.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import hashlib
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import credit_common as cc
import benchmark as bm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "amendment_a1_bootstrap.json"
RECORDED = json.loads((ROOT / "results" / "benchmark_metrics.json")
                      .read_text())

BOOT_SEED = 20261117
N_BOOT = 5000
SPLIT_SEEDS = list(range(1001, 1011))
ARMS = ["control-champion", "xgboost-tuned", "lightgbm-tuned"]
CHALLENGERS = ["xgboost-tuned", "lightgbm-tuned"]


def fit_predict(arm, Xtr, ytr, Xte):
    model = bm.make_model(arm, RECORDED[arm]["params"])
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def delta_block(y, preds):
    """Unrounded AUC/KS for every arm and the four deltas."""
    m = {arm: cc.metrics_unrounded(y, preds[arm]) for arm in ARMS}
    d = {}
    for ch in CHALLENGERS:
        d[f"{ch} auc_delta"] = m[ch]["auc"] - m["control-champion"]["auc"]
        d[f"{ch} ks_delta"] = m[ch]["ks"] - m["control-champion"]["ks"]
    return m, d


def run_bootstrap() -> dict:
    Xtr, Xte, ytr, yte = cc.load_matrices()
    preds = {arm: fit_predict(arm, Xtr, ytr, Xte) for arm in ARMS}

    # Gate: the refit must reproduce the recorded unrounded metrics.
    check = {}
    ok = True
    for arm in ARMS:
        got = cc.metrics_unrounded(yte, preds[arm])
        want = RECORDED[arm]["metrics_full"]
        diffs = {k: abs(got[k] - want[k]) for k in ("auc", "ks", "brier")}
        check[arm] = {"recomputed": {k: round(got[k], 6)
                                     for k in ("auc", "ks", "brier")},
                      "recorded": want,
                      "max_abs_diff": max(diffs.values())}
        if max(diffs.values()) > 2e-6:
            ok = False
    if not ok:
        return {"status": "REFIT MISMATCH: intervals not computed",
                "refit_check": check}

    _, point = delta_block(yte, preds)
    n = len(yte)
    rng = np.random.RandomState(BOOT_SEED)
    keys = list(point.keys())
    draws = {k: np.empty(N_BOOT) for k in keys}
    t0 = time.perf_counter()
    for b in range(N_BOOT):
        idx = rng.randint(0, n, size=n)
        yb = yte[idx]
        pb = {arm: preds[arm][idx] for arm in ARMS}
        _, d = delta_block(yb, pb)
        for k in keys:
            draws[k][b] = d[k]
        if (b + 1) % 1000 == 0:
            print(f"bootstrap {b + 1}/{N_BOOT} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    intervals = {}
    for k in keys:
        arr = draws[k]
        intervals[k] = {
            "point_estimate_unrounded": point[k],
            "ci95_percentile": [float(np.percentile(arr, 2.5)),
                                float(np.percentile(arr, 97.5))],
            "bootstrap_mean": float(arr.mean()),
            "bootstrap_sd": float(arr.std(ddof=1)),
            "frac_resamples_le_zero": float(np.mean(arr <= 0)),
            "includes_zero": bool(np.percentile(arr, 2.5) <= 0
                                  <= np.percentile(arr, 97.5)),
        }
    return {"status": "ok", "refit_check": check,
            "n_test_rows": int(n), "n_resamples": N_BOOT,
            "seed": BOOT_SEED, "intervals": intervals}


def run_splits(existing: dict) -> dict:
    from sklearn.model_selection import train_test_split
    scratch = Path(tempfile.mkdtemp(prefix="amendment_a1_trials_"))
    bm.RESULTS = scratch  # tune_arm writes trial JSONs here, not results/
    df = cc.add_features(cc.load_frame())
    X = df[cc.FEATURES + cc.CATS]
    y = df["default"]
    per_split = existing.get("repeated_splits", {}).get("per_split", [])
    done = {r["split_seed"] for r in per_split}
    for seed in SPLIT_SEEDS:
        if seed in done:
            continue
        t0 = time.perf_counter()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cc.TEST_SIZE, stratify=y, random_state=seed)
        pre = cc.make_preprocessor()
        Xtr = pre.fit_transform(X_train)
        Xte = pre.transform(X_test)
        if hasattr(Xtr, "toarray"):
            Xtr = Xtr.toarray()
        if hasattr(Xte, "toarray"):
            Xte = Xte.toarray()
        Xtr = np.asarray(Xtr, dtype=np.float64)
        Xte = np.asarray(Xte, dtype=np.float64)
        ytr = np.asarray(y_train)
        yte = np.asarray(y_test)

        preds = {"control-champion": fit_predict("control-champion",
                                                 Xtr, ytr, Xte)}
        best = {}
        for arm in CHALLENGERS:
            tuned = bm.tune_arm(arm, Xtr, ytr)
            best[arm] = {"best_params": tuned["best_params"],
                         "best_cv_auc": tuned["best_cv_auc"]}
            model = bm.make_model(arm, tuned["best_params"])
            model.fit(Xtr, ytr)
            preds[arm] = model.predict_proba(Xte)[:, 1]
        metrics, deltas = delta_block(yte, preds)
        per_split.append({
            "split_seed": seed,
            "metrics_unrounded": {a: metrics[a] for a in ARMS},
            "deltas_unrounded": deltas,
            "tuning": best,
            "wall_time_s": round(time.perf_counter() - t0, 1),
        })
        existing["repeated_splits"] = summarise(per_split)
        save(existing)
        print(f"split seed {seed} done "
              f"({per_split[-1]['wall_time_s']}s): {deltas}", flush=True)
    return existing


def summarise(per_split: list) -> dict:
    keys = list(per_split[0]["deltas_unrounded"].keys())
    summary = {}
    for k in keys:
        vals = np.array([r["deltas_unrounded"][k] for r in per_split])
        summary[k] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n_positive": int(np.sum(vals > 0)),
            "n_nonpositive": int(np.sum(vals <= 0)),
        }
    return {"split_seeds": SPLIT_SEEDS, "per_split": per_split,
            "distribution": summary}


def save(obj: dict) -> None:
    obj["amendment"] = "A1 (DECISIONS.md D6, 2026-11-17)"
    obj["updated_utc"] = datetime.now(timezone.utc).isoformat()
    cc.dump_json(obj, OUT)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    existing = {}
    if OUT.exists():
        existing = json.loads(OUT.read_text())
    existing["method"] = {
        "bootstrap": "paired resampling of the frozen test rows, "
                     "percentile 95% intervals, unrounded deltas",
        "splits": "frozen protocol re-run per split via "
                  "benchmark.tune_arm; control at published config",
        "data_sha256": hashlib.sha256(cc.DATA.read_bytes()).hexdigest(),
        "machine": platform.node(),
        "versions": bm.library_versions(),
    }
    if mode in ("bootstrap", "all"):
        print("running paired bootstrap ...", flush=True)
        existing["bootstrap"] = run_bootstrap()
        save(existing)
        print("bootstrap done", flush=True)
        if existing["bootstrap"]["status"] != "ok":
            print(existing["bootstrap"]["status"])
            return
    if mode in ("splits", "all"):
        print("running repeated splits ...", flush=True)
        existing = run_splits(existing)
        save(existing)
        print("splits done", flush=True)


if __name__ == "__main__":
    main()
