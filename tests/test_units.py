"""Unit tests on synthetic data only (CI never fetches the corpus)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import credit_common as cc
from benchmark import GRIDS, sample_params
from verdicts import compute_verdicts


def test_ks_stat_perfect_and_constant():
    y = np.array([0, 0, 1, 1])
    assert cc.ks_stat(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert cc.ks_stat(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0


def test_metrics_for_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    m = cc.metrics_for(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc"] == 1.0 and m["ks"] == 1.0
    assert m["gini"] == 1.0 and m["brier"] < 0.2


def test_add_features_engineering():
    cols = {c: [1000, 0] for c in
            ["LIMIT_BAL", "BILL_AMT1", "BILL_AMT6", "PAY_AMT1"]}
    for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        cols[c] = [2, -1]
    df = cc.add_features(pd.DataFrame(cols))
    assert df.loc[0, "utilization"] == 1.0
    assert np.isnan(df.loc[1, "utilization"])
    assert df.loc[0, "pay_ratio"] == 1.0
    assert df.loc[0, "delinq_mean"] == 2.0 and df.loc[1, "delinq_mean"] == 0.0
    assert df.loc[0, "bill_trend"] == 0.0


def test_sampler_deterministic_and_in_grid():
    for arm, grid in GRIDS.items():
        a = sample_params(np.random.RandomState(7), grid)
        b = sample_params(np.random.RandomState(7), grid)
        assert a == b
        for name, value in a.items():
            assert value in grid[name]


def _metrics(auc_c, ks_c, brier_c, auc_x, ks_x, brier_x,
             auc_l, ks_l, brier_l):
    def rec(auc, ks, brier):
        return {"metrics": {"auc": auc, "ks": ks, "gini": 0.5,
                            "brier": brier},
                "metrics_full": {"auc": auc, "ks": ks, "brier": brier}}
    return {"control-champion": rec(auc_c, ks_c, brier_c),
            "xgboost-tuned": rec(auc_x, ks_x, brier_x),
            "lightgbm-tuned": rec(auc_l, ks_l, brier_l)}


def test_verdict_bands_h1_h2():
    m = _metrics(0.7800, 0.4200, 0.1350,
                 0.7860, 0.4200, 0.1350,
                 0.7800, 0.4350, 0.1350)
    v = compute_verdicts(m, False)
    assert v["H1"] == "CONFIRMED" and v["h1_delta_auc"] == 0.006
    assert v["H2"] == "CONFIRMED" and v["h2_delta_ks"] == 0.015
    m = _metrics(0.7800, 0.4200, 0.1350,
                 0.7820, 0.4200, 0.1350,
                 0.7800, 0.4190, 0.1350)
    v = compute_verdicts(m, False)
    assert v["H1"] == "INCONCLUSIVE" and v["H2"] == "REFUTED"
    m = _metrics(0.7800, 0.4200, 0.1350,
                 0.7790, 0.4200, 0.1350,
                 0.7800, 0.4200, 0.1350)
    assert compute_verdicts(m, False)["H1"] == "REFUTED"


def test_verdict_h3_orderings():
    same = _metrics(0.7800, 0.4200, 0.1370,
                    0.7850, 0.4200, 0.1360,
                    0.7900, 0.4200, 0.1350)
    v = compute_verdicts(same, False)
    assert v["H3"] == "REFUTED"
    assert v["order_by_auc"] == v["order_by_brier"]
    flipped = _metrics(0.7800, 0.4200, 0.1350,
                       0.7850, 0.4200, 0.1370,
                       0.7900, 0.4200, 0.1360)
    v = compute_verdicts(flipped, False)
    assert v["H3"] == "CONFIRMED"
    assert v["order_by_auc"] != v["order_by_brier"]


def test_verdicts_not_run_when_gate_failed():
    v = compute_verdicts({}, True)
    assert v["H1"] == v["H2"] == v["H3"] == "NOT RUN"
