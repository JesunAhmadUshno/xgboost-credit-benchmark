"""Shared data and metric code for the benchmark.

The loading, feature, and metric definitions re-implement
JesunAhmadUshno/credit-risk-lab src/credit_risk_lab.py at
the pinned commit fee580d0beee43c76dc57c6c366c211533054956
exactly, following the line-faithful re-implementation the
sibling study mlops-evidence-lab verified on 2026-11-08
(its rerun reproduced the published champion figures to
the last printed digit). The subject repo is Apache-2.0.

The one structural difference from the subject is frozen in
PREREGISTRATION.md: the subject's Pipeline (preprocessing
then model) is factored so the fitted ColumnTransformer is
applied once and every benchmark arm receives the identical
dense matrix. The control arm is the test that this
factoring changed nothing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "credit_card.xls"
TARGET_RAW = "default payment next month"
SEED = 42
TEST_SIZE = 0.30

FEATURES = (["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1", "utilization",
             "pay_ratio", "delinq_mean", "delinq_max", "bill_trend"] +
            ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"])
CATS = ["SEX", "EDUCATION", "MARRIAGE"]


def load_frame(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={TARGET_RAW: "default"})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pay = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["utilization"] = df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["pay_ratio"] = df["PAY_AMT1"] / df["BILL_AMT1"].replace(0, np.nan)
    df["pay_ratio"] = df["pay_ratio"].fillna(0).clip(-1, 5)
    df["delinq_mean"] = df[pay].clip(lower=0).mean(axis=1)
    df["delinq_max"] = df[pay].clip(lower=0).max(axis=1)
    df["bill_trend"] = (df["BILL_AMT1"] - df["BILL_AMT6"]) / df["LIMIT_BAL"].replace(0, np.nan)
    df["dpd_months"] = df["PAY_0"].clip(lower=0)
    return df


def make_preprocessor() -> ColumnTransformer:
    """The subject's preprocessing step, unfitted."""
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATS),
    ])


def load_split():
    """The frozen split: stratified 70/30, seed 42, as registered."""
    from sklearn.model_selection import train_test_split
    df = add_features(load_frame())
    X, y = df[FEATURES + CATS], df["default"]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=SEED)


def load_matrices():
    """Split, fit the preprocessor on train only, transform both.

    Returns (X_train, X_test, y_train, y_test) as dense
    float64 arrays and the target Series unchanged.
    """
    X_train, X_test, y_train, y_test = load_split()
    pre = make_preprocessor()
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    if hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
    if hasattr(Xte, "toarray"):
        Xte = Xte.toarray()
    return (np.asarray(Xtr, dtype=np.float64),
            np.asarray(Xte, dtype=np.float64),
            np.asarray(y_train), np.asarray(y_test))


def ks_stat(y, p) -> float:
    order = np.argsort(-p)
    y = np.asarray(y)[order]
    tp = np.cumsum(y) / max(y.sum(), 1)
    fp = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(tp - fp))


def metrics_for(y_true, p_test) -> dict:
    """AUC, KS, Gini, Brier on the frozen test split, 4 decimals."""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    auc = roc_auc_score(y_true, p_test)
    return {
        "auc": round(float(auc), 4),
        "ks": round(ks_stat(np.asarray(y_true), p_test), 4),
        "gini": round(float(2 * auc - 1), 4),
        "brier": round(float(brier_score_loss(y_true, p_test)), 4),
    }


def metrics_unrounded(y_true, p_test) -> dict:
    """The same metrics at full precision (H3 ordering uses these)."""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    auc = roc_auc_score(y_true, p_test)
    return {
        "auc": float(auc),
        "ks": ks_stat(np.asarray(y_true), p_test),
        "gini": float(2 * auc - 1),
        "brier": float(brier_score_loss(y_true, p_test)),
    }


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
