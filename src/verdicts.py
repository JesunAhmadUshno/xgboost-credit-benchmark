"""Verdict computation from the frozen pass lines.

Pure functions over the recorded metrics, so the verdicts
in RESULTS.md are reproducible from results/benchmark_
metrics.json by anyone, and unit-testable without the data.
The pass lines are those frozen in PREREGISTRATION.md.
"""

H1_MARGIN = 0.005
H2_MARGIN = 0.010


def _band(delta: float, margin: float) -> str:
    if delta >= margin:
        return "CONFIRMED"
    if delta <= 0.0:
        return "REFUTED"
    return "INCONCLUSIVE"


def compute_verdicts(metrics: dict, gate_failed: bool) -> dict:
    """metrics: the benchmark_metrics.json mapping of arm to record."""
    if gate_failed:
        return {"H1": "NOT RUN", "H2": "NOT RUN", "H3": "NOT RUN",
                "h1_delta_auc": None, "h2_delta_ks": None,
                "order_by_auc": None, "order_by_brier": None}
    control = metrics["control-champion"]["metrics"]
    xgb = metrics["xgboost-tuned"]["metrics"]
    lgb = metrics["lightgbm-tuned"]["metrics"]
    d_auc = round(xgb["auc"] - control["auc"], 4)
    d_ks = round(lgb["ks"] - control["ks"], 4)

    arms = ["control-champion", "xgboost-tuned", "lightgbm-tuned"]
    full = {a: metrics[a]["metrics_full"] for a in arms}
    aucs = [full[a]["auc"] for a in arms]
    briers = [full[a]["brier"] for a in arms]
    if len(set(aucs)) < 3 or len(set(briers)) < 3:
        h3 = "INCONCLUSIVE"
        order_auc = order_brier = None
    else:
        order_auc = [a for a, _ in sorted(
            zip(arms, aucs), key=lambda t: -t[1])]
        order_brier = [a for a, _ in sorted(
            zip(arms, briers), key=lambda t: t[1])]
        h3 = "CONFIRMED" if order_auc != order_brier else "REFUTED"
    return {"H1": _band(d_auc, H1_MARGIN), "H2": _band(d_ks, H2_MARGIN),
            "H3": h3, "h1_delta_auc": d_auc, "h2_delta_ks": d_ks,
            "order_by_auc": order_auc, "order_by_brier": order_brier}
