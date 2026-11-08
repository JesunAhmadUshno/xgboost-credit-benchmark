"""Run the Jev claim_check gate on the headline verdicts.

Builds the claim from results/verdicts.json and
results/benchmark_metrics.json (never hand-typed), writes
the spec to results/jev-claim-spec.json, calls the TypeSafe
connector helper (the vault flow; the key never appears
here), and saves the raw response to
results/jev-claim-check.json. The bands are the house
claim_check bands: clear at noul 0.85 or above, block at
0.10 or below, review between.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path.home() / "workspace/custom-connectors/custom.typesafe/bin/typesafe_jev.py"
PUBLISHED = {"auc": 0.7810, "ks": 0.4196, "brier": 0.1355}


def main() -> None:
    verdicts = json.loads((ROOT / "results" / "verdicts.json").read_text())
    metrics = json.loads(
        (ROOT / "results" / "benchmark_metrics.json").read_text())
    c = metrics["control-champion"]["metrics"]
    x = metrics["xgboost-tuned"]["metrics"]
    l = metrics["lightgbm-tuned"]["metrics"]
    s = metrics["sklearn-tuned"]["metrics"]
    claim = (
        f"Under one pre-registered protocol (PREREGISTRATION.md is the "
        f"first commit of the repository; verdicts computed by "
        f"scripts/make_results.py from the recorded artifacts): the "
        f"control arm reproduces credit-risk-lab's published champion "
        f"exactly (AUC {c['auc']:.4f} vs published {PUBLISHED['auc']:.4f}, "
        f"KS {c['ks']:.4f} vs {PUBLISHED['ks']:.4f}, Brier "
        f"{c['brier']:.4f} vs {PUBLISHED['brier']:.4f}). "
        f"H1 is {verdicts['H1']}: tuned XGBoost test AUC {x['auc']:.4f} "
        f"against the champion {c['auc']:.4f} (delta "
        f"{verdicts['h1_delta_auc']:+.4f}, frozen margin +0.005). "
        f"H2 is {verdicts['H2']}: tuned LightGBM test KS {l['ks']:.4f} "
        f"against the champion {c['ks']:.4f} (delta "
        f"{verdicts['h2_delta_ks']:+.4f}, frozen margin +0.010). "
        f"H3 is {verdicts['H3']}: the AUC ranking "
        f"{verdicts['order_by_auc']} and the Brier ranking "
        f"{verdicts['order_by_brier']} "
        f"{'differ' if verdicts['H3'] == 'CONFIRMED' else 'are identical'}"
        f". The equally tuned sklearn reference arm scored AUC "
        f"{s['auc']:.4f}, KS {s['ks']:.4f}, Brier {s['brier']:.4f}."
    )
    evidence = (ROOT / "RESULTS.md").read_text()
    spec = {
        "model": "jev-1.13.0",
        "state": {"claim": claim, "evidence": evidence},
        "questions": {
            "supported": {
                "type": "noul",
                "instructions": "Does the `evidence` in state fully support the `claim` in state, with no part of the claim going beyond the evidence?",
                "criteria": {
                    "true": "Every part of the claim is directly backed by the evidence",
                    "false": "Any part of the claim is missing from, contradicted by, or stronger than the evidence",
                },
            }
        },
    }
    spec_path = ROOT / "results" / "jev-claim-spec.json"
    spec_path.write_text(json.dumps(spec, indent=1))
    proc = subprocess.run(
        [sys.executable, str(HELPER), "evaluate", str(spec_path)],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("jev helper failed")
    outer = json.loads(proc.stdout)
    (ROOT / "results" / "jev-claim-check.json").write_text(
        json.dumps(outer, indent=1))
    body = outer.get("response", outer)
    answers = body.get("answers", {})
    noul = answers.get("supported", {}).get("noul")
    route = ("clear" if noul is not None and noul >= 0.85 else
             "block" if noul is not None and noul <= 0.10 else "review")
    print(json.dumps({
        "model": body.get("model"), "noul": noul, "route": route,
        "usage": body.get("usage"),
    }, indent=1))


if __name__ == "__main__":
    main()
