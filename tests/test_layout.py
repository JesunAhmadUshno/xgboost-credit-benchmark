"""Portfolio standard checks: the lock holds, required files exist, no public grade."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# SHA-256 of PREREGISTRATION.md as committed in f5bb562 (tag proposal-v1), before any code or result.
LOCKED_SHA256 = "2a17541652bb4464df9fd5a6e7876b9d506c617bad0692394020b682b8fea04f"
GRADE_MARKERS = ("of 10,000", "/10,000", "/10000", "Grade:")


def test_preregistration_is_unchanged_since_the_lock():
    digest = hashlib.sha256((ROOT / "PREREGISTRATION.md").read_bytes()).hexdigest()
    assert digest == LOCKED_SHA256, \
        "PREREGISTRATION.md changed after its lock; record an amendment in DECISIONS.md instead"


def test_required_files_exist():
    need = ["README.md", "PREREGISTRATION.md", "AUDIT.md", "RESULTS.md", "DECISIONS.md",
            "CHANGELOG.md", "CITATION.cff", "SECURITY.md", "LICENSE",
            "results/benchmark_metrics.json", "results/verdicts.json", "results/run_manifest.json"]
    missing = [p for p in need if not (ROOT / p).exists()]
    assert not missing, "missing: " + ", ".join(missing)


def test_no_public_grade():
    public = list(ROOT.glob("*.md")) + [ROOT / "CITATION.cff"]
    for path in public:
        text = path.read_text()
        hits = [m for m in GRADE_MARKERS if m in text]
        assert not hits, f"{path.name} carries a grade marker {hits}; grades are internal"
