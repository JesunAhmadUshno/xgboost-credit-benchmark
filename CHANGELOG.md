# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning; no release is tagged yet.

## [Unreleased]

### Changed
- 2026-11-08: Adopted the portfolio standard v2. README opens with the
  question, headline, field, job family and verdict (Functional), then
  reproduce, lifecycle and a layout table mapping the locked file
  names to their roles. AUDIT.md is the report: its grade section
  left the public repo (grades are internal) and its deductions are
  kept in full as "Threats to validity", beside a new "How the
  evidence is protected" section and a table scoring H1 to H3. A
  test pins the SHA-256 of PREREGISTRATION.md as locked in f5bb562.
  CI runs on Python 3.11 and 3.12 with read-only permissions and
  checks that RESULTS.md and results/verdicts.json regenerate
  unchanged. CodeQL and Dependabot added. DECISIONS.md D6.

### Added
- 2026-11-08: Results: all six arms on the machine of record; H1 and
  H2 INCONCLUSIVE (sub-margin beats), H3 CONFIRMED; README, AUDIT,
  DECISIONS (9e69930).
- 2026-11-08: Benchmark harness: six frozen arms, equal tuning budget,
  reproduction gate, verdict generator (94e41d8).
- 2026-11-08: Pre-registration (f5bb562), before any code or result.
