"""
Regression test for evaluation/threshold_analysis.py.

Runs the script against the tamborin prediction files and compares the
generated markdown report to the golden reference at
reports/pruner/scier/scier_preds/threshold_analysis_report.md.

The test writes to a temporary directory so it never touches the golden file.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

GOLDEN_REPORT = ROOT / "reports" / "pruner" / "scier" / "scier_preds" / "threshold_analysis_report.md"
SCRIPT        = ROOT / "evaluation" / "threshold_analysis.py"
DEV_PRED      = ROOT / "saves" / "scier" / "pruner" / "tamborin" / "ent_pred_dev.json"
TRAIN_PRED    = ROOT / "saves" / "scier" / "pruner" / "tamborin" / "ent_pred_train.json"
TEST_PRED     = ROOT / "saves" / "scier" / "pruner" / "tamborin" / "ent_pred_test.json"


def _skip_if_missing(*paths: Path):
    for p in paths:
        if not p.exists():
            pytest.skip(f"Required file not found: {p}")


def test_report_matches_golden(tmp_path):
    _skip_if_missing(GOLDEN_REPORT, SCRIPT, DEV_PRED, TRAIN_PRED, TEST_PRED)

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--pred",       str(DEV_PRED.relative_to(ROOT)),
            "--train-pred", str(TRAIN_PRED.relative_to(ROOT)),
            "--test-pred",  str(TEST_PRED.relative_to(ROOT)),
            "--dataset",    "scier",
            "--out-dir",    str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    assert result.returncode == 0, (
        f"Script exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    generated = (tmp_path / "threshold_analysis_report.md").read_text()
    golden    = GOLDEN_REPORT.read_text()

    assert generated == golden, (
        "Generated report does not match the golden reference.\n\n"
        "--- golden ---\n" + golden[:500] + "\n\n"
        "--- generated ---\n" + generated[:500]
    )
