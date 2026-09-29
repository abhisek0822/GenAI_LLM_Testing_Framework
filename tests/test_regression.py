from __future__ import annotations

from scripts.compare_baseline import main


def test_regression_baseline_script_passes() -> None:
    assert main() == 0
