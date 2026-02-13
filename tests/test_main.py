"""Tests for main entry point and run_verification pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_run_verification_imports() -> None:
    from src.main import run_verification
    assert callable(run_verification)


def test_run_verification_returns_report() -> None:
    from src.main import run_verification
    report = run_verification("A test claim that is short.")
    assert "original_input" in report
    assert "claims" in report
    assert "final_verification" in report
    assert "graph_summary" in report
    assert len(report["claims"]) >= 1
    assert len(report["final_verification"]) >= 1


def test_run_verification_with_correction_params() -> None:
    from src.main import run_verification
    params = {"intervention_threshold": 0.5, "causal_bias": 0.4, "causal_truth": 0.85}
    report = run_verification("A test claim that is short.", correction_params=params)
    assert report.get("correction_params") == params
