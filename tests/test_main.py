"""Minimal test for main entry point."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_main_imports() -> None:
    from src.main import main
    assert callable(main)
