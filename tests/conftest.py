"""Shared pytest setup for Project 2 (provided by the harness)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def real_data_dir():
    """Path to input_data/. Skips the test when the dataset is not present."""
    d = ROOT / "input_data"
    if not any(d.glob("*.csv")):
        pytest.skip("Dataset not found in input_data/")
    return d
