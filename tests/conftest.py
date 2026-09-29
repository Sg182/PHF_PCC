import os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDED = [ROOT, os.path.join(ROOT, "PHF_fort")]


def pytest_configure(config):
    ld = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if not all(p in ld for p in NEEDED):
        pytest.exit("LD_LIBRARY_PATH must include PHF_PCC and PHF_PCC/PHF_fort (use tests/run_tests.sh)", returncode=3)
    # numpy-2 pickles / scipy 1.6 shims are only needed by the driver subprocess (run.py handles them)
