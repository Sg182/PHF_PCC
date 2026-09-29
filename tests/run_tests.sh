#!/bin/bash
# PHF optimizer regression tests.  Fast suite ~5 min; PHF_SLOW_TESTS=1 adds the N2/H4 SGHF energy tests (~30 min).
#   bash tests/run_tests.sh            # fast
#   PHF_SLOW_TESTS=1 bash tests/run_tests.sh
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
CCENV=/home/sg182/miniconda3/envs/cc
export LD_LIBRARY_PATH="$ROOT:$ROOT/PHF_fort:$CCENV/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTHONUNBUFFERED=1
cd "$ROOT" && exec "$CCENV/bin/python" -m pytest -x -v "$HERE" "$@"
