#!/usr/bin/env python
"""
Launcher for the PHF-PCC driver in the `cc` conda env (python 3.8, numpy 1.20, scipy 1.6).

    cd /home/sg182/PCC/PHF_PCC
    export OMP_NUM_THREADS=8
    /home/sg182/miniconda3/envs/cc/bin/python run.py -s H2.json

Same arguments as PCC_driver.py (-s settings.json [-g geom.xyz] [-b basis]).  It only
(1) sets LD_LIBRARY_PATH for the Fortran modules and re-execs itself if needed,
(2) aliases numpy._core -> numpy.core so numpy-2 orbital pickles (*_orb) load,
(3) gives scipy 1.6's OptimizeResult the .success attribute basinhopping lacks,
then calls PCC_driver.driver().  With a modern env (numpy 2 / scipy >= 1.9) none of
this is needed and `python PCC_driver.py -s H2.json` works directly.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CCLIB = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
NEEDED = [HERE, os.path.join(HERE, "PHF_fort"), CCLIB]

ld = os.environ.get("LD_LIBRARY_PATH", "")
if not all(p in ld.split(":") for p in NEEDED):
    os.environ["LD_LIBRARY_PATH"] = ":".join(NEEDED + ([ld] if ld else []))
    if not sys.stdout.isatty():
        os.environ["PYTHONUNBUFFERED"] = "1"                        # keep progress lines when redirected to a file
    os.execv(sys.executable, [sys.executable] + sys.argv)          # shared libs resolve at exec time

import numpy.core, numpy.core.multiarray, numpy.core.numeric, numpy.core._multiarray_umath
for name, mod in (("numpy._core", numpy.core), ("numpy._core.multiarray", numpy.core.multiarray),
                  ("numpy._core.numeric", numpy.core.numeric),
                  ("numpy._core._multiarray_umath", numpy.core._multiarray_umath)):
    sys.modules.setdefault(name, mod)

from scipy.optimize import OptimizeResult
if not hasattr(OptimizeResult, "success"):
    _orig = OptimizeResult.__getattr__

    def _getattr(self, name):
        try:
            return _orig(self, name)
        except AttributeError:
            if name == "success":
                return self.get("status", 0) == 0 if "status" in self else True
            raise
    OptimizeResult.__getattr__ = _getattr

os.chdir(HERE)
sys.path.insert(0, HERE)
import PCC_driver
PCC_driver.driver()
