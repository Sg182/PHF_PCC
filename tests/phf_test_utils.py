"""Helpers for the PHF optimizer regression tests (run with tests/run_tests.sh)."""
import json
import os
import pickle
import subprocess
import sys

import numpy as np
from pyscf import gto, scf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # PHF_PCC
PY = sys.executable
TMP = os.path.join(HERE, "tmp")
os.makedirs(TMP, exist_ok=True)

H4_ATOMS = "H 0.6 0.6 0; H -0.6 0.6 0; H -0.6 -0.6 0; H 0.6 -0.6 0"      # square, side 1.2 A
N2_ATOMS = "N 0 0 0; N 0 0 1.5"

BASE_SETTINGS = {"unit": "Ang", "basis": ["cc-pvdz"], "do_CC": False, "PG": "None", "Irrep": "Ag",
                 "J": 0, "M": 0, "CmplxConj": 0, "VERBOSE": 1, "chkpoint": False, "read_PHF": False,
                 "read_PCC": False, "refine_PHF": False, "DIIS": True}


def write_xyz(atoms, name):
    rows = [a.strip().split() for a in atoms.split(";")]
    path = os.path.join(TMP, name + ".xyz")
    with open(path, "w") as f:
        f.write(f"{len(rows)}\n{name}\n" + "".join(" ".join(r) + "\n" for r in rows))
    return path


def uhf_neel(mol):
    """Broken-symmetry UHF from an alternating alpha/beta guess, internal stability followed."""
    uhf = scf.UHF(mol)
    dma, dmb = (d.copy() for d in uhf.get_init_guess())
    for k, (b, e) in enumerate(tuple(x[2:]) for x in mol.aoslice_by_atom()):
        up = 1.5 if k % 2 == 0 else 0.5
        dma[b:e, b:e] *= up; dmb[b:e, b:e] *= 2.0 - up
    uhf.kernel((dma, dmb))
    for _ in range(5):
        mo, _, st, _ = uhf.stability(return_status=True)
        if st:
            break
        uhf.kernel(uhf.make_rdm1(mo, uhf.mo_occ))
    return uhf


def ghf_layout(uhf):
    Ca, Cb = uhf.mo_coeff; n = Ca.shape[0]; na, nb = uhf.nelec
    C = np.zeros((2 * n, 2 * n))
    C[:n, :na] = Ca[:, :na]; C[n:, na:na + nb] = Cb[:, :nb]
    C[:n, na + nb:na + nb + n - na] = Ca[:, na:]; C[n:, 2 * n - (n - nb):] = Cb[:, nb:]
    return C


def cant(C, mol, nocc, theta_deg):
    """Symmetric +/- theta/2 canting about y on even/odd atoms; full orthonormalisation in the AO metric."""
    n = C.shape[0] // 2; Cn = C.copy()
    for k, (b, e) in enumerate(tuple(x[2:]) for x in mol.aoslice_by_atom()):
        t = np.deg2rad(theta_deg) / 2.0 * (1.0 if k % 2 == 0 else -1.0)
        a_blk = Cn[b:e, :nocc].copy(); b_blk = Cn[n + b:n + e, :nocc].copy()
        Cn[b:e, :nocc] = np.cos(t) * a_blk - np.sin(t) * b_blk
        Cn[n + b:n + e, :nocc] = np.sin(t) * a_blk + np.cos(t) * b_blk
    S2 = np.kron(np.eye(2), mol.intor("int1e_ovlp"))
    occ = Cn[:, :nocc]; w, v = np.linalg.eigh(occ.T @ S2 @ occ); occ = occ @ v @ np.diag(w ** -0.5) @ v.T
    vir = Cn[:, nocc:] - occ @ (occ.T @ S2 @ Cn[:, nocc:]); w, v = np.linalg.eigh(vir.T @ S2 @ vir)
    vir = vir @ v @ np.diag(w ** -0.5) @ v.T
    return np.hstack((occ, vir))


def make_seed(atoms, basis, name, cant_deg=0.0):
    """orb_init pickle: stability-followed Neel UHF, optionally canted (noncollinear).  Returns (path, mol)."""
    mol = gto.M(atom=atoms, basis=basis, unit="Ang", verbose=0)
    C = ghf_layout(uhf_neel(mol))
    if cant_deg:
        C = cant(C, mol, mol.nelectron, cant_deg)
    path = os.path.join(TMP, name + ".p")
    pickle.dump(C, open(path, "wb"))
    return path, mol


def write_settings(name, **kw):
    s = dict(BASE_SETTINGS); s.update(kw); s["mol_name"] = name
    path = os.path.join(TMP, name + ".json")
    json.dump(s, open(path, "w"), indent=1)
    return path


class Captured(Exception):
    pass


def capture_phf_args(settings_file):
    """Run the driver up to the PHF optimizer call and capture (z0, eg_args, settings, Values)."""
    sys.path.insert(0, ROOT)
    cwd = os.getcwd(); os.chdir(ROOT)
    import PCC_run_PHF as M
    import PCC_driver
    store = {}
    orig = M.phf_optimize

    def fake(z0, eg_args, settings, Values, use_ipopt, nhop, gtol):
        store.update(z0=z0, eg_args=eg_args, settings=settings, Values=Values, gtol=gtol, M=M)
        raise Captured()
    M.phf_optimize = fake
    try:
        PCC_driver.driver(file=settings_file)
    except Captured:
        pass
    finally:
        M.phf_optimize = orig
        os.chdir(cwd)
    assert "eg_args" in store, "driver did not reach the PHF optimizer"
    return store


def run_driver(settings_file, timeout=3600):
    """Run PHF_PCC/run.py in a subprocess; return (E_PHF, gmax, stdout)."""
    out = subprocess.run([PY, os.path.join(ROOT, "run.py"), "-s", settings_file], cwd=ROOT,
                         capture_output=True, text=True, timeout=timeout)
    txt = out.stdout + out.stderr
    e = [float(l.split("=")[1]) for l in txt.splitlines() if l.startswith("E(PHF)=")]
    g = [float(l.split("max|G| = ")[1].split(",")[0]) for l in txt.splitlines() if "PHF optimizer:" in l]
    assert out.returncode == 0 and e, "driver failed:\n" + txt[-3000:]
    return e[-1], g[-1], txt


def spin_blocks(Values):
    """Boolean masks (flattened (vir,occ)) for same-spin and spin-flip Thouless blocks."""
    nv, no = Values.NVrtSO, Values.NOccSO
    same = ((np.arange(nv)[:, None] < Values.NVrtA) == (np.arange(no)[None, :] < Values.NOccA)).reshape(-1)
    return same, ~same
