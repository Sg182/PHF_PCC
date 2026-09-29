"""Regression tests for the PHF optimizer stack: analytic gradient (real/imag, same-spin/spin-flip),
orb_init handling, post-solve validation, and reference energies.  See run_tests.sh."""
import os
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from phf_test_utils import (H4_ATOMS, N2_ATOMS, capture_phf_args, make_seed, run_driver, spin_blocks,
                            write_settings, write_xyz)

SLOW = pytest.mark.skipif(os.environ.get("PHF_SLOW_TESTS") != "1", reason="set PHF_SLOW_TESTS=1")


def _fd_block_errors(store, x, h=1e-4, per_block=6, seed=0):
    """max |analytic - central FD| per (real/imag) x (same-spin/spin-flip) block at parameter vector x."""
    M, args, V = store["M"], store["eg_args"], store["Values"]
    fun = M.EandG; rng = np.random.default_rng(seed)
    n = len(x); m = n // 2; same, flip = spin_blocks(V)
    f0, g = fun(x, *args)
    out = {}
    for part, off in (("real", 0), ("imag", m)):
        for blk, mask in (("same-spin", same), ("spin-flip", flip)):
            idx = rng.choice(np.where(mask)[0], per_block, replace=False)
            errs, fds = [], []
            for c in idx:
                e = np.zeros(n); e[off + c] = h
                fd = (fun(x + e, *args)[0] - fun(x - e, *args)[0]) / (2 * h)
                errs.append(abs(g[off + c] - fd)); fds.append(abs(fd))
            out[(part, blk)] = (max(errs), max(fds))
    return out


@pytest.fixture(scope="module")
def h4_sghf_complex():
    "square H4, SP=1, CmplxConj=1, [6,6] grid, orthonormal +/-1 deg canted seed"
    xyz = write_xyz(H4_ATOMS, "H4sq")
    seed, _ = make_seed(H4_ATOMS, "cc-pvdz", "H4_canted2", cant_deg=2.0)
    js = write_settings("H4_sghf_cplx", geom_file=xyz, orb_init=seed, SP=1, ngrid=[6, 6], CmplxConj=1)
    return capture_phf_args(js)


def test_sghf_gradient_all_blocks_at_nonzero_z(h4_sghf_complex):
    """Real and imaginary gradient components, same-spin and spin-flip blocks, at z != 0 (complex)."""
    st = h4_sghf_complex; z0 = st["z0"]; m = len(z0) // 2
    rng = np.random.default_rng(7)
    x = z0 + rng.standard_normal(len(z0)) * np.r_[0.05 * np.ones(m), 0.01 * np.ones(m)]
    errs = _fd_block_errors(st, x)
    for key, (err, fdmax) in errs.items():
        assert err < 1e-6, f"{key}: |analytic-FD| = {err:.2e} (max|FD| {fdmax:.2e})"
    # the blocks must actually be exercised (nonzero derivatives) for the check to be meaningful
    assert errs[("real", "spin-flip")][1] > 1e-4 and errs[("imag", "same-spin")][1] > 1e-4


def test_sghf_gradient_at_kicked_start(h4_sghf_complex):
    """At the (kicked) starting point itself: real-parameter blocks exact."""
    st = h4_sghf_complex
    errs = _fd_block_errors(st, st["z0"].copy(), seed=1)
    assert errs[("real", "same-spin")][0] < 1e-6 and errs[("real", "spin-flip")][0] < 1e-6


def test_suhf_gradient_same_spin_and_flip_inert():
    """SP=2: same-spin block exact; spin-flip block projected out (zero analytic and FD)."""
    xyz = write_xyz(H4_ATOMS, "H4sq")
    seed, _ = make_seed(H4_ATOMS, "cc-pvdz", "H4_uhf")
    js = write_settings("H4_suhf", geom_file=xyz, orb_init=seed, SP=2, ngrid=[1, 8])
    st = capture_phf_args(js); z0 = st["z0"]; m = len(z0) // 2
    rng = np.random.default_rng(3)
    x = z0 + 0.05 * rng.standard_normal(len(z0)) * np.r_[np.ones(m), np.zeros(m)]
    errs = _fd_block_errors(st, x)
    assert errs[("real", "same-spin")][0] < 1e-6 and errs[("real", "same-spin")][1] > 1e-3
    assert errs[("real", "spin-flip")][0] < 1e-9 and errs[("real", "spin-flip")][1] < 1e-9


def test_nonorthonormal_seed_is_orthonormalised_with_warning(capsys):
    """A deliberately non-orthonormal (but full-rank) orb_init is Lowdin-fixed; the reference used is unitary."""
    import pickle
    xyz = write_xyz(H4_ATOMS, "H4sq")
    seed, mol = make_seed(H4_ATOMS, "cc-pvdz", "H4_uhf")
    C = pickle.load(open(seed, "rb")); n = mol.nelectron
    C[:, n:] += 0.05 * C[:, :n] @ np.ones((n, C.shape[1] - n))        # virtuals contaminated with occupied
    C[:, :n] *= 1.1                                                     # occupied not normalised
    bad = seed.replace(".p", "_bad.p"); pickle.dump(C, open(bad, "wb"))
    js = write_settings("H4_badseed", geom_file=xyz, orb_init=bad, SP=2, ngrid=[1, 8])
    st = capture_phf_args(js)
    out = capsys.readouterr().out
    assert "WARNING: orb_init was not orthonormal" in out
    MOs = st["eg_args"][2]
    assert np.abs(MOs.conj().T @ MOs - np.eye(MOs.shape[0])).max() < 1e-10


def test_rank_deficient_seed_is_rejected():
    import pickle
    xyz = write_xyz(H4_ATOMS, "H4sq")
    seed, mol = make_seed(H4_ATOMS, "cc-pvdz", "H4_uhf")
    C = pickle.load(open(seed, "rb")); C[:, 1] = C[:, 0]                # two identical occupied orbitals
    bad = seed.replace(".p", "_rankdef.p"); pickle.dump(C, open(bad, "wb"))
    js = write_settings("H4_rankdef", geom_file=xyz, orb_init=bad, SP=2, ngrid=[1, 8])
    with pytest.raises(ValueError, match="rank|linearly dependent"):
        capture_phf_args(js)


def test_invalid_optimizer_results_are_rejected(h4_sghf_complex):
    """Post-solve validation must reject NaN parameters, diverged parameters, and an unconverged point,
    independent of the optimizer's own success flag."""
    st = h4_sghf_complex; M = st["M"]; args = st["eg_args"]; z0 = st["z0"]; V = st["Values"]
    E0 = M.EandG(z0, *args)[0]
    fake = lambda x: OptimizeResult(x=np.asarray(x, float), fun=0.0, success=True, status=0, nfev=1)
    with pytest.raises(RuntimeError, match="nonfinite"):
        M.validate_phf_result(fake(np.where(np.arange(len(z0)) == 0, np.nan, z0)), E0, args, 1e-6, "test", V.Enuc, 0)
    with pytest.raises(RuntimeError, match="diverged"):
        M.validate_phf_result(fake(z0 + 10.0), E0, args, 1e-6, "test", V.Enuc, 0)
    with pytest.raises(RuntimeError, match=r"max\|G\|"):
        M.validate_phf_result(fake(z0), E0, args, 1e-9, "test", V.Enuc, 0)       # start point: gradient not small
    with pytest.raises(RuntimeError, match="energy rose"):
        M.validate_phf_result(fake(z0), E0 - 1.0, args, 1e3, "test", V.Enuc, 0)  # claimed start energy lower


def test_sghf_kick_applied_for_collinear_reference_only():
    xyz = write_xyz(H4_ATOMS, "H4sq")
    seed_col, _ = make_seed(H4_ATOMS, "cc-pvdz", "H4_uhf")
    seed_cant, _ = make_seed(H4_ATOMS, "cc-pvdz", "H4_canted2", cant_deg=2.0)
    st_col = capture_phf_args(write_settings("H4_kick_col", geom_file=xyz, orb_init=seed_col, SP=1, ngrid=[6, 6]))
    st_cant = capture_phf_args(write_settings("H4_kick_cant", geom_file=xyz, orb_init=seed_cant, SP=1, ngrid=[6, 6]))
    same, flip = spin_blocks(st_col["Values"]); m = len(st_col["z0"]) // 2
    assert np.abs(st_col["z0"][:m][flip]).max() > 1e-3          # collinear reference -> kicked
    assert np.abs(st_col["z0"][:m][same]).max() == 0.0
    assert np.abs(st_cant["z0"]).max() == 0.0                    # noncollinear reference -> no kick


def test_n2_suhf_energy_auto_ipopt():
    """N2 1.5 A cc-pVDZ SUHF via the default policy (IPOPT): energy and independently recomputed gradient."""
    xyz = write_xyz(N2_ATOMS, "N2")
    seed, _ = make_seed(N2_ATOMS, "cc-pvdz", "N2_uhf")
    js = write_settings("N2_suhf", geom_file=xyz, orb_init=seed, SP=2, ngrid=[1, 10])
    E, gmax, txt = run_driver(js, timeout=1200)
    assert "PHF optimizer: IPOPT" in txt
    assert abs(E - (-108.8677272779)) < 2e-7, E
    assert gmax < 1e-5, gmax


@SLOW
def test_n2_sghf_energy_auto_bfgs_hops():
    xyz = write_xyz(N2_ATOMS, "N2")
    seed, _ = make_seed(N2_ATOMS, "cc-pvdz", "N2_canted2", cant_deg=2.0)
    js = write_settings("N2_sghf", geom_file=xyz, orb_init=seed, SP=1, ngrid=[14, 10])
    E, gmax, txt = run_driver(js, timeout=7200)
    assert "PHF optimizer: BFGS + 3 basin hop(s)" in txt
    assert abs(E - (-108.8987192550)) < 2e-6, E
    assert gmax < 1e-4, gmax


@SLOW
def test_h4_sghf_energy_auto_bfgs_hops():
    xyz = write_xyz(H4_ATOMS, "H4sq")
    seed, _ = make_seed(H4_ATOMS, "cc-pvdz", "H4_canted2", cant_deg=2.0)
    js = write_settings("H4_sghf", geom_file=xyz, orb_init=seed, SP=1, ngrid=[14, 10])
    E, gmax, txt = run_driver(js, timeout=3600)
    assert "PHF optimizer: BFGS + 3 basin hop(s)" in txt
    assert abs(E - (-2.0754973503)) < 2e-6, E
    assert gmax < 1e-4, gmax
