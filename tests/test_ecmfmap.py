"""Gate for the ECMF-keyed map — β eliminated from the runtime.

β exists only because a speed line is multivalued in inlet ``Wc``. ECMF is
monotonic in β on every speed line of every supplied compressor and fan map, so
`β ↔ ECMF` is a bijection and ``PR(ECMF, Nc)`` is single-valued. Re-tabulating at
build time removes β from the interface and removes an inconsistency with it.

**The inconsistency, and why it is worth a file of its own.**
:meth:`BetaMap.evaluate_at_ecmf` inverts a *precomputed* ECMF array, while the
caller forms ECMF from separately interpolated ``Wc``, ``PR`` and ``τ``. Between
nodes those two disagree at O(Δβ²) — they coincide only where the nodes are.
Measured end-to-end through the solver on ``HighPqPCompr`` Nc 0.700 f = 0.85 the
converged mass flow is off by 5.54e−06, 1.57e−06 and 2.69e−07 at densify 9, 18
and 36: observed order 2.18, and on ``SubsonicCompressor`` the error passes
through zero, so it is a convergent discretisation of the map rather than a bias.

Keyed on ECMF the question cannot arise, because the table is evaluated *at* the
measured key. That is what :func:`test_the_point_reproduces_its_own_key` pins,
and it holds to machine precision rather than to a tolerance.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from q1d.gas import PerfectGas
from q1d.maps import T_REF, BetaMap, ECMFMap, load_beta_map

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

GAS = PerfectGas(1.4, 1004.7)
DATA = Path(__file__).resolve().parents[1] / "data" / "maps"
MAPS = sorted(DATA.glob("*.xlsx"))


def beta_map(path, factor=9):
    return load_beta_map(path, GAS).densify(factor)


def sample_speeds(m: BetaMap, n=6):
    idx = np.linspace(0, len(m.corrected_speed) - 1, n).astype(int)
    return [float(m.corrected_speed[i]) for i in idx]


@pytest.mark.parametrize("path", MAPS)
class TestConstruction:
    def test_it_keeps_the_speed_axis(self, path):
        b = beta_map(path)
        e = ECMFMap.from_beta_map(b)
        assert np.array_equal(e.corrected_speed, b.corrected_speed)

    def test_the_key_ascends_down_every_column(self, path):
        e = ECMFMap.from_beta_map(beta_map(path))
        assert np.all(np.diff(e.ecmf, axis=0) > 0.0), "the key must be usable by np.interp"

    def test_it_carries_no_beta(self, path):
        e = ECMFMap.from_beta_map(beta_map(path))
        pt = e.evaluate(float(e.ecmf[:, 0].mean()), float(e.corrected_speed[0]))
        assert math.isnan(pt.beta), "beta must be absent, not silently zero"


@pytest.mark.parametrize("path", MAPS)
def test_the_point_reproduces_its_own_key(path):
    """The property the β path does not have, to machine precision.

    ``ECMF ≡ Wc·√τ/PR``. Ask the map for a point at some ECMF and the point it
    returns must satisfy that identity *at the ECMF asked for*. On the β path
    this holds only at nodes; here it holds everywhere, because the table is
    indexed by the key rather than inverted onto it.
    """
    b = beta_map(path)
    e = ECMFMap.from_beta_map(b)
    worst = 0.0
    for nc in sample_speeds(b):
        line = b._speed_line(nc)[0]
        lo, hi = float(line.min()), float(line.max())
        for frac in (0.05, 0.25, 0.5, 0.75, 0.95):
            q = lo + frac * (hi - lo)
            pt = e.evaluate(q, nc)
            tau = 1.0 + pt.corrected_work / (GAS.cp * T_REF)
            worst = max(worst, abs(pt.Wc * math.sqrt(tau) / pt.PR / q - 1.0))
    assert worst < 1e-13, f"key not reproduced: {worst:.3e}"


@pytest.mark.parametrize("path", MAPS)
def test_it_agrees_with_the_beta_path(path):
    """Same machine, different bookkeeping — so the answers must match.

    Loose on purpose: the two use different interpolants, and the *point* of the
    new one is that it does not carry the β path's O(Δβ²) inconsistency. This
    asserts they describe the same compressor, not that they are bit-identical.
    """
    b = beta_map(path)
    e = ECMFMap.from_beta_map(b)
    worst_pr = worst_cw = 0.0
    for nc in sample_speeds(b):
        line = b._speed_line(nc)[0]
        lo, hi = float(line.min()), float(line.max())
        for frac in (0.05, 0.25, 0.5, 0.75, 0.95):
            q = lo + frac * (hi - lo)
            a, c = b.evaluate_at_ecmf(q, nc), e.evaluate(q, nc)
            worst_pr = max(worst_pr, abs(c.PR / a.PR - 1.0))
            worst_cw = max(worst_cw, abs(c.corrected_work / a.corrected_work - 1.0))
    assert worst_pr < 1e-3, f"PR differs by {worst_pr:.3e}"
    assert worst_cw < 1e-3, f"corrected work differs by {worst_cw:.3e}"


@pytest.mark.parametrize("path", MAPS)
def test_it_clamps_rather_than_extrapolates(path):
    """§3.8 measured extrapolation as the worst error source on these maps."""
    b = beta_map(path)
    e = ECMFMap.from_beta_map(b)
    nc = sample_speeds(b)[len(sample_speeds(b)) // 2]
    line = b._speed_line(nc)[0]
    lo, hi = float(line.min()), float(line.max())
    assert e.evaluate(lo - 10.0 * (hi - lo), nc).PR == pytest.approx(
        e.evaluate(lo, nc).PR, rel=1e-9
    )
    assert e.evaluate(hi + 10.0 * (hi - lo), nc).PR == pytest.approx(
        e.evaluate(hi, nc).PR, rel=1e-9
    )


def test_a_turbine_is_refused_with_a_useful_message():
    """ECMF keys 135/135 compressor lines and only 7/22 turbine lines.

    A turbine wants PR, which is monotonic on 22/22. Refusing at build time with
    the reason beats returning a map that is quietly multivalued. Built from a
    synthetic non-monotonic map so the test does not depend on turbine files
    that are not in ``data/``.
    """
    b = beta_map(MAPS[0], factor=1)
    bad = BetaMap(
        name="synthetic-turbine",
        beta=b.beta,
        corrected_speed=b.corrected_speed,
        Wc=b.Wc,
        PR=b.PR,
        efficiency=b.efficiency,
        corrected_work=b.corrected_work,
        ecmf=b.ecmf.copy(),
        gas=b.gas,
    )
    bad.ecmf[len(bad.beta) // 2, :] = bad.ecmf[0, :]  # break monotonicity
    with pytest.raises(ValueError, match="not monotonic"):
        ECMFMap.from_beta_map(bad)


def test_n_key_must_be_usable():
    with pytest.raises(ValueError, match="n_key"):
        ECMFMap.from_beta_map(beta_map(MAPS[0], factor=1), n_key=1)
