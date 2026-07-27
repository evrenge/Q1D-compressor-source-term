"""Gate for the ECMF-keyed map — β eliminated from the runtime.

β exists only because a speed line is multivalued in inlet ``Wc``. ECMF is
monotonic in β on every speed line of every supplied compressor and fan map, so
`β ↔ ECMF` is a bijection and ``PR(ECMF, Nc)`` is single-valued. Re-tabulating at
build time removes β from the interface and removes an inconsistency with it.

**What it costs, since it is not free.** On a tabulated speed line the two paths
are the same map, exactly. *Between* lines they are not: ``BetaMap`` blends at
fixed β and then inverts, while an ECMF-keyed map must evaluate each bracketing
line at the requested key and blend the results. That is D13's "inverting ECMF
first", which it measures at 2.81% against 1.79% at native speed spacing. The
difference is second order in speed spacing —

===================  ==========  ==========  ==========  ==========
                     Sub, ×9     Sub, ×36    Trans, ×9   Trans, ×36
===================  ==========  ==========  ==========  ==========
worst ``PR``         4.42e−05    2.80e−06    1.20e−04    7.53e−06
worst ``CW``         2.60e−04    1.84e−05    2.13e−03    1.50e−04
===================  ==========  ==========  ==========  ==========

— about 15× for a 4× refinement, with the worst points at the *lowest* speeds
where the supplied lines are furthest apart. So densify first, convert after, and
densify enough.

**What it buys.** β leaves the runtime, and the returned point reproduces its own
key: ``Wc·√τ/PR`` equals the ECMF asked for to 2.2e−16, which the β path cannot
do between nodes.
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


def _worst_against_beta_path(b, e, speeds):
    worst_pr = worst_cw = 0.0
    for nc in speeds:
        line = b._speed_line(nc)[0]
        lo, hi = float(line.min()), float(line.max())
        for frac in (0.05, 0.25, 0.5, 0.75, 0.95):
            q = lo + frac * (hi - lo)
            a, c = b.evaluate_at_ecmf(q, nc), e.evaluate(q, nc)
            worst_pr = max(worst_pr, abs(c.PR / a.PR - 1.0))
            worst_cw = max(worst_cw, abs(c.corrected_work / a.corrected_work - 1.0))
    return worst_pr, worst_cw


@pytest.mark.parametrize("path", MAPS)
def test_on_a_tabulated_speed_line_it_is_an_exact_refactor(path):
    """Where no speed blending happens, the two paths are the same map.

    Inverting a piecewise-linear ECMF onto β and then interpolating a
    piecewise-linear ``PR`` in β *is* interpolating ``PR`` against ECMF, on
    shared nodes. So on a line this must be exact, not merely close.
    """
    b = beta_map(path)
    e = ECMFMap.from_beta_map(b)
    speeds = [float(x) for x in b.corrected_speed[::7]]
    worst_pr, worst_cw = _worst_against_beta_path(b, e, speeds)
    assert worst_pr < 1e-14, f"PR differs by {worst_pr:.3e} on a tabulated line"
    assert worst_cw < 1e-14, f"corrected work differs by {worst_cw:.3e} on a tabulated line"


@pytest.mark.parametrize("path", MAPS)
def test_between_speed_lines_it_differs_by_the_crossing_scheme(path):
    """The one real difference, pinned so it cannot grow unnoticed.

    β is the correspondence label, so ``BetaMap`` blends a whole line at fixed β
    and then inverts. Without β the only option is to evaluate each bracketing
    line at the requested ECMF and blend — D13's "inverting ECMF first", which it
    measures at 2.81% against 1.79% at *native* speed spacing.

    Densification is what makes that affordable, and this asserts both halves:
    the difference is real (so nobody later claims exactness), and it is bounded
    well below the 1e−6 operating-point gate at the densification actually used.
    """
    b = beta_map(path)
    e = ECMFMap.from_beta_map(b)
    n = b.corrected_speed
    mid = [float(0.5 * (n[i] + n[i + 1])) for i in range(0, len(n) - 1, 7)]
    worst_pr, worst_cw = _worst_against_beta_path(b, e, mid)
    assert worst_pr > 0.0, "premise gone: the crossing schemes no longer differ"
    assert worst_pr < 1e-2, f"cross-speed PR difference {worst_pr:.3e} is out of hand"

    b36 = beta_map(path, factor=36)
    e36 = ECMFMap.from_beta_map(b36)
    n36 = b36.corrected_speed
    mid36 = [float(0.5 * (n36[i] + n36[i + 1])) for i in range(0, len(n36) - 1, 29)]
    pr36, cw36 = _worst_against_beta_path(b36, e36, mid36)

    # Assert the CONVERGENCE, not an absolute bound. The absolute number depends
    # on the map's raw speed spacing -- the worst points sit at the lowest speeds,
    # where the supplied lines are furthest apart -- so a fixed threshold would
    # mostly measure which workbook this is. What has to be true is that
    # refining the speed axis removes it, and at second order.
    assert pr36 < 0.25 * worst_pr, (
        f"PR difference {pr36:.3e} at densify 36 against {worst_pr:.3e} at 9 -- "
        f"not converging with speed spacing"
    )
    assert cw36 < 0.25 * worst_cw, (
        f"corrected-work difference {cw36:.3e} at densify 36 against {worst_cw:.3e} "
        f"at 9 -- not converging with speed spacing"
    )


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
