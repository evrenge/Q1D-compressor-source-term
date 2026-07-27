"""Gate for similarity scaling — the fix for the high-pressure-ratio blockage.

``Fx`` and ``SWx`` are computed from the *lagged* sample. Injecting them as a
fixed force in newtons and a fixed heat rate in watts freezes the dimensional
**level** along with the operating point, and that contradicts the map: a map
asserts a pressure *ratio* and a corrected work, both invariant to the absolute
pressure level and to the mass flow. The consequences are measurable and were
measured (``PLAN.md`` §3.16): a fixed-force device is a rising branch from
PR 2.26, and a fixed-rate device runs away because ``Δh₀ = Ẇ/W`` grows as the
flow falls.

Two properties have to hold together, and they are what this file pins:

* **inert at the design point** — both scale ratios are one at any steady state,
  so no Phase 3 result moves. If this failed the fix would be buying stability
  by changing the answer, which is worthless.
* **decisive above PR 2.3** — ``SubsonicCompressor`` at Nc 1.0 is PR 2.141,
  above the threshold. Unscaled it never reaches the gate; scaled it holds to
  ~1e-10. The negative half is deliberate: without it a later regression that
  quietly reverted the injection would leave every test passing.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from q1d.analytic import state_from_flux
from q1d.boundary import StagnationInletStaticOutlet
from q1d.compressor import InletFlowCompressor, LocalLevelFilter
from q1d.design import design_from_map
from q1d.gas import PerfectGas
from q1d.grid import Grid
from q1d.maps import load_beta_map
from q1d.solver import NonPhysicalState, ReferenceState, Solver, SolverConfig

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

GAS = PerfectGas(1.4, 1004.7)
DATA = Path(__file__).resolve().parents[1] / "data" / "maps"
MAP = DATA / "SubsonicCompressor.xlsx"

#: Nc 1.0 on this map is PR 2.141 at the mid-line point — above the PR 2.26
#: threshold in normalised terms and, more to the point, measurably unstable
#: without the fix (largest eigenvalue +3.91, against −65.9 with it).
SPEED = 1.0


class TestLocalLevelFilter:
    """The reference level the scaling divides by."""

    def test_zero_tau_is_a_pass_through(self):
        f = LocalLevelFilter(0.0)
        p, m = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        rp, rm = f.update(0.0, p, m)
        assert rp is p and rm is m

    def test_first_call_seeds_from_the_field(self):
        """Ratios must be exactly one on the first evaluation of a seeded run."""
        f = LocalLevelFilter(1e-2)
        p, m = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        rp, rm = f.update(0.0, p, m)
        assert np.array_equal(rp, p) and np.array_equal(rm, m)
        assert rp is not p, "must copy, or the reference aliases the live field"

    def test_it_advances_once_per_step_not_once_per_stage(self):
        """A repeated ``t`` is a Runge-Kutta stage, not a new step."""
        f = LocalLevelFilter(1e-2)
        f.update(0.0, np.array([1.0]), np.array([1.0]))
        for _ in range(5):
            rp, _ = f.update(0.0, np.array([2.0]), np.array([2.0]))
            assert rp[0] == pytest.approx(1.0)
        rp, _ = f.update(1e-2, np.array([2.0]), np.array([2.0]))
        assert 1.0 < rp[0] < 2.0

    def test_unit_dc_gain(self):
        """Held input, the reference converges to it — so a steady state is fixed."""
        f = LocalLevelFilter(1e-3)
        f.update(0.0, np.array([1.0]), np.array([1.0]))
        t = 0.0
        for _ in range(200):
            t += 1e-4
            rp, _ = f.update(t, np.array([2.0]), np.array([5.0]))
        assert rp[0] == pytest.approx(2.0, rel=1e-6)

    def test_reset_forgets_the_reference(self):
        f = LocalLevelFilter(1e-2)
        f.update(0.0, np.array([1.0]), np.array([1.0]))
        f.reset()
        assert f._p is None and math.isnan(f._t)


def build(speed=SPEED, scaling=True, ncell=201, n_smear=21, offset=12, lag=1e-2,
          cfl=2.0, inlet_mach=0.45):
    """Duct sized from the map's mid-line point, seeded with the steady profile."""
    m = load_beta_map(MAP, GAS).densify(9)
    ecmf = m._speed_line(speed)[0]
    mid = 0.5 * (float(ecmf.min()) + float(ecmf.max()))
    d = design_from_map(m, speed, mid, inlet_mach=inlet_mach)
    A = d.area
    grid = Grid.uniform(0.0, 1.0, ncell, A)
    cell = (ncell - n_smear) // 2
    bc = StagnationInletStaticOutlet(p0_in=d.p01, T0_in=d.T01, p_back=d.p_back)
    ref = ReferenceState(rho=d.station1.rho, u=d.station1.u, p=d.station1.p)
    disk = InletFlowCompressor(
        cell=cell, beta_map=m, corrected_speed=speed, sample_offset=offset,
        n_smear=n_smear, inlet_lag=lag, similarity_scaling=scaling,
    )
    s = Solver(grid, GAS, bc, ref, config=SolverConfig(cfl=cfl), source=disk)

    rho, u, p = np.empty(ncell), np.empty(ncell), np.empty(ncell)
    rho[:cell], u[:cell], p[:cell] = d.station1.rho, d.station1.u, d.station1.p
    rho[cell + n_smear:] = d.station2.rho
    u[cell + n_smear:] = d.station2.u
    p[cell + n_smear:] = d.station2.p
    F = np.array([
        d.station1.rho * d.station1.u * A,
        (d.station1.rho * d.station1.u**2 + d.station1.p) * A,
        d.station1.rho * d.station1.u * (GAS.cp * d.station1.T + 0.5 * d.station1.u**2) * A,
    ])
    step = np.array([0.0, d.Fx, d.SWx]) / n_smear
    for k in range(n_smear):
        x = state_from_flux(F + 0.5 * step, A, GAS)
        rho[cell + k], u[cell + k], p[cell + k] = x.rho, x.u, x.p
        F = F + step
    s.set_state(rho, u, p)
    return s, disk, d


def test_the_map_reaches_a_pressure_ratio_worth_testing():
    """Guards the premise: below PR ~2.3 the two forms behave the same."""
    _, _, d = build()
    assert d.point.PR > 2.0, f"PR {d.point.PR} is below the interesting range"


def test_scaling_is_inert_at_the_design_point():
    """Both ratios are one at a steady state, so the operating point cannot move.

    Evaluated on the seeded field, where the level filter has just taken its
    reference from that same field. Any difference here would be a change to the
    converged answer, which no amount of stability would justify.
    """
    on, disk_on, _ = build(scaling=True)
    off, disk_off, _ = build(scaling=False)
    q_on, q_off = on.source(on), off.source(off)
    assert np.allclose(q_on, q_off, rtol=1e-12, atol=0.0)
    assert disk_on.last.Fx == pytest.approx(disk_off.last.Fx, rel=1e-12)
    assert disk_on.last.SWx == pytest.approx(disk_off.last.SWx, rel=1e-12)


def test_scaling_conserves_the_total_source_at_the_design_point():
    """The smeared total must still be exactly the map's ``Fx`` and ``SWx``."""
    s, disk, _ = build(scaling=True)
    q = s.source(s)
    assert q[1].sum() == pytest.approx(disk.last.Fx, rel=1e-12)
    assert q[2].sum() == pytest.approx(disk.last.SWx, rel=1e-12)


def _march(scaling, steps):
    s, disk, d = build(scaling=scaling)
    for k in range(steps):
        try:
            s.advance(s.timestep())
        except (NonPhysicalState, ValueError):
            return None, d, k
    return disk.last.W / d.W - 1.0, d, steps


@pytest.mark.slow
def test_similarity_scaling_holds_the_operating_point_above_pr_2():
    """The gate. 201 cells, real map, seeded steady, then held."""
    err, d, _ = _march(True, 18_000)
    assert err is not None, f"scaled run failed at PR {d.point.PR:.3f}"
    assert abs(err) < 1e-6, f"PR {d.point.PR:.3f}: W off by {err:.3e}"


@pytest.mark.slow
def test_the_whole_pressure_ratio_fits_in_one_node():
    """A single-cell disk, carrying the entire pressure ratio.

    This is the property the smear was hiding. Spread over 21 cells, PR 2.14 is
    only 1.037 per node; in one cell it is 2.14, and that is what lets a map
    move over its speed line without the node count having to change with it.

    It works only because the level reference sits one cell *upstream* of the
    forced cell. Reading it from the forced cell is self-referential and the
    loop gain is of order ``(PR−1)/(2·n_smear)``, which a single cell cannot
    afford: on ``HPC01`` at PR 5.040 that form gives ``max Re(λ) = +88.3`` and
    dies at step 513, against −92.3 and a 8.6e−12 hold for this one
    (``PLAN.md`` §3.20).
    """
    s, disk, d = build(scaling=True, n_smear=1)
    for k in range(18_000):
        try:
            s.advance(s.timestep())
        except (NonPhysicalState, ValueError) as ex:  # pragma: no cover
            pytest.fail(f"single-node disk at PR {d.point.PR:.3f} failed at step {k}: {ex}")
    err = disk.last.W / d.W - 1.0
    assert abs(err) < 1e-6, f"PR {d.point.PR:.3f} in one node: W off by {err:.3e}"


def test_the_level_reference_is_read_upstream_of_the_forced_cells():
    """Pins the shift itself, not just its consequences.

    Cheap and direct: perturbing the cell immediately upstream of a single-cell
    disk must change the source, and perturbing the forced cell itself must not.
    A refactor that "simplified" the reference back onto the forced cell would
    pass every other test in this file until it met a narrow smear.
    """
    s, disk, _ = build(scaling=True, n_smear=1)
    base = s.source(s)[1].sum()
    cell = disk.cell

    up = s.cv.copy()
    s.cv[2, cell] *= 1.0001  # interior index `cell-1` -> full-array index `cell`
    s._update_pressure()
    s._sync_boundaries()
    moved = s.source(s)[1].sum()
    s.cv[:] = up
    s._update_pressure()
    s._sync_boundaries()

    s.cv[2, cell + 1] *= 1.0001  # the forced cell itself
    s._update_pressure()
    s._sync_boundaries()
    unmoved = s.source(s)[1].sum()

    assert moved != base, "source ignores the cell upstream of the disk"
    assert unmoved == base, "source responds to the cell it is forcing"


@pytest.mark.slow
def test_without_scaling_the_same_case_does_not_hold():
    """The other half of the gate: assert the old injection *fails*.

    Without this a regression that reverted to fixed-force injection would leave
    every other test in this file passing, because they all check invariance at
    the design point — which the broken form also satisfies. Phase 3 shipped a
    gate with exactly that shape and it let four phases of regression through
    (``PLAN.md`` §3.10).
    """
    err, d, reached = _march(False, 18_000)
    if err is None:
        return  # went non-physical, which is a failure and is the point
    assert abs(err) > 1e-6, (
        f"unscaled injection held to {err:.3e} at PR {d.point.PR:.3f}; if this is "
        "genuine the threshold has moved and §3.16 needs remeasuring"
    )
