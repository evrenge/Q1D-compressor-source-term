"""Gate for the residual-driven β closure — the fix for the refused speed lines.

``InletFlowCompressor`` inverts the map on inlet corrected flow, and that inverse
does not exist on 13 of the 45 tabulated speed lines across the four supplied
compressor maps (``PLAN.md`` §3.26). The cause is rank deficiency: at Nc 1.144 on
``TranssonicCompressor`` the whole β range spans 9.97e−03 in ``Wc`` against
4.94e−01 in ``PR``.

:class:`FlowMatchedCompressor` never forms that inverse. β is a state relaxed on
the flow *residual*, scaled by the map's own ECMF slope — the only one of the
three candidate slopes that neither vanishes nor reverses anywhere on the
supplied maps. Three properties have to hold together, and they are what this
file pins:

* **it solves the same equation** — at steady state ``Wc_meas = Wc_map(β)``, so a
  line the inverse can do must give the identical answer. If it did not, this
  would be a different machine rather than a different solver.
* **the fixed point attracts** — seeded 0.2 away in β it walks back. Without this
  the first test only shows the closure is inert where it was seeded.
* **it works where the inverse refuses** — the point of the exercise. The
  negative half is deliberate: the test asserts that ``evaluate_at_Wc`` really
  does raise on the line being used, so the case cannot quietly become an
  ordinary one when a map or a densification setting changes.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from q1d.analytic import state_from_flux
from q1d.boundary import StagnationInletStaticOutlet
from q1d.compressor import CompositeSource, FlowMatchedCompressor
from q1d.design import design_from_map
from q1d.gas import PerfectGas
from q1d.grid import Grid
from q1d.maps import ScaledMap, load_beta_map
from q1d.solver import NonPhysicalState, ReferenceState, Solver, SolverConfig

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

GAS = PerfectGas(1.4, 1004.7)
DATA = Path(__file__).resolve().parents[1] / "data" / "maps"

#: Nc 1.0 here is PR 2.14 at mid-line and the inverse handles it — the control.
INVERTIBLE = (DATA / "SubsonicCompressor.xlsx", 1.0)
#: Nc 0.88 here is vertical to within the tabulation; the inverse refuses it.
REFUSED = (DATA / "TranssonicCompressor.xlsx", 0.88)


def build(path, speed, frac=0.5, seed_offset=0.0, ncell=201, n_smear=7, offset=2,
          lag=1e-2, cfl=2.0, inlet_mach=0.45, **kw):
    """Duct sized from a point along the speed line, seeded with its steady profile."""
    m = load_beta_map(path, GAS).densify(9)
    ecmf = m._speed_line(speed)[0]
    lo, hi = float(ecmf.min()), float(ecmf.max())
    d = design_from_map(m, speed, lo + frac * (hi - lo), inlet_mach=inlet_mach)
    A = d.area
    grid = Grid.uniform(0.0, 1.0, ncell, A)
    cell = (ncell - n_smear) // 2
    disk = FlowMatchedCompressor(
        cell=cell, beta_map=m, corrected_speed=speed, sample_offset=offset,
        n_smear=n_smear, inlet_lag=lag,
        beta0=float(np.clip(d.point.beta + seed_offset, 0.0, 1.0)), **kw,
    )
    s = Solver(
        grid, GAS,
        StagnationInletStaticOutlet(p0_in=d.p01, T0_in=d.T01, p_back=d.p_back),
        ReferenceState(rho=d.station1.rho, u=d.station1.u, p=d.station1.p),
        config=SolverConfig(cfl=cfl), source=disk,
    )

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


def march(s, disk, d, steps, tol=1e-11):
    """Run to residual convergence; return the mass-flow error and converged β."""
    for k in range(steps):
        try:
            s.advance(s.timestep())
        except (NonPhysicalState, ValueError) as ex:  # pragma: no cover
            pytest.fail(f"PR {d.point.PR:.3f} failed at step {k}: {ex}")
        if k % 500 == 0 and k > 0:
            if float(s.residual_norm()[0]) / d.W * s.grid.dx.mean() < tol:
                break
    return float(s.face_fluxes()[0].mean()) / d.W - 1.0, disk.beta


class TestConstruction:
    def test_n_smear_must_be_positive(self):
        m = load_beta_map(INVERTIBLE[0], GAS)
        with pytest.raises(ValueError, match="n_smear"):
            FlowMatchedCompressor(cell=10, beta_map=m, corrected_speed=1.0, n_smear=0)

    def test_the_disk_may_not_sample_itself(self):
        m = load_beta_map(INVERTIBLE[0], GAS)
        with pytest.raises(ValueError, match="sample_offset"):
            FlowMatchedCompressor(cell=10, beta_map=m, corrected_speed=1.0, sample_offset=0)

    def test_gain_and_tau_must_be_positive(self):
        m = load_beta_map(INVERTIBLE[0], GAS)
        with pytest.raises(ValueError, match="gain"):
            FlowMatchedCompressor(cell=10, beta_map=m, corrected_speed=1.0, gain=0.0)
        with pytest.raises(ValueError, match="tau"):
            FlowMatchedCompressor(cell=10, beta_map=m, corrected_speed=1.0, tau=-1.0)

    def test_it_does_not_refuse_a_line_the_inverse_cannot_do(self):
        """The whole point: construction must not depend on invertibility."""
        m = load_beta_map(REFUSED[0], GAS).densify(9)
        assert not m.inlet_closure_is_invertible(REFUSED[1])
        FlowMatchedCompressor(cell=10, beta_map=m, corrected_speed=REFUSED[1])


class TestBetaState:
    def test_beta_starts_where_it_was_seeded(self):
        s, disk, d = build(*INVERTIBLE, seed_offset=0.15)
        s.source(s)
        assert disk.beta == pytest.approx(d.point.beta + 0.15, abs=1e-12)

    def test_beta_advances_once_per_step_not_once_per_stage(self):
        """A repeated ``t`` is a Runge-Kutta stage. Advancing per stage would
        multiply the effective gain by five and is not visible in a converged run."""
        s, disk, _ = build(*INVERTIBLE, seed_offset=0.15)
        s.source(s)
        first = disk.beta
        for _ in range(4):
            s.source(s)
            assert disk.beta == first
        s.t += 1e-5
        s.source(s)
        assert disk.beta != first

    @pytest.mark.parametrize("path", sorted(DATA.glob("*.xlsx")))
    def test_only_the_ecmf_slope_is_safe_to_divide_by(self, path):
        """Guards the one decision that makes a vertical speed line workable.

        The step is a damped Newton step, so the denominator is a slope, and the
        choice of *which* slope is the whole design. Measured here on every
        tabulated line of every supplied map:

        * ``∂lnWc/∂β`` reverses sign — that is what stops ``evaluate_at_Wc``
          inverting the refused lines, and Newton on it is singular, not merely
          stiff;
        * ``∂lnPR/∂β`` reverses too, at the **surge peak**, on 20 of the 45
          lines. It drives β the wrong way past the peak and killed
          ``TwoStgRadialCompr`` Nc 0.600 near surge, a point the inverse holds;
        * ``∂lnECMF/∂β`` never reverses and never falls below 0.225 — the
          documented property ECMF was introduced for.
        """
        raw = load_beta_map(path, GAS)
        m = raw.densify(9)
        floor, reversals = np.inf, {"Wc": 0, "PR": 0}
        speeds = sorted({float(x) for x in raw.corrected_speed.ravel()})
        for nc in speeds:
            e, wc, pr, _, _ = m._speed_line(nc)
            d_e = np.gradient(np.log(e), m.beta)
            assert not (d_e[:-1] * d_e[1:] < 0.0).any(), (
                f"{path.stem} Nc {nc}: ECMF slope reverses; the scale is unsafe"
            )
            floor = min(floor, float(np.abs(d_e).min()))
            for key, a in (("Wc", wc), ("PR", pr)):
                d = np.gradient(np.log(a), m.beta)
                reversals[key] += bool((d[:-1] * d[1:] < 0.0).any())
        assert floor > 0.2, f"{path.stem}: |dlnECMF/dbeta| falls to {floor:.3g}"
        assert reversals["Wc"] or reversals["PR"], (
            f"{path.stem}: premise gone — neither Wc nor PR reverses on any line, "
            f"so this map no longer demonstrates why ECMF is needed"
        )


@pytest.mark.slow
class TestItSolvesTheSameClosure:
    """On a line the inverse can do, the answer must be the inverse's answer."""

    @pytest.mark.parametrize("frac", [0.15, 0.5, 0.85])
    def test_converged_beta_is_the_design_beta(self, frac):
        s, disk, d = build(*INVERTIBLE, frac=frac)
        err, beta = march(s, disk, d, 40_000)
        assert abs(err) < 1e-6, f"W off by {err:.3e}"
        assert beta == pytest.approx(d.point.beta, abs=1e-7), (
            f"landed on beta {beta:.6f}, design {d.point.beta:.6f}"
        )

    @pytest.mark.parametrize("offset", [-0.2, +0.2])
    def test_a_displaced_seed_walks_back(self, offset):
        """Without this the first test only shows inertness at the seed."""
        s, disk, d = build(*INVERTIBLE, frac=0.5, seed_offset=offset)
        err, beta = march(s, disk, d, 40_000)
        assert abs(err) < 1e-6, f"W off by {err:.3e}"
        assert beta == pytest.approx(d.point.beta, abs=1e-7)


@pytest.mark.slow
class TestItWorksWhereTheInverseRefuses:
    def test_the_inverse_really_does_refuse_this_line(self):
        m = load_beta_map(REFUSED[0], GAS).densify(9)
        with pytest.raises(ValueError, match="not monotonic"):
            m.evaluate_at_Wc(float(m._speed_line(REFUSED[1])[1].mean()), REFUSED[1])

    @pytest.mark.parametrize("frac", [0.15, 0.5, 0.85])
    def test_it_holds_the_operating_point_anyway(self, frac):
        s, disk, d = build(*REFUSED, frac=frac)
        err, beta = march(s, disk, d, 40_000)
        assert abs(err) < 1e-6, f"PR {d.point.PR:.3f}: W off by {err:.3e}"
        assert 0.0 <= beta <= 1.0
        assert disk.reversals == 0


class TestScaledMap:
    """Stage stacking: the same shape sized to a stage's own inlet flow."""

    def test_the_dimensionless_quantities_pass_through_untouched(self):
        m = load_beta_map(INVERTIBLE[0], GAS).densify(9)
        s = ScaledMap(m, 0.4)
        wc = float(m._speed_line(1.0)[1].mean())
        a, b = m.evaluate_at_Wc(wc, 1.0), s.evaluate_at_Wc(wc * 0.4, 1.0)
        assert b.PR == pytest.approx(a.PR, rel=1e-14)
        assert b.corrected_work == pytest.approx(a.corrected_work, rel=1e-14)
        assert b.beta == pytest.approx(a.beta, rel=1e-14)

    def test_the_flow_quantities_come_back_in_the_stage_s_own_units(self):
        m = load_beta_map(INVERTIBLE[0], GAS).densify(9)
        s = ScaledMap(m, 0.4)
        wc = float(m._speed_line(1.0)[1].mean())
        a, b = m.evaluate_at_Wc(wc, 1.0), s.evaluate_at_Wc(wc * 0.4, 1.0)
        assert b.Wc == pytest.approx(0.4 * a.Wc, rel=1e-14)
        assert b.ecmf == pytest.approx(0.4 * a.ecmf, rel=1e-14)

    def test_the_speed_line_scales_with_it(self):
        """`FlowMatchedCompressor` reads the line directly, so it must scale too."""
        m = load_beta_map(INVERTIBLE[0], GAS).densify(9)
        e0, wc0, pr0, cw0, ef0 = m._speed_line(1.0)
        e1, wc1, pr1, cw1, ef1 = ScaledMap(m, 0.4)._speed_line(1.0)
        assert np.allclose(wc1, 0.4 * wc0, rtol=1e-14)
        assert np.allclose(e1, 0.4 * e0, rtol=1e-14)
        assert np.array_equal(pr1, pr0) and np.array_equal(cw1, cw0)
        assert np.array_equal(ef1, ef0)

    def test_a_unit_scale_is_the_identity(self):
        m = load_beta_map(INVERTIBLE[0], GAS).densify(9)
        wc = float(m._speed_line(1.0)[1].mean())
        assert ScaledMap(m, 1.0).evaluate_at_Wc(wc, 1.0) == m.evaluate_at_Wc(wc, 1.0)

    def test_a_non_positive_scale_is_refused(self):
        m = load_beta_map(INVERTIBLE[0], GAS)
        with pytest.raises(ValueError, match="scale"):
            ScaledMap(m, 0.0)


class TestCompositeSource:
    """Several components on one duct — how an engine gets assembled."""

    def test_it_sums_its_members(self):
        s, disk, _ = build(*INVERTIBLE)
        one = disk(s)
        assert np.allclose(CompositeSource(disk, disk)(s), 2.0 * one, rtol=1e-14)

    def test_it_accepts_a_sequence_as_well_as_varargs(self):
        s, disk, _ = build(*INVERTIBLE)
        assert np.allclose(CompositeSource([disk, disk])(s), CompositeSource(disk, disk)(s))

    def test_an_empty_composite_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            CompositeSource()

    def test_disjoint_members_do_not_overlap(self):
        """Two disks at different cells must write to different cells."""
        m = load_beta_map(INVERTIBLE[0], GAS).densify(9)
        s, disk, _ = build(*INVERTIBLE, n_smear=1)
        other = FlowMatchedCompressor(
            cell=disk.cell + 20, beta_map=m, corrected_speed=1.0,
            sample_offset=2, n_smear=1, beta0=disk.beta,
        )
        q = CompositeSource(disk, other)(s)
        assert q[1, disk.cell] != 0.0 and q[1, disk.cell + 20] != 0.0
        assert math.isclose(q[1].sum(), disk.last.Fx + other.last.Fx, rel_tol=1e-12)
