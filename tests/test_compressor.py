"""Phase 3 gate — the Q1D solver must reproduce the 0D answer.

The gate is deliberately checked in more than one configuration. Three Phase 2
claims passed their tests and were still wrong, each because it was measured
where the problem was invisible, so the mass-flow match here is run across an
ambient matrix *and* several mesh densities *and* a long hold.

The gate also pins ``T02`` independently of ``p02``. With ``PR`` and ``eta``
both fixed, a disk that silently applied no efficiency at all would still
produce the right pressure rise — so a test that checked only ``p02`` would
pass on an isentropic disk, which is exactly the failure P1 exists to prevent.
"""

import math

import numpy as np
import pytest

from q1d.analytic import zero_d_compressor
from q1d.boundary import StagnationInletStaticOutlet
from q1d.compressor import (
    ActuatorDisk,
    ConstantCompressorMap,
    InletFilter,
    rotor_period,
)
from q1d.gas import AIR_LEGACY
from q1d.grid import Grid
from q1d.solver import ReferenceState, Solver, SolverConfig

GAS = AIR_LEGACY
PR, ETA, AREA = 1.2, 0.9, 0.1
P01, T01, PB = 101325.0, 288.15, 101325.0


def build(
    p01=P01,
    T01=T01,
    pb=PB,
    pr=PR,
    eta=ETA,
    n=99,
    n_smear=1,
    sample_offset=12,
    area=AREA,
    cfl=2.0,
    inlet_lag=0.0,
):
    """Duct with a compressor disk at mid-length, initialised at the 0D answer."""
    exact = zero_d_compressor(p01, T01, pb, pr, eta, area, area, GAS)
    grid = Grid.uniform(0.0, 1.0, n, area)
    bc = StagnationInletStaticOutlet(p0_in=p01, T0_in=T01, p_back=pb)
    reference = ReferenceState(rho=exact.station1.rho, u=exact.station1.u, p=p01)
    disk = ActuatorDisk(
        cell=n // 2,
        compressor_map=ConstantCompressorMap(pr, eta),
        sample_offset=sample_offset,
        n_smear=n_smear,
        inlet_lag=inlet_lag,
    )
    solver = Solver(grid, GAS, bc, reference, SolverConfig(cfl=cfl), source=disk)
    # start from the exact upstream state everywhere; the disk builds the rest
    solver.set_state(rho=exact.station1.rho, u=exact.station1.u, p=exact.station1.p)
    return solver, grid, disk, exact


def converged_mass_flow(solver, grid, tol=1e-12, max_steps=120_000):
    result = solver.run(max_steps=max_steps, tol=tol)
    return result, float(np.mean(solver.face_fluxes()[0]))


def stagnation(solver, grid, index):
    rho, u, p, c = solver.primitives()
    mach = u[index] / c[index]
    T = p[index] / (rho[index] * GAS.R)
    T0 = T * (1.0 + 0.5 * GAS.gm1 * mach**2)
    return float(T0), float(p[index] * (T0 / T) ** GAS.g_over_gm1)


# ===========================================================================
# The gate
# ===========================================================================


def test_matches_the_zero_d_reference_at_the_design_point():
    solver, grid, disk, exact = build()
    result, W = converged_mass_flow(solver, grid)
    assert result.converged, f"residual {result.final_residual}"
    assert W == pytest.approx(exact.W, rel=1e-8), f"W={W:.10f} vs {exact.W:.10f}"
    assert exact.W == pytest.approx(21.490742688146575, rel=1e-12)


def test_pressure_ratio_and_temperature_ratio_are_both_reproduced():
    """T02 is pinned independently, so an isentropic disk cannot pass.

    With PR and eta both constant, the pressure rise alone does not
    distinguish a correct disk from one that ignores efficiency entirely: only
    T02 carries eta. The isentropic T02 here is 303.558 K against the actual
    305.270 K — a gap of (1/eta - 1)(T02s - T01)/T02s = 0.564%, which is five
    orders of magnitude larger than the 1e-8 gate. A pressure-only check would
    pass on an isentropic disk; this one cannot.
    """
    solver, grid, disk, exact = build()
    assert converged_mass_flow(solver, grid)[0].converged

    T0_up, p0_up = stagnation(solver, grid, 10)
    T0_dn, p0_dn = stagnation(solver, grid, grid.n_interior - 10)

    assert p0_up == pytest.approx(P01, rel=1e-8)
    assert T0_up == pytest.approx(T01, rel=1e-8)
    assert p0_dn / p0_up == pytest.approx(PR, rel=1e-8)
    assert T0_dn == pytest.approx(exact.T02, rel=1e-8)

    # and the isentropic value is decisively excluded
    T02_isentropic = T01 * PR**GAS.gm1_over_g
    gap = abs(T0_dn - T02_isentropic) / T02_isentropic
    assert gap == pytest.approx(0.00564, rel=0.02)
    assert gap > 1e5 * 1e-8, "efficiency must be distinguishable far above the gate"


AMBIENT = [
    (101325.0, 288.15, 101325.0),
    (60000.0, 250.0, 60000.0),
    (150000.0, 320.0, 150000.0),
    (101325.0, 288.15, 99000.0),
    (80000.0, 260.0, 78000.0),
]


@pytest.mark.parametrize("p01,T0,pb", AMBIENT)
def test_matches_across_ambient_conditions(p01, T0, pb):
    """Sweeping ambient tests the *interface*, not the physics.

    For a perfect gas the dimensionless source terms are exactly inlet
    invariant (``PLAN.md`` §3.2, verified to 1e-15), so any failure here means
    the local state is being measured or applied wrongly — which is precisely
    the defect the prototype had.
    """
    solver, grid, disk, exact = build(p01=p01, T01=T0, pb=pb)
    result, W = converged_mass_flow(solver, grid)
    assert result.converged
    assert W == pytest.approx(exact.W, rel=1e-8), f"W={W:.10f} vs exact {exact.W:.10f}"


@pytest.mark.parametrize("n", [59, 99, 199])
def test_matches_across_mesh_densities(n):
    """The answer must not depend on the mesh.

    It should not, even slightly: upstream and downstream of the disk the flow
    is uniform, so the Roe dissipation vanishes and the only mesh-dependent
    region is the smear, across which conservation is exact.
    """
    solver, grid, disk, exact = build(n=n)
    result, W = converged_mass_flow(solver, grid)
    assert result.converged
    assert W == pytest.approx(exact.W, rel=1e-8), f"n={n}: W={W:.10f} vs {exact.W:.10f}"


def test_converged_operating_point_holds():
    """D10 hold test: reaching the point is not the same as staying there."""
    solver, grid, disk, exact = build()
    converge, W0 = converged_mass_flow(solver, grid)
    assert converge.converged

    hold = solver.run(max_steps=10 * max(converge.steps, 100))
    W1 = float(np.mean(solver.face_fluxes()[0]))
    assert abs(W1 / W0 - 1.0) < 1e-8, "mass flow drifted during the hold"
    assert np.all(hold.residual_history[-1] <= hold.residual_history[0] * 10.0)


# ===========================================================================
# The interface itself
# ===========================================================================


def test_disk_measures_the_upstream_state_it_is_supposed_to():
    solver, grid, disk, exact = build()
    assert converged_mass_flow(solver, grid)[0].converged
    assert disk.last.p01 == pytest.approx(P01, rel=1e-8)
    assert disk.last.T01 == pytest.approx(T01, rel=1e-8)
    assert disk.last.W == pytest.approx(exact.W, rel=1e-7)
    assert disk.last.phi1 == pytest.approx(exact.phi1, rel=1e-7)
    assert disk.last.Fx == pytest.approx(exact.Fx, rel=1e-6)
    assert disk.last.SWx == pytest.approx(exact.SWx, rel=1e-6)


def test_sampling_distance_barely_matters_now_that_the_station_reads_fluxes():
    """PLAN.md §7 Q1 said measurement would decide this. It did, then it changed.

    **The original finding.** The disk appeared to perturb the field *upstream*
    of itself — a numerical boundary layer from the reconstruction stencil —
    whose influence decayed by ~10x every two to three cells, so the default
    standoff was set to 12 cells to clear it. Measured ``p01`` error at the
    station: −1.9e−04 at offset 1, −1.1e−05 at 3, +3.6e−11 at 12.

    **What it actually was.** Not the field: the error of averaging a sharp
    profile over a cell. The scheme conserves *face fluxes*, and reading the
    station from those instead — :meth:`Solver.station_state_at` — is exact one
    cell from the disk: −8.9e−12 at offset 1, −9.6e−12 at 12, flat throughout,
    and it holds at PR 1.2/1.6/2.0 with and without a downstream taper
    (``PLAN.md`` §3.23).

    So this test now pins the opposite property to the one it was written for.
    That is worth roughly **ten cells per blade row** of mesh budget, which is
    the difference between an affordable engine model and an unaffordable one.

    ``sample_offset`` is kept as a parameter — there are physical reasons to
    stand a station off — but it is no longer paying for a numerical artefact.
    """
    errors = {}
    for offset in (1, 3, 5, 8, 12):
        solver, grid, disk, exact = build(sample_offset=offset)
        result, W = converged_mass_flow(solver, grid)
        assert result.converged, f"offset {offset} did not converge"
        errors[offset] = abs(W / exact.W - 1.0)

    # every standoff clears the gate now, including sampling right next door
    for offset, err in errors.items():
        assert err < 1e-9, f"offset {offset} missed the gate: {err:.3e}"

    # and they agree with each other: no decay left to see
    spread = max(errors.values()) - min(errors.values())
    assert spread < 1e-9, f"readings still depend on standoff: {errors}"


@pytest.mark.parametrize("n_smear", [1, 3, 7])
def test_smearing_does_not_move_the_operating_point(n_smear):
    """Spreading the source changes the local profile, not the balance.

    Global conservation is exact regardless of how the source is distributed,
    so the far-field states — and hence the mass flow — must be identical.
    """
    solver, grid, disk, exact = build(n_smear=n_smear)
    result, W = converged_mass_flow(solver, grid)
    assert result.converged
    assert W == pytest.approx(exact.W, rel=1e-8), f"n_smear {n_smear}: {W:.10f}"


def test_smearing_spreads_the_source_but_conserves_its_total():
    solver, grid, disk, exact = build(n_smear=5)
    q = disk(solver)
    assert np.count_nonzero(q[1]) == 5
    assert q[1].sum() == pytest.approx(disk.last.Fx, rel=1e-12)
    assert q[2].sum() == pytest.approx(disk.last.SWx, rel=1e-12)


def test_disk_rejects_a_varying_area_placement():
    """A zero-thickness disk has one area; see PLAN.md §4.5.

    A *monotone taper* is used deliberately. An earlier version of this test
    put the disk at cell 49 of a symmetric bump on a 99-cell mesh — which is
    the apex, where dA/dx = 0 and the two faces straddling the disk have equal
    area by symmetry. The check correctly did not fire, and the test failed for
    a reason that had nothing to do with the code.
    """

    def taper(x):
        return 0.1 - 0.02 * x

    grid = Grid.uniform(0.0, 1.0, 99, taper)
    bc = StagnationInletStaticOutlet(p0_in=P01, T0_in=T01, p_back=PB)
    disk = ActuatorDisk(cell=49, compressor_map=ConstantCompressorMap(PR, ETA))
    solver = Solver(grid, GAS, bc, ReferenceState(1.2, 176.0, P01), source=disk)
    solver.set_state(rho=1.2, u=176.0, p=99000.0)
    with pytest.raises(ValueError, match="area varies across the disk"):
        solver.residual()


def test_disk_rejects_bad_configuration():
    with pytest.raises(ValueError, match="n_smear"):
        ActuatorDisk(cell=10, compressor_map=ConstantCompressorMap(PR, ETA), n_smear=0)
    with pytest.raises(ValueError, match="sample_offset"):
        ActuatorDisk(cell=10, compressor_map=ConstantCompressorMap(PR, ETA), sample_offset=0)

    # does not fit in the mesh
    grid = Grid.uniform(0.0, 1.0, 20, AREA)
    bc = StagnationInletStaticOutlet(p0_in=P01, T0_in=T01, p_back=PB)
    disk = ActuatorDisk(cell=1, compressor_map=ConstantCompressorMap(PR, ETA), sample_offset=3)
    solver = Solver(grid, GAS, bc, ReferenceState(1.2, 176.0, P01), source=disk)
    solver.set_state(rho=1.2, u=176.0, p=99000.0)
    with pytest.raises(ValueError, match="does not fit"):
        solver.residual()


def test_source_is_zero_for_a_unity_pressure_ratio():
    """A disk that does nothing must inject nothing, and leave the duct alone."""
    solver, grid, disk, exact = build(pr=1.0, eta=1.0, pb=99000.0)
    q = disk(solver)
    assert disk.last.SWx == pytest.approx(0.0, abs=1e-9)
    assert np.max(np.abs(q[2])) == pytest.approx(0.0, abs=1e-9)
    # Fx is not identically zero -- with PR = 1 the static states still differ
    # only by round-off, so it is tiny but not exactly zero
    assert abs(disk.last.Fx) < 1e-6


# -- the inlet filter (PLAN.md 3.11) ----------------------------------------


class TestInletFilter:
    """The lag that stops the disk responding to its own acoustic echo."""

    def test_zero_tau_is_a_pass_through(self):
        f = InletFilter(0.0)
        for t in (0.0, 1.0, 2.0):
            assert f.update(t, 300.0, 2.0e5, 20.0) == (300.0, 2.0e5, 20.0)

    def test_negative_tau_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            InletFilter(-1.0)

    def test_first_call_seeds_from_the_measurement(self):
        """A converged seed must not be disturbed by switching the filter on."""
        f = InletFilter(1e-3)
        assert f.update(0.0, 300.0, 2.0e5, 20.0) == (300.0, 2.0e5, 20.0)

    def test_unit_dc_gain(self):
        """A held input must be tracked exactly — this is why tau cannot bias
        the converged answer, and it is the property that makes the filter a
        stability device rather than a fudge factor."""
        f = InletFilter(1e-3)
        f.update(0.0, 300.0, 2.0e5, 20.0)
        t = 0.0
        for _ in range(20000):
            t += 1e-6
            T0, p0, W = f.update(t, 310.0, 2.2e5, 21.0)
        assert T0 == pytest.approx(310.0, rel=1e-8)
        assert p0 == pytest.approx(2.2e5, rel=1e-8)
        assert W == pytest.approx(21.0, rel=1e-8)

    def test_advances_once_per_step_not_once_per_stage(self):
        """The solver calls a source at every RK stage; integrating the filter
        at each would run it at five times the physical rate."""
        f = InletFilter(1e-3)
        f.update(0.0, 300.0, 2.0e5, 20.0)
        once = f.update(1e-4, 310.0, 2.2e5, 21.0)
        for _ in range(4):  # four more stages at the same time
            again = f.update(1e-4, 310.0, 2.2e5, 21.0)
        assert again == once

    def test_approaches_a_step_at_the_stated_rate(self):
        """After one tau the response is 1 - 1/e of the step."""
        f = InletFilter(1e-3)
        f.update(0.0, 300.0, 1.0e5, 10.0)
        T0, p0, W = f.update(1e-3, 400.0, 2.0e5, 20.0)
        for got, lo, hi in ((T0, 300.0, 400.0), (p0, 1.0e5, 2.0e5), (W, 10.0, 20.0)):
            assert got == pytest.approx(lo + (hi - lo) * (1 - math.exp(-1.0)), rel=1e-12)

    def test_reset_forgets_the_state(self):
        f = InletFilter(1e-3)
        f.update(0.0, 300.0, 2.0e5, 20.0)
        f.update(1e-3, 400.0, 3.0e5, 25.0)
        f.reset()
        assert f.update(2e-3, 350.0, 2.5e5, 22.0) == (350.0, 2.5e5, 22.0)


class TestRotorPeriod:
    def test_matches_one_revolution(self):
        assert rotor_period(60.0) == pytest.approx(1.0)
        assert rotor_period(10_000.0) == pytest.approx(6e-3)

    def test_rejects_non_positive(self):
        for rpm in (0.0, -100.0):
            with pytest.raises(ValueError, match="rpm must be positive"):
                rotor_period(rpm)


class TestDiskInletLagIsWiredIn:
    def test_disk_defaults_to_no_lag(self):
        d = ActuatorDisk(cell=50, compressor_map=ConstantCompressorMap(1.2, 0.9))
        assert d.inlet_lag == 0.0
        assert d._filter.tau == 0.0

    def test_disk_accepts_a_lag(self):
        d = ActuatorDisk(cell=50, compressor_map=ConstantCompressorMap(1.2, 0.9), inlet_lag=1e-3)
        assert d._filter.tau == 1e-3


class TestHighPressureRatio:
    """The gate that was missing, and that let a regression through four phases.

    Phase 3 checked one pressure ratio, 1.2, which happens to sit just inside
    the stable region. The disk fails from 1.4 (``PLAN.md`` §3.10) because
    runtime evaluation from the locally measured inlet state closes an acoustic
    loop of gain ``Fx/(p01 A)`` — the prototype's frozen source table had been
    suppressing it, and removing that bug exposed it (§3.11).

    Real maps need PR 1.5–2.5 and radial machines reach 14, so a gate at a
    single benign pressure ratio was never a gate.

    These cases are sized at ``M1 = 0.45`` and seeded with the exact discrete
    steady profile. The Phase 3 default sits at ``M1 = 0.665``, 11% from choke,
    and the lag slows the source's response enough that a start from a uniform
    field overshoots into it — a real interaction, and the reason a startup
    transient wants either margin or a steady seed.
    """

    MACH = 0.45

    @classmethod
    def _case(cls, pr, inlet_lag, eta=ETA, n=99, n_smear=1, cfl=2.0):
        from q1d.analytic import (
            compressor_exit_stagnation,
            flow_function,
            state_from_flux,
            static_from_stagnation,
        )

        A = AREA
        W = flow_function(cls.MACH, GAS) * A * P01 / math.sqrt(GAS.R * T01)
        T02, p02 = compressor_exit_stagnation(T01, P01, pr, eta, GAS)
        st1 = static_from_stagnation(T01, P01, W, A, GAS)
        st2 = static_from_stagnation(T02, p02, W, A, GAS)
        exact = zero_d_compressor(P01, T01, st2.p, pr, eta, A, A, GAS)

        grid = Grid.uniform(0.0, 1.0, n, A)
        cell = (n - n_smear) // 2
        bc = StagnationInletStaticOutlet(p0_in=P01, T0_in=T01, p_back=st2.p)
        reference = ReferenceState(rho=st1.rho, u=st1.u, p=P01)
        disk = ActuatorDisk(
            cell=cell,
            compressor_map=ConstantCompressorMap(pr, eta),
            sample_offset=12,
            n_smear=n_smear,
            inlet_lag=inlet_lag,
        )
        solver = Solver(grid, GAS, bc, reference, SolverConfig(cfl=cfl), source=disk)

        Fx = (st2.p - st1.p) * A + W * (st2.u - st1.u)
        SWx = W * GAS.cp * (T02 - T01)
        rho = np.empty(n)
        u = np.empty(n)
        pr_ = np.empty(n)
        rho[:cell], u[:cell], pr_[:cell] = st1.rho, st1.u, st1.p
        rho[cell + n_smear :] = st2.rho
        u[cell + n_smear :] = st2.u
        pr_[cell + n_smear :] = st2.p
        F = np.array(
            [
                st1.rho * st1.u * A,
                (st1.rho * st1.u**2 + st1.p) * A,
                st1.rho * st1.u * (GAS.cp * st1.T + 0.5 * st1.u**2) * A,
            ]
        )
        q = np.array([0.0, Fx, SWx]) / n_smear
        for k in range(n_smear):
            x = state_from_flux(F + 0.5 * q, A, GAS)
            rho[cell + k], u[cell + k], pr_[cell + k] = x.rho, x.u, x.p
            F = F + q
        solver.set_state(rho, u, pr_)
        return solver, disk, exact

    @pytest.mark.parametrize("pr", [1.4, 1.6])
    def test_the_inlet_lag_recovers_the_operating_point(self, pr):
        """With the lag the disk lands on the closed-form answer; without it
        the same case misses by orders of magnitude."""
        solver, disk, exact = self._case(pr, inlet_lag=3e-3)
        solver.run(max_steps=60_000, tol=1e-13)
        assert disk.last.W == pytest.approx(exact.W, rel=1e-6)

        bare, bare_disk, _ = self._case(pr, inlet_lag=0.0)
        try:
            bare.run(max_steps=60_000, tol=1e-13)
            unlagged = abs(bare_disk.last.W / exact.W - 1.0)
        except Exception:  # noqa: BLE001 - divergence is the point
            unlagged = float("inf")
        assert unlagged > 1e-5, (
            "the unlagged disk is expected to miss the point at this pressure "
            "ratio; if it now holds, this regression test has lost its teeth"
        )

    def test_the_lag_does_not_move_the_converged_answer(self):
        """Unit DC gain: the same point, whatever tau. This is what separates a
        stability device from a fudge factor."""
        answers = []
        for tau in (3e-3, 3e-2):
            solver, disk, exact = self._case(1.4, inlet_lag=tau)
            solver.run(max_steps=60_000, tol=1e-13)
            answers.append(disk.last.W / exact.W - 1.0)
        assert answers[0] == pytest.approx(answers[1], abs=1e-8)

    def test_the_phase_three_point_is_unchanged_by_the_lag(self):
        """PR 1.2 already held; switching the filter on must not disturb it."""
        solver, disk, exact = self._case(1.2, inlet_lag=3e-3)
        solver.run(max_steps=60_000, tol=1e-13)
        assert disk.last.W == pytest.approx(exact.W, rel=1e-8)
