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

import numpy as np
import pytest

from q1d.analytic import zero_d_compressor
from q1d.boundary import StagnationInletStaticOutlet
from q1d.compressor import ActuatorDisk, ConstantCompressorMap
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


def test_sampling_distance_sensitivity_decays_and_sets_the_default():
    """PLAN.md §7 Q1 said measurement would decide this, and it did.

    The disk perturbs the field *upstream* as well as downstream — a numerical
    boundary layer from the reconstruction stencil, not a physical one. Its
    influence decays geometrically with standoff, and the default is chosen
    from where that decay clears the gate rather than assumed.

    This test would have caught the original default of 3, which sat inside the
    layer and cost 1.8e-5 in mass flow.
    """
    errors = {}
    for offset in (1, 3, 5, 8, 12):
        solver, grid, disk, exact = build(sample_offset=offset)
        result, W = converged_mass_flow(solver, grid)
        assert result.converged, f"offset {offset} did not converge"
        errors[offset] = abs(W / exact.W - 1.0)

    # monotone decay, by roughly an order of magnitude every 2-3 cells
    for near, far in ((1, 3), (3, 5), (5, 8), (8, 12)):
        assert errors[far] < errors[near], f"error grew from offset {near} to {far}: {errors}"

    # sampling inside the layer misses the gate; the default clears it
    assert errors[3] > 1e-8, "offset 3 should be inside the boundary layer"
    assert errors[12] < 1e-9, f"default offset must clear the gate, got {errors[12]:.3e}"


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
    """A zero-thickness disk has one area; see PLAN.md §4.5."""

    def bump(x):
        return 0.1 - 0.02 * np.exp(-(((x - 0.5) / 0.2) ** 2))

    grid = Grid.uniform(0.0, 1.0, 99, bump)
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
