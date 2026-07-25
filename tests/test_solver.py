"""Phase 2 gate — the discretisation must be sound before any compressor
physics can be blamed for anything.

No source term appears anywhere in this file. Three independent references:

1. **Well-balancedness** — a stagnant field in a *varying-area* duct must have
   an exactly zero residual, to machine precision. This is the only test that
   exercises the ``p·dA`` geometric source against the pressure flux.
2. **Sod shock tube** — against ``q1d.riemann``, which is itself pinned to
   published values.
3. **Steady nozzle** — against the analytic area–Mach relation via
   ``q1d.analytic.flow_function``.
"""

import numpy as np
import pytest

from q1d.analytic import mach_from_flow_function, static_from_stagnation
from q1d.boundary import StagnationInletStaticOutlet, Transmissive
from q1d.gas import AIR_LEGACY, PerfectGas
from q1d.grid import Grid
from q1d.riemann import SOD_LEFT, SOD_RIGHT, sample_profile
from q1d.solver import (
    CFL_STABILITY_LIMIT,
    NonPhysicalState,
    ReferenceState,
    Solver,
    SolverConfig,
)

GAS = AIR_LEGACY
SOD_GAS = PerfectGas(gamma=1.4, cp=1005.0)


def bump_area(x):
    """Smooth converging–diverging area distribution, 0.1 -> 0.06 -> 0.1."""
    return 0.1 - 0.04 * np.exp(-(((x - 0.5) / 0.15) ** 2))


# ===========================================================================
# 1. Well-balancedness
# ===========================================================================


def _stagnant_solver(order, area, bc):
    grid = Grid.uniform(0.0, 1.0, 80, 0.1 if area == "constant" else bump_area)
    ref = ReferenceState(rho=1.2, u=100.0, p=101325.0)
    solver = Solver(grid, GAS, bc, ref, SolverConfig(order=order))
    solver.set_state(rho=101325.0 / (GAS.R * 288.15), u=0.0, p=101325.0)
    return solver, grid


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("area", ["constant", "varying"])
def test_stagnant_field_has_exactly_zero_residual(order, area):
    """The `p·dA` source must telescope against the pressure flux to zero.

    With ``u = 0`` and uniform ``p``, the momentum flux difference is
    ``p·A_right − p·A_left`` and the geometric source is
    ``p·(A_right − A_left)``. They must cancel *identically*, not
    approximately — otherwise every result on a real flowpath carries a
    spurious body force.

    The ghost cells are set to the interior state exactly here, so this
    isolates the property of the *scheme*. Round-off contributed by the
    boundary conditions is measured separately below.
    """
    solver, _ = _stagnant_solver(order, area, Transmissive())
    solver.cv[:, 0] = solver.cv[:, 1]
    solver.cv[:, -1] = solver.cv[:, -2]
    solver.p[0], solver.p[-1] = solver.p[1], solver.p[-2]

    residual = solver.residual()
    # The momentum equation is the one the geometric source acts on, and it
    # is well-balanced *bitwise*.
    assert np.max(np.abs(residual[1])) == 0.0, (
        f"max |momentum residual| = {np.max(np.abs(residual[1])):.3e}"
    )
    # Mass and energy cannot be bitwise zero when the area varies: recovering
    # rho = (rho*A)/A is not an exact round trip, so the initial field itself
    # is non-uniform at the ulp level. They must sit at round-off.
    scale = 101325.0 * float(np.max(np.abs(solver.grid.a_face)))
    assert np.max(np.abs(residual[0])) < 1e-14 * scale
    assert np.max(np.abs(residual[2])) < 1e-14 * scale


@pytest.mark.parametrize("area", ["constant", "varying"])
def test_characteristic_boundaries_add_only_round_off_to_a_stagnant_field(area):
    """The BC quadratic costs a few ulps, and it is the same in both areas.

    Solving ``c_b`` from the outgoing Riemann invariant and mapping back
    through the isentropic relations does not reproduce the stagnation state
    bit-for-bit. The resulting residual is ~1e-14 relative to ``p·A`` and is
    independent of whether the area varies — confirming it comes from the
    boundary, not from the geometric source term.
    """
    bc = StagnationInletStaticOutlet(p0_in=101325.0, T0_in=288.15, p_back=101325.0)
    solver, grid = _stagnant_solver(2, area, bc)
    scale = 101325.0 * float(np.max(grid.a_face))  # p*A, the momentum flux scale
    relative = np.max(np.abs(solver.residual())) / scale
    assert relative < 1e-13, f"relative residual {relative:.3e}"


def test_stagnant_field_stays_stagnant_under_time_marching():
    grid = Grid.uniform(0.0, 1.0, 80, bump_area)
    ref = ReferenceState(rho=1.2, u=100.0, p=101325.0)
    bc = StagnationInletStaticOutlet(p0_in=101325.0, T0_in=288.15, p_back=101325.0)
    solver = Solver(grid, GAS, bc, ref)

    rho = 101325.0 / (GAS.R * 288.15)
    solver.set_state(rho=rho, u=0.0, p=101325.0)
    solver.run(max_steps=1000, t_end=1e9)  # step count is the real limit here

    r, u, p, _ = solver.primitives()
    # The residual starts at bitwise zero, but recovering p from cv is not an
    # exact round trip when the area varies, so ulp-level noise enters and
    # random-walks. It saturates around 1e-5 m/s against c = 340 (3e-8
    # relative) and does not grow exponentially -- confirmed out to 4000 steps.
    assert np.max(np.abs(u[1:-1])) < 1e-4
    assert np.allclose(p[1:-1], 101325.0, rtol=1e-7)
    assert np.allclose(r[1:-1], rho, rtol=1e-7)


def test_uniform_flow_in_a_constant_area_duct_is_preserved():
    grid = Grid.uniform(0.0, 1.0, 60, 0.1)
    ref = ReferenceState(rho=1.2, u=100.0, p=101325.0)
    solver = Solver(grid, GAS, Transmissive(), ref)
    rho, u, p = 1.2, 120.0, 101325.0
    solver.set_state(rho=rho, u=u, p=p)
    assert np.max(np.abs(solver.residual())) < 1e-9

    solver.run(max_steps=100, t_end=1e9)
    r, uu, pp, _ = solver.primitives()
    assert np.allclose(uu[1:-1], u, rtol=1e-10)
    assert np.allclose(pp[1:-1], p, rtol=1e-10)
    assert np.allclose(r[1:-1], rho, rtol=1e-10)


# ===========================================================================
# 2. Sod shock tube against the exact Riemann solution
# ===========================================================================


def _run_sod(n_cells: int, order: int = 2, t_end: float = 0.2, limiter_factor: float = 1.5):
    grid = Grid.uniform(0.0, 1.0, n_cells, 1.0)
    ref = ReferenceState(rho=1.0, u=1.0, p=1.0)
    solver = Solver(
        grid,
        SOD_GAS,
        Transmissive(),
        ref,
        SolverConfig(cfl=0.8, order=order, entropy_fix=0.05, limiter_factor=limiter_factor),
    )
    x = grid.x_cell
    solver.set_state(
        rho=np.where(x < 0.5, SOD_LEFT.rho, SOD_RIGHT.rho),
        u=np.where(x < 0.5, SOD_LEFT.u, SOD_RIGHT.u),
        p=np.where(x < 0.5, SOD_LEFT.p, SOD_RIGHT.p),
    )
    solver.run(max_steps=20_000, t_end=t_end)
    return solver, grid, x


def test_sod_matches_the_exact_solution():
    solver, grid, x = _run_sod(400)
    rho, u, p, _ = solver.primitives()
    rho_e, u_e, p_e = sample_profile(x, 0.2, SOD_LEFT, SOD_RIGHT, SOD_GAS)

    # L1 error, which is the meaningful norm for a solution with discontinuities
    l1_rho = np.mean(np.abs(rho[1:-1] - rho_e))
    l1_u = np.mean(np.abs(u[1:-1] - u_e))
    l1_p = np.mean(np.abs(p[1:-1] - p_e))
    assert l1_rho < 6e-3, f"L1(rho) = {l1_rho:.4e}"
    assert l1_u < 8e-3, f"L1(u) = {l1_u:.4e}"
    assert l1_p < 5e-3, f"L1(p) = {l1_p:.4e}"


def test_sod_error_reduces_under_refinement():
    errors = []
    for n in (100, 200, 400):
        solver, grid, x = _run_sod(n)
        rho, _, _, _ = solver.primitives()
        rho_e, _, _ = sample_profile(x, 0.2, SOD_LEFT, SOD_RIGHT, SOD_GAS)
        errors.append(np.mean(np.abs(rho[1:-1] - rho_e)))
    assert errors[1] < errors[0] and errors[2] < errors[1], errors
    # first-order-ish convergence is expected for a captured discontinuity
    rate = np.log2(errors[0] / errors[2]) / 2.0
    assert 0.5 < rate < 1.5, f"observed L1 order {rate:.2f}"


def test_sod_second_order_beats_first_order():
    err = {}
    for order in (1, 2):
        solver, grid, x = _run_sod(200, order=order)
        rho, _, _, _ = solver.primitives()
        rho_e, _, _ = sample_profile(x, 0.2, SOD_LEFT, SOD_RIGHT, SOD_GAS)
        err[order] = np.mean(np.abs(rho[1:-1] - rho_e))
    assert err[2] < 0.8 * err[1], err


def test_sod_overshoot_is_bounded_and_controlled_by_the_limiter():
    """This scheme is not strictly monotone, and the overshoot is understood.

    A ~0.5% density overshoot appears at the *head of the rarefaction fan*, not
    at the shock. The van Albada epsilon is a smoothness threshold: near the
    head the per-cell variation is smaller than eps, so the limiter disengages
    and the reconstruction reverts to unlimited central differencing across
    what is a discontinuity in the derivative.

    That makes it a property of `limiter_factor`, not a bug, and the test says
    so by checking the overshoot shrinks monotonically as the threshold drops.
    Undershoot on the right state and negative velocity are genuinely excluded.
    """
    solver, _, x = _run_sod(400)
    rho, u, p, _ = solver.primitives()
    overshoot = rho[1:-1].max() - SOD_LEFT.rho
    assert 0.0 < overshoot < 0.01, f"density overshoot {overshoot:.3e}"
    assert x[int(np.argmax(rho[1:-1]))] < 0.30  # near the fan head at x = 0.263

    # There is a matching undershoot at the foot of the shock, an order of
    # magnitude smaller and from the same cause.
    undershoot = SOD_RIGHT.rho - rho[1:-1].min()
    assert 0.0 < undershoot < 2e-3, f"density undershoot {undershoot:.3e}"
    assert x[int(np.argmin(rho[1:-1]))] > 0.80  # at the shock, x = 0.850

    # Pressure behaves the same way, with both an overshoot above the left
    # state and an undershoot below the right one.
    p_over = p[1:-1].max() - SOD_LEFT.p
    p_under = SOD_RIGHT.p - p[1:-1].min()
    assert 0.0 < p_over < 2e-2, f"pressure overshoot {p_over:.3e}"
    assert 0.0 < p_under < 3e-3, f"pressure undershoot {p_under:.3e}"

    # No reverse flow anywhere -- that would be a genuine failure, not an
    # extremum artefact.
    assert u[1:-1].min() >= -1e-9

    # Tightening the threshold must reduce *all four*, which is what identifies
    # the cause as the smoothness parameter rather than the Riemann solver.
    previous = (overshoot, undershoot, p_over, p_under)
    for limfac in (1.0, 0.6, 0.3):
        s, _, _ = _run_sod(400, limiter_factor=limfac)
        r, _, q, _ = s.primitives()
        current = (
            r[1:-1].max() - SOD_LEFT.rho,
            SOD_RIGHT.rho - r[1:-1].min(),
            q[1:-1].max() - SOD_LEFT.p,
            SOD_RIGHT.p - q[1:-1].min(),
        )
        for name, now, before in zip(
            ("rho over", "rho under", "p over", "p under"), current, previous, strict=True
        ):
            assert now < before, f"limfac={limfac}: {name} {now:.3e} !< {before:.3e}"
        previous = current


def test_sod_conserves_mass_and_energy():
    solver, grid, _ = _run_sod(200)
    # transmissive boundaries let waves out, but at t = 0.2 none has reached
    # them, so the interior totals must still equal the initial ones
    total_mass = np.sum(solver.cv[0, 1:-1] * grid.dx)
    total_energy = np.sum(solver.cv[2, 1:-1] * grid.dx)
    x = grid.x_cell
    rho0 = np.where(x < 0.5, SOD_LEFT.rho, SOD_RIGHT.rho)
    p0 = np.where(x < 0.5, SOD_LEFT.p, SOD_RIGHT.p)
    assert total_mass == pytest.approx(np.sum(rho0 * grid.a_cell[1:-1] * grid.dx), rel=1e-10)
    assert total_energy == pytest.approx(
        np.sum(p0 / SOD_GAS.gm1 * grid.a_cell[1:-1] * grid.dx), rel=1e-10
    )


# ===========================================================================
# 3. Steady nozzle against the area–Mach relation
# ===========================================================================


def test_subsonic_nozzle_matches_the_area_mach_relation():
    """Fully subsonic venturi: isentropic, so Phi(x) = W*sqrt(R*T0)/(A(x)*p0).

    Every station must sit on the analytic area–Mach curve for the mass flow
    the solver settles at. This checks the area terms and the characteristic
    boundary conditions together, with no source term involved.
    """
    p0_in, T0_in, p_back = 101325.0, 288.15, 95000.0
    grid = Grid.uniform(0.0, 1.0, 200, bump_area)
    ref = ReferenceState(rho=1.2, u=150.0, p=101325.0)
    bc = StagnationInletStaticOutlet(p0_in=p0_in, T0_in=T0_in, p_back=p_back)
    solver = Solver(grid, GAS, bc, ref, SolverConfig(cfl=1.5, local_time_stepping=True))

    st = static_from_stagnation(T0_in, p0_in, 15.0, float(np.max(grid.a_face)), GAS)
    solver.set_state(rho=st.rho, u=st.u, p=st.p)
    result = solver.run(max_steps=40_000, tol=1e-11)
    assert result.converged, f"residual {result.final_residual}"

    rho, u, p, c = solver.primitives()

    # The conserved quantity is the *face* mass flux. It must be uniform to
    # round-off. Cell-centred rho*u*A(x_centre) is not conserved and differs by
    # O(dx^2) wherever the area has curvature -- see the refinement test below.
    mass_flux = solver.face_fluxes()[0]
    spread = float(np.std(mass_flux) / np.mean(mass_flux))
    assert spread < 1e-11, f"face mass flux spread {spread:.3e}"

    W_mean = float(np.mean(mass_flux))
    mach = u[1:-1] / c[1:-1]
    phi_expected = W_mean * np.sqrt(GAS.R * T0_in) / (grid.a_cell[1:-1] * p0_in)
    mach_expected = np.array([mach_from_flow_function(float(f), GAS) for f in phi_expected])
    assert np.max(np.abs(mach - mach_expected)) < 2e-4

    # Stagnation pressure must be preserved: the flow is isentropic. The
    # residual loss is numerical entropy generation through the throat, and it
    # converges at ~3rd order -- see the refinement test below. At n=200 it is
    # 1.16e-4, so this bound is a measured value, not an aspiration.
    pt = p[1:-1] * (1.0 + 0.5 * GAS.gm1 * mach**2) ** GAS.g_over_gm1
    assert np.max(np.abs(pt / p0_in - 1.0)) < 2e-4


def test_nozzle_discretisation_errors_converge_under_refinement():
    """The two residual errors in the nozzle are discretisation, not defects.

    * Cell-centred ``rho*u*A(x_centre)`` is not the conserved quantity — it is
      the face flux that telescopes — so its spread is an ``O(dx^2)`` artefact.
    * Stagnation pressure loss is numerical entropy generation through the
      throat, and converges at ~3rd order (1.10e-3, 1.16e-4, 1.53e-5 at
      n = 100, 200, 400).

    A *plateau* in either would mean a genuine spurious source, which matters
    directly for Phase 3: the compressor model reads p0 upstream of the disk,
    so anything that erodes stagnation pressure corrupts the map lookup.
    """
    spreads = []
    pt_errors = []
    for n in (100, 200, 400):
        grid = Grid.uniform(0.0, 1.0, n, bump_area)
        bc = StagnationInletStaticOutlet(p0_in=101325.0, T0_in=288.15, p_back=95000.0)
        solver = Solver(
            grid,
            GAS,
            bc,
            ReferenceState(rho=1.2, u=150.0, p=101325.0),
            SolverConfig(cfl=1.5, local_time_stepping=True),
        )
        st = static_from_stagnation(288.15, 101325.0, 15.0, float(np.max(grid.a_face)), GAS)
        solver.set_state(rho=st.rho, u=st.u, p=st.p)
        assert solver.run(max_steps=40_000, tol=1e-11).converged
        rho, u, p, c = solver.primitives()
        W = rho[1:-1] * u[1:-1] * grid.a_cell[1:-1]
        spreads.append(float(np.std(W) / np.mean(W)))
        mach = u[1:-1] / c[1:-1]
        pt = p[1:-1] * (1.0 + 0.5 * GAS.gm1 * mach**2) ** GAS.g_over_gm1
        pt_errors.append(float(np.max(np.abs(pt / 101325.0 - 1.0))))

    order = np.log2(spreads[0] / spreads[2]) / 2.0
    assert 1.7 < order < 2.3, f"cell-W order {order:.2f} from spreads {spreads}"

    pt_order = np.log2(pt_errors[0] / pt_errors[2]) / 2.0
    assert pt_order > 2.5, f"stagnation-pressure loss order {pt_order:.2f} from {pt_errors}"
    assert pt_errors[2] < 5e-5


def test_nozzle_throat_is_the_fastest_point():
    p0_in, T0_in, p_back = 101325.0, 288.15, 95000.0
    grid = Grid.uniform(0.0, 1.0, 120, bump_area)
    ref = ReferenceState(rho=1.2, u=150.0, p=101325.0)
    bc = StagnationInletStaticOutlet(p0_in=p0_in, T0_in=T0_in, p_back=p_back)
    solver = Solver(grid, GAS, bc, ref, SolverConfig(local_time_stepping=True))
    solver.set_state(rho=1.2, u=100.0, p=99000.0)
    solver.run(max_steps=40_000, tol=1e-10)

    _, u, p, c = solver.primitives()
    mach = (u / c)[1:-1]
    throat = int(np.argmin(grid.a_cell[1:-1]))
    assert int(np.argmax(mach)) == pytest.approx(throat, abs=1)
    assert int(np.argmin(p[1:-1])) == pytest.approx(throat, abs=1)


# ===========================================================================
# Convergence, robustness and configuration
# ===========================================================================


def test_residual_falls_by_many_orders_of_magnitude():
    grid = Grid.uniform(0.0, 1.0, 100, bump_area)
    ref = ReferenceState(rho=1.2, u=150.0, p=101325.0)
    bc = StagnationInletStaticOutlet(p0_in=101325.0, T0_in=288.15, p_back=95000.0)
    solver = Solver(grid, GAS, bc, ref, SolverConfig(local_time_stepping=True))
    solver.set_state(rho=1.2, u=100.0, p=99000.0)
    result = solver.run(max_steps=40_000, tol=1e-12)

    assert result.converged
    first, last = result.residual_history[0], result.residual_history[-1]
    assert np.all(last < first * 1e-10)


def test_converged_solution_holds(monkeypatch):
    """The D10 hold test: reaching the point is not the same as staying there.

    Run 10x longer than convergence took and require no secular drift. This is
    what catches slow instabilities that a converged-and-stopped run hides.
    """
    grid = Grid.uniform(0.0, 1.0, 100, bump_area)
    ref = ReferenceState(rho=1.2, u=150.0, p=101325.0)
    bc = StagnationInletStaticOutlet(p0_in=101325.0, T0_in=288.15, p_back=95000.0)
    solver = Solver(grid, GAS, bc, ref, SolverConfig(local_time_stepping=True))
    solver.set_state(rho=1.2, u=100.0, p=99000.0)

    converge = solver.run(max_steps=40_000, tol=1e-11)
    assert converge.converged
    rho, u, _, _ = solver.primitives()
    W_converged = float(np.mean(rho[1:-1] * u[1:-1] * grid.a_cell[1:-1]))

    hold = solver.run(max_steps=10 * converge.steps, t_end=1e9)
    rho, u, _, _ = solver.primitives()
    W_held = float(np.mean(rho[1:-1] * u[1:-1] * grid.a_cell[1:-1]))

    assert abs(W_held / W_converged - 1.0) < 1e-9, "mass flow drifted during the hold"
    assert np.all(hold.residual_history[-1] <= hold.residual_history[0] * 10.0), (
        "residual grew during the hold — slow instability"
    )


def test_reference_state_rejects_nonpositive_values():
    for kwargs in [
        {"rho": 0.0, "u": 1.0, "p": 1.0, "vol": 1.0},
        {"rho": 1.0, "u": -1.0, "p": 1.0, "vol": 1.0},
        {"rho": 1.0, "u": 1.0, "p": 0.0, "vol": 1.0},
        {"rho": 1.0, "u": 1.0, "p": 1.0, "vol": 0.0},
    ]:
        with pytest.raises(ValueError):
            ReferenceState(**kwargs)


def test_non_physical_state_raises_rather_than_propagating_nan():
    grid = Grid.uniform(0.0, 1.0, 40, 0.1)
    ref = ReferenceState(rho=1.2, u=100.0, p=101325.0)
    solver = Solver(grid, GAS, Transmissive(), ref, SolverConfig(cfl=50.0))
    x = grid.x_cell
    solver.set_state(rho=np.where(x < 0.5, 10.0, 0.01), u=0.0, p=np.where(x < 0.5, 1e7, 1e3))
    with pytest.raises(NonPhysicalState):
        solver.run(max_steps=500, t_end=1e9)


def test_local_time_stepping_reaches_the_same_steady_state():
    def solve(local):
        grid = Grid.uniform(0.0, 1.0, 100, bump_area)
        ref = ReferenceState(rho=1.2, u=150.0, p=101325.0)
        bc = StagnationInletStaticOutlet(p0_in=101325.0, T0_in=288.15, p_back=95000.0)
        s = Solver(grid, GAS, bc, ref, SolverConfig(local_time_stepping=local))
        s.set_state(rho=1.2, u=100.0, p=99000.0)
        r = s.run(max_steps=60_000, tol=1e-11)
        rho, u, _, _ = s.primitives()
        return r, float(np.mean(rho[1:-1] * u[1:-1] * grid.a_cell[1:-1]))

    r_global, W_global = solve(False)
    r_local, W_local = solve(True)
    assert r_global.converged and r_local.converged
    assert W_local == pytest.approx(W_global, rel=1e-8)
    assert r_local.steps < r_global.steps  # that is the point of it


def test_grid_validates_its_shapes():
    with pytest.raises(ValueError, match="at least 3"):
        Grid.uniform(0.0, 1.0, 2, 0.1)
    grid = Grid.uniform(0.0, 1.0, 10, bump_area)
    assert grid.n_interior == 10
    assert not grid.is_constant_area
    assert Grid.uniform(0.0, 1.0, 10, 0.1).is_constant_area
    # da must telescope to the total area change
    assert np.sum(grid.da) == pytest.approx(grid.a_face[-1] - grid.a_face[0], rel=1e-14)


def test_measured_cfl_stability_limit():
    """Pin the true stability limit, which is not the one the coefficients imply.

    ``RK5`` is Blazek's *hybrid* 5-stage set, tuned for dissipation evaluated at
    stages 1/3/5 only. This solver evaluates it at every stage, so that tuning
    does not apply. Measured on a stagnant varying-area duct whose residual is
    bitwise zero at t=0, so any growth is pure amplification rather than
    physics: CFL 2.44 holds at the round-off floor, CFL 2.46 does not.

    The legacy default was 2.5 — past the limit.
    """

    def noise_after(cfl, steps=400):
        grid = Grid.uniform(0.0, 1.0, 80, bump_area)
        bc = StagnationInletStaticOutlet(p0_in=101325.0, T0_in=288.15, p_back=101325.0)
        solver = Solver(
            grid, GAS, bc, ReferenceState(rho=1.2, u=100.0, p=101325.0), SolverConfig(cfl=cfl)
        )
        solver.set_state(rho=101325.0 / (GAS.R * 288.15), u=0.0, p=101325.0)
        try:
            solver.run(max_steps=steps, t_end=1e9)
        except NonPhysicalState:
            return float("inf")
        _, u, _, _ = solver.primitives()
        return float(np.max(np.abs(u[1:-1])))

    assert noise_after(2.44) < 1e-4, "CFL 2.44 should be stable"
    assert noise_after(2.46) > 1e-2, "CFL 2.46 should be unstable"
    assert 2.44 < CFL_STABILITY_LIMIT < 2.46
    assert SolverConfig().cfl < CFL_STABILITY_LIMIT  # default carries margin

    # and it is a genuine floor, not slow growth: 10x the steps stays bounded
    assert noise_after(2.0, steps=4000) < 1e-4
