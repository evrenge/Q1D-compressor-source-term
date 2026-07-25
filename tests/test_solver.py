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

import math

import numpy as np
import pytest

from q1d.analytic import mach_from_flow_function, static_from_stagnation
from q1d.boundary import StagnationInletStaticOutlet, Transmissive
from q1d.gas import AIR_LEGACY, PerfectGas
from q1d.grid import Grid
from q1d.riemann import SOD_LEFT, SOD_RIGHT, RiemannState, sample_profile
from q1d.solver import (
    CFL_STABILITY_LIMIT_ORDER2,
    NonPhysicalState,
    ReferenceState,
    Solver,
    SolverConfig,
)

GAS = AIR_LEGACY
SOD_GAS = PerfectGas(gamma=1.4, cp=1005.0)


def bump_area(x):
    """Smooth converging–diverging area distribution, 0.1 -> 0.06 -> 0.1.

    NOTE: symmetric about x = 0.5. Mutation testing showed that symmetry makes
    several geometry bugs invisible (reversing ``a_cell`` is a no-op on it), so
    ``skew_area`` below exists to break it and is used wherever geometry
    orientation matters.
    """
    return 0.1 - 0.04 * np.exp(-(((x - 0.5) / 0.15) ** 2))


def skew_area(x):
    """Deliberately asymmetric area: no mirror symmetry, monotone-ish ramp."""
    return 0.12 - 0.05 * np.exp(-(((x - 0.32) / 0.11) ** 2)) - 0.02 * x


def stretched_grid(n=80, ratio=1.03, area=skew_area):
    """Geometrically stretched mesh — ``Grid.uniform`` cannot produce one.

    Every grid elsewhere in the suite has uniform ``dx``, which hides bugs in
    any expression that mixes cell and face indices.
    """
    w = ratio ** np.arange(n)
    x_face = np.concatenate(([0.0], np.cumsum(w / w.sum())))
    x_cell = 0.5 * (x_face[1:] + x_face[:-1])
    a_face = np.asarray(area(x_face), dtype=float)
    a_in = np.asarray(area(x_cell), dtype=float)
    dx = x_face[1:] - x_face[:-1]
    vol_in = a_in * dx
    return Grid(
        x_face=x_face,
        a_face=a_face,
        x_cell=x_cell,
        dx=dx,
        da=a_face[1:] - a_face[:-1],
        a_cell=np.concatenate(([a_in[0]], a_in, [a_in[-1]])),
        vol=np.concatenate(([vol_in[0]], vol_in, [vol_in[-1]])),
    )


# ===========================================================================
# 1. Well-balancedness
# ===========================================================================


def _stagnant_flux_scales(grid):
    """Natural flux scale of each equation for a stagnant duct at 1 atm.

    Normalising all three by ``p*A`` (as an earlier version did) is wrong: the
    energy flux scale is ~1e6 W while the momentum one is ~1e4 N, so a common
    denominator makes a one-ulp energy residual look 100x worse than a one-ulp
    momentum residual.
    """
    a = float(np.max(grid.a_face))
    rho = 101325.0 / (GAS.R * 288.15)
    c = GAS.speed_of_sound(288.15)
    return np.array([rho * c * a, 101325.0 * a, 101325.0 / GAS.gm1 * c * a])


def _stagnant_solver(order, area, bc):
    areas = {"constant": 0.1, "varying": bump_area, "skew": skew_area}
    grid = Grid.uniform(0.0, 1.0, 80, areas[area])
    ref = ReferenceState(rho=1.2, u=100.0, p=101325.0)
    solver = Solver(grid, GAS, bc, ref, SolverConfig(order=order))
    solver.set_state(rho=101325.0 / (GAS.R * 288.15), u=0.0, p=101325.0)
    return solver, grid


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("area", ["constant", "varying", "skew"])
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

    The pressure field is forced bitwise uniform here, because that is the
    hypothesis of the property. See the companion test below for what the
    time-stepper actually realises, which is *not* bitwise zero when the area
    varies — recovering ``p`` from ``cv`` is not an exact round trip, so
    neighbouring cells differ by an ulp and the cancellation is only to
    round-off. Conflating the two is how an earlier version of this test
    claimed bitwise zero for a state the solver never visits.
    """
    solver, _ = _stagnant_solver(order, area, Transmissive())
    solver.cv[:, 0] = solver.cv[:, 1]
    solver.cv[:, -1] = solver.cv[:, -2]
    solver.p[:] = solver.p[1]  # bitwise-uniform pressure: the hypothesis

    residual = solver.residual()
    assert np.max(np.abs(residual[1])) == 0.0, (
        f"max |momentum residual| = {np.max(np.abs(residual[1])):.3e}"
    )
    # Mass and energy are a *different* hypothesis: they involve the
    # reconstructed density, and rho = (rho0*A)/A is not bitwise uniform for an
    # arbitrary area law (2.1e-16 on the skew grid). Well-balancedness proper
    # is the momentum statement above; these are held to round-off.
    # (An earlier version asserted "< 1e-14 * scale" = 1e-10 absolute and
    # attributed the slack to a round trip it never measured -- six orders
    # loose, and unable to catch a real defect.)
    scales = _stagnant_flux_scales(solver.grid)
    assert np.max(np.abs(residual[0])) / scales[0] < 1e-14
    assert np.max(np.abs(residual[2])) / scales[2] < 1e-14


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("area", ["constant", "varying", "skew"])
def test_realised_stagnant_residual_is_round_off(order, area):
    """What the time-stepper actually produces, as opposed to the ideal above.

    ``set_state`` derives ``p`` from ``cv``, so with a varying area the
    recovered pressure is not bitwise uniform and the geometric source cancels
    the pressure flux only to round-off: 1.8e-12 N at order 1, 3.6e-12 at
    order 2, against a ``p·A`` scale of ~1e4 N — about one ulp. Constant area
    stays exactly zero because the round trip is exact there.
    """
    solver, grid = _stagnant_solver(order, area, Transmissive())
    solver.cv[:, 0] = solver.cv[:, 1]
    solver.cv[:, -1] = solver.cv[:, -2]
    solver.p[0], solver.p[-1] = solver.p[1], solver.p[-2]

    relative = np.max(np.abs(solver.residual()).max(axis=1) / _stagnant_flux_scales(grid))
    assert relative < 1e-14, f"relative residual {relative:.3e}"
    if area == "constant":
        assert np.max(np.abs(solver.residual())) == 0.0


@pytest.mark.parametrize("area", ["constant", "varying", "skew"])
def test_characteristic_boundaries_add_only_round_off_to_a_stagnant_field(area):
    """The BC quadratic costs a few ulps, and it is the same in both areas.

    Solving ``c_b`` from the outgoing Riemann invariant and mapping back
    through the isentropic relations does not reproduce the stagnation state
    bit-for-bit. The resulting residual is ~1e-14 relative to ``p·A`` and is
    independent of whether the area varies — confirming it comes from the
    boundary, not from the geometric source term.
    """
    bc = StagnationInletStaticOutlet(p0_in=101325.0, T0_in=288.15, p_back=101325.0)
    solver, _ = _stagnant_solver(2, area, bc)
    # Measured: the characteristic quadratic reproduces the stagnation state
    # bit-for-bit at u = 0, so the boundaries cost nothing beyond the ulp-level
    # density non-uniformity already present. An earlier version asserted
    # "< 1e-13" while claiming "a few ulps" -- the number asserted was never
    # the number measured.
    scales = _stagnant_flux_scales(solver.grid)
    relative = np.max(np.abs(solver.residual()).max(axis=1) / scales)
    assert relative < 1e-14, f"relative residual {relative:.3e}"


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
    so by checking every extremum shrinks monotonically as the threshold drops.
    Density, pressure and velocity all show it; the scheme is not strictly
    monotone for systems, and pretending otherwise would mean tuning tolerances
    until they happened to pass.
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

    # Velocity too. The exact solution has u >= 0 everywhere, so any negative
    # value is reverse flow -- but it belongs to the same artefact family, at
    # ~0.8% of the star-region velocity of 0.927. Its location migrates between
    # the shock foot and the fan head as the threshold changes, so unlike the
    # others it gets no position assertion.
    u_under = -u[1:-1].min()
    assert 0.0 < u_under < 2e-2, f"reverse flow {u_under:.3e}"

    # Tightening the threshold must reduce *all five*, which is what identifies
    # the cause as the smoothness parameter rather than the Riemann solver.
    names = ("rho over", "rho under", "p over", "p under", "u under")
    previous = (overshoot, undershoot, p_over, p_under, u_under)
    for limfac in (1.0, 0.6, 0.3):
        s, _, _ = _run_sod(400, limiter_factor=limfac)
        r, w, q, _ = s.primitives()
        current = (
            r[1:-1].max() - SOD_LEFT.rho,
            SOD_RIGHT.rho - r[1:-1].min(),
            q[1:-1].max() - SOD_LEFT.p,
            SOD_RIGHT.p - q[1:-1].min(),
            -w[1:-1].min(),
        )
        for name, now, before in zip(names, current, previous, strict=True):
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

    # No stopping criterion: run exactly 10x the steps convergence took.
    hold = solver.run(max_steps=10 * converge.steps)
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
    """The stability limit is mesh-dependent, and the shipped default is safe.

    ``RK5`` is Blazek's *hybrid* 5-stage set, tuned for dissipation evaluated at
    stages 1/3/5 only. This solver evaluates it at every stage, so that tuning
    does not apply. Measured on a stagnant varying-area duct whose residual is
    bitwise zero at t=0, so any growth is pure amplification rather than
    physics: CFL 2.44 holds at the round-off floor, CFL 2.46 does not.

    Measured limits (order 2, stagnant varying-area duct, residual bitwise zero
    at t=0 so growth is pure amplification): 2.475 at n=40, 2.452 at n=80,
    2.440 at n=160, 2.435 at n=320 — falling with refinement. Order 1 is far
    more forgiving at ~3.13. The legacy default of 2.5 is past the limit on
    every mesh tested.
    """

    def noise_after(cfl, steps=400, n=80):
        grid = Grid.uniform(0.0, 1.0, n, bump_area)
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

    # The shipped default must be stable across meshes and over a long horizon.
    # This is the assertion that actually protects a user.
    for n in (80, 160, 320):
        assert noise_after(SolverConfig().cfl, steps=5000, n=n) < 1e-4, f"default unstable at n={n}"

    # The limit is NOT a single number: it falls with mesh refinement, and 2.44
    # -- previously pinned as "stable" -- is unstable by n=160 once the horizon
    # is long enough to reveal it. The 400-step probe that produced that claim
    # was too short.
    assert noise_after(2.44, steps=400, n=80) < 1e-4  # what the old test saw
    assert noise_after(2.44, steps=5000, n=160) > 1e-2  # what it missed
    assert noise_after(2.6, steps=400, n=80) > 1e-2  # comfortably past it either way

    assert SolverConfig().cfl < CFL_STABILITY_LIMIT_ORDER2  # default carries margin


# ===========================================================================
# The source hook — the one line Phase 3 is built on
# ===========================================================================


def test_source_hook_sign_and_shape_contract():
    """`rhs -= source(solver)`, so a positive source *adds* to the conserved state.

    Until now no test executed this line at all. Its sign convention and shape
    contract are the entire interface between the solver and the actuator disk,
    so they are pinned directly against the residual rather than inferred from
    a converged answer.
    """
    grid = Grid.uniform(0.0, 1.0, 40, 0.1)
    solver = Solver(grid, GAS, Transmissive(), ReferenceState(rho=1.2, u=120.0, p=101325.0))
    solver.set_state(rho=1.2, u=120.0, p=101325.0)

    q = np.zeros((3, grid.n_interior))
    q[1, 17] = 1731.24  # a blade force, in newtons
    q[2, 17] = 3.7e5  # shaft power, in watts

    without = solver.residual().copy()
    solver.source = lambda s: q
    with_source = solver.residual()

    # rhs -= q, exactly, with no other change
    assert np.allclose(without - with_source, q, rtol=0.0, atol=1e-9)
    assert with_source.shape == (3, grid.n_interior)


def test_positive_momentum_source_accelerates_the_flow():
    """Sign check in physical terms, not just algebraic."""
    grid = Grid.uniform(0.0, 1.0, 60, 0.1)
    bc = StagnationInletStaticOutlet(p0_in=101325.0, T0_in=288.15, p_back=95000.0)
    solver = Solver(grid, GAS, bc, ReferenceState(rho=1.2, u=150.0, p=101325.0))
    solver.set_state(rho=1.19, u=150.0, p=99000.0)
    assert solver.run(max_steps=4000, tol=1e-9).converged
    base = solver.cv.copy()
    _, u0, _, _ = solver.primitives()
    reference = float(np.mean(u0[1:-1]))

    # A bare constant point force is not a self-regulating actuator disk -- it
    # keeps pushing regardless of state -- so this is deliberately a
    # fixed-step comparison rather than a convergence test. The sign is what is
    # under test, not the steady point.
    for force, expected_faster in ((+500.0, True), (-500.0, False)):
        solver.cv[:] = base
        solver._update_pressure()
        solver._sync_boundaries()
        q = np.zeros((3, grid.n_interior))
        q[1, 30] = force
        solver.source = lambda s, q=q: q
        solver.run(max_steps=600)
        _, u1, _, _ = solver.primitives()
        faster = float(np.mean(u1[1:-1])) > reference
        assert faster is expected_faster, f"force {force:+.0f} N gave the wrong direction"


def test_source_hook_integral_balance_is_exact():
    """Total conserved quantity changes by exactly the injected amount.

    With `Transmissive` boundaries and a uniform initial state the net flux
    through the domain is zero, so one RK step must move the volume integral of
    `cv` by exactly `dt * sum(q)`. This is the property that makes the Phase 3
    tolerance reachable, tested here without any compressor physics involved.
    """
    grid = Grid.uniform(0.0, 1.0, 50, 0.1)
    solver = Solver(grid, GAS, Transmissive(), ReferenceState(rho=1.2, u=100.0, p=101325.0))
    solver.set_state(rho=1.2, u=100.0, p=101325.0)

    q = np.zeros((3, grid.n_interior))
    q[1, 25] = 2000.0
    q[2, 25] = 5.0e5
    solver.source = lambda s: q

    before = (solver.cv[:, 1:-1] * grid.dx).sum(axis=1)
    dt = solver.timestep()
    solver.advance(dt)
    after = (solver.cv[:, 1:-1] * grid.dx).sum(axis=1)

    expected = dt * q.sum(axis=1)
    assert np.allclose(after - before, expected, rtol=1e-9, atol=1e-6)


# ===========================================================================
# Supersonic boundary branches — advertised in PLAN.md, never executed
# ===========================================================================


def _supersonic_bc(mach=2.0, p0=400000.0, T0=500.0):
    p_static = p0 * (1.0 + 0.5 * GAS.gm1 * mach**2) ** (-GAS.g_over_gm1)
    return StagnationInletStaticOutlet(
        p0_in=p0, T0_in=T0, p_back=0.5 * p_static, p_static_in=p_static
    ), p_static


def test_supersonic_inflow_requires_and_validates_a_third_condition():
    bc_ok, p_static = _supersonic_bc()
    T = 500.0 * (p_static / 400000.0) ** GAS.gm1_over_g
    rho, u, p = p_static / (GAS.R * T), 2.0 * GAS.speed_of_sound(T), p_static
    c = GAS.speed_of_sound(T)

    ghost = bc_ok.left(rho, u, p, c, GAS)
    assert ghost.p == pytest.approx(p_static, rel=1e-12)
    assert ghost.u / GAS.speed_of_sound(ghost.p / (ghost.rho * GAS.R)) == pytest.approx(
        2.0, rel=1e-9
    )

    # all three characteristics enter, so p0/T0 alone under-specify the state
    bc_missing = StagnationInletStaticOutlet(p0_in=400000.0, T0_in=500.0, p_back=50000.0)
    with pytest.raises(ValueError, match="third condition"):
        bc_missing.left(rho, u, p, c, GAS)

    # and an imposed triple that is not actually supersonic is rejected
    bc_bad = StagnationInletStaticOutlet(
        p0_in=400000.0, T0_in=500.0, p_back=50000.0, p_static_in=399000.0
    )
    with pytest.raises(ValueError, match="classified as supersonic"):
        bc_bad.left(rho, u, p, c, GAS)


def test_supersonic_outflow_extrapolates_at_both_ends():
    bc, _ = _supersonic_bc()
    rho, u, p = 0.5, 900.0, 50000.0
    c = math.sqrt(GAS.gamma * p / rho)
    assert u > c

    right = bc.right(rho, u, p, c, GAS)
    assert (right.rho, right.u, right.p) == pytest.approx((rho, u, p), rel=1e-14)
    # reverse supersonic flow through the left boundary is also extrapolation
    left = bc.left(rho, -u, p, c, GAS)
    assert (left.rho, left.u, left.p) == pytest.approx((rho, -u, p), rel=1e-14)


def test_uniform_supersonic_flow_is_preserved_through_both_boundaries():
    """End-to-end exercise of the supersonic inlet and outlet together."""
    mach, p0, T0 = 2.0, 400000.0, 500.0
    bc, p_static = _supersonic_bc(mach, p0, T0)
    T = T0 * (p_static / p0) ** GAS.gm1_over_g
    rho = p_static / (GAS.R * T)
    u = mach * GAS.speed_of_sound(T)

    grid = Grid.uniform(0.0, 1.0, 60, 0.1)
    solver = Solver(grid, GAS, bc, ReferenceState(rho=rho, u=u, p=p_static))
    solver.set_state(rho=rho, u=u, p=p_static)
    solver.run(max_steps=300)

    r, uu, pp, cc = solver.primitives()
    assert np.allclose(uu[1:-1], u, rtol=1e-9)
    assert np.allclose(pp[1:-1], p_static, rtol=1e-9)
    assert np.allclose(r[1:-1], rho, rtol=1e-9)
    assert np.all(uu[1:-1] > cc[1:-1])  # still supersonic everywhere


@pytest.mark.parametrize("order", [1, 2])
def test_well_balanced_on_a_stretched_mesh(order):
    """Non-uniform `dx`, which `Grid.uniform` cannot produce.

    Every other grid in the suite has uniform spacing, which hides bugs in any
    expression mixing cell and face indices — mutation testing showed several
    geometry errors surviving for exactly that reason.
    """
    grid = stretched_grid(n=80, ratio=1.03, area=skew_area)
    assert grid.dx.max() / grid.dx.min() > 5.0, "mesh is not actually stretched"

    solver = Solver(
        grid,
        GAS,
        Transmissive(),
        ReferenceState(rho=1.2, u=100.0, p=101325.0),
        SolverConfig(order=order),
    )
    solver.set_state(rho=101325.0 / (GAS.R * 288.15), u=0.0, p=101325.0)
    solver.cv[:, 0] = solver.cv[:, 1]
    solver.cv[:, -1] = solver.cv[:, -2]
    solver.p[:] = solver.p[1]

    assert np.max(np.abs(solver.residual()[1])) == 0.0

    # and it stays put under time marching
    solver.set_state(rho=101325.0 / (GAS.R * 288.15), u=0.0, p=101325.0)
    solver.run(max_steps=400)
    _, u, p, _ = solver.primitives()
    assert np.max(np.abs(u[1:-1])) < 1e-4
    assert np.allclose(p[1:-1], 101325.0, rtol=1e-7)


# ===========================================================================
# The Harten entropy fix
# ===========================================================================

#: Transonic rarefaction: `u - c` runs from -0.433 to +0.300 through the fan,
#: so the sonic point lies inside it. Sod does NOT have this property -- its
#: `u - c` goes from -1.183 to -0.070 and never crosses zero -- which is why
#: the entire suite could pass with the entropy fix deleted.
TRANSONIC_LEFT = RiemannState(rho=1.0, u=0.75, p=1.0)
TRANSONIC_RIGHT = RiemannState(rho=0.125, u=0.0, p=0.1)
TRANSONIC_X0 = 0.3


def _sonic_point_error(n, entropy_fix, order=1, t_end=0.2):
    """Max density error in the cells straddling the sonic point."""
    grid = Grid.uniform(0.0, 1.0, n, 1.0)
    x = grid.x_cell
    solver = Solver(
        grid,
        SOD_GAS,
        Transmissive(),
        ReferenceState(rho=1.0, u=1.0, p=1.0),
        SolverConfig(cfl=0.8, order=order, entropy_fix=entropy_fix),
    )
    solver.set_state(
        rho=np.where(x < TRANSONIC_X0, TRANSONIC_LEFT.rho, TRANSONIC_RIGHT.rho),
        u=np.where(x < TRANSONIC_X0, TRANSONIC_LEFT.u, TRANSONIC_RIGHT.u),
        p=np.where(x < TRANSONIC_X0, TRANSONIC_LEFT.p, TRANSONIC_RIGHT.p),
    )
    solver.run(max_steps=20_000, t_end=t_end)
    rho, _, _, _ = solver.primitives()
    exact, _, _ = sample_profile(
        x, t_end, TRANSONIC_LEFT, TRANSONIC_RIGHT, SOD_GAS, x0=TRANSONIC_X0
    )
    near = np.abs(x - TRANSONIC_X0) < 0.03  # the sonic point sits at x/t = 0
    return float(np.max(np.abs(rho[1:-1][near] - exact[near])))


def test_entropy_fix_prevents_an_entropy_violating_expansion_shock():
    """Without it the sonic-point error stops converging under refinement.

    At a sonic point the `u - c` eigenvalue vanishes, so the Roe dissipation for
    that wave vanishes with it and the scheme admits a stationary expansion
    shock -- an entropy-violating weak solution. The Harten fix floors the
    eigenvalue and removes it.

    Nothing in this suite exercised that until now: Sod has no sonic point (its
    `u - c` never changes sign), so deleting the entropy fix altogether moved
    Sod's L1 error by 0.001% and every test still passed. This test uses a
    genuinely transonic rarefaction instead.

    First order is used deliberately: at second order the MUSCL reconstruction
    masks most of the effect (1.22x rather than 4.8x), which is exactly why the
    default configuration hid it.
    """
    with_fix = (_sonic_point_error(100, 0.05), _sonic_point_error(400, 0.05))
    without = (_sonic_point_error(100, 0.0), _sonic_point_error(400, 0.0))

    # with the fix, refining the mesh reduces the error
    assert with_fix[1] < 0.5 * with_fix[0], f"expected convergence, got {with_fix}"
    # without it, refining does not help -- the expansion shock does not vanish
    assert without[1] > 0.8 * without[0], f"expected stagnation, got {without}"
    # and at the finer mesh the difference is several-fold
    assert without[1] > 3.0 * with_fix[1], f"fix={with_fix[1]:.3e} nofix={without[1]:.3e}"


def test_entropy_fix_is_inactive_where_there_is_no_sonic_point():
    """Documents why Sod could never have caught this.

    The fix should do almost nothing on a problem whose rarefaction is entirely
    subsonic or entirely supersonic -- and that is precisely Sod.
    """
    errors = []
    for entropy_fix in (0.0, 0.05, 0.2):
        grid = Grid.uniform(0.0, 1.0, 200, 1.0)
        x = grid.x_cell
        solver = Solver(
            grid,
            SOD_GAS,
            Transmissive(),
            ReferenceState(rho=1.0, u=1.0, p=1.0),
            SolverConfig(cfl=0.8, order=2, entropy_fix=entropy_fix),
        )
        solver.set_state(
            rho=np.where(x < 0.5, SOD_LEFT.rho, SOD_RIGHT.rho),
            u=np.where(x < 0.5, SOD_LEFT.u, SOD_RIGHT.u),
            p=np.where(x < 0.5, SOD_LEFT.p, SOD_RIGHT.p),
        )
        solver.run(max_steps=20_000, t_end=0.2)
        rho, _, _, _ = solver.primitives()
        exact, _, _ = sample_profile(x, 0.2, SOD_LEFT, SOD_RIGHT, SOD_GAS)
        errors.append(float(np.mean(np.abs(rho[1:-1] - exact))))

    spread = (max(errors) - min(errors)) / min(errors)
    assert spread < 0.05, f"entropy_fix should barely matter on Sod, spread {spread:.3f}"
