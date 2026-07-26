"""Phase 1 gate — the 0D reference must be trustworthy standing alone.

Reference values come from two independent places:

* closed-form algebra evaluated during review (``PLAN.md`` §3.1, §3.2), and
* the frozen ``legacy/`` scripts, whose outputs are recorded here to full
  double precision.

Nothing in this file imports ``legacy/`` — those scripts execute on import and
write files. The legacy numbers are pinned as literals instead, so a change in
either implementation shows up as a test failure rather than as agreement by
construction.
"""

import math

import pytest

from q1d.analytic import (
    CompressorSolution,
    InfeasibleOperatingPoint,
    InletChokeLimited,
    choked_mass_flow,
    compressor_exit_stagnation,
    compressor_source_terms,
    d_flow_function,
    flow_function,
    mach_from_flow_function,
    max_flow_function,
    minimum_back_pressure,
    state_from_flux,
    static_from_stagnation,
    zero_d_compressor,
)
from q1d.gas import AIR_LEGACY, PerfectGas

GAS = AIR_LEGACY

# --- the standard test case (PLAN.md §3.1) ---------------------------------
P01 = 101325.0
T01 = 288.15
PB = 101325.0
PR = 1.2
ETA = 0.9
A1 = A2 = 0.1

# --- values printed by legacy/source_map.py, full precision ----------------
LEGACY_W_FF = 21.490742688146575
LEGACY_T02 = 305.27011981156431
LEGACY_P02 = 121590.0
LEGACY_RGAS = 287.14285714285705
# (W, Fx, SWx) sampled from Sources.npy
LEGACY_MAP_POINTS = [
    (0.0, 2026.5, 0.0),
    (5.808080808080808, 2010.1233190944911, 99932.2145061388),
    (11.616161616161616, 1957.7568148074133, 199864.4290122776),
    (17.424242424242426, 1855.899874184983, 299796.64351841639),
    (23.0, 1660.4446232133303, 395731.56944430963),
]
# legacy setStatic(T0, P0[kPa], W, A) -> (Ps[kPa], V)
LEGACY_SETSTATIC = [
    (288.15, 101325.0, 21.490742688146575, 0.1, 75331.211805093446, 216.87514419193988),
    (288.15, 101325.0, 10.0, 0.1, 97048.727977574345, 84.212447498441989),
]


# ===========================================================================
# Flow function
# ===========================================================================


def test_flow_function_endpoints():
    assert flow_function(0.0, GAS) == 0.0
    # Phi(1) = sqrt(gamma) * ((gamma+1)/2) ** (-(gamma+1)/(2(gamma-1)))
    expected = math.sqrt(1.4) * (1.2 ** (-3.0))
    assert max_flow_function(GAS) == pytest.approx(expected, rel=1e-15)


def test_flow_function_is_unimodal_with_peak_at_sonic():
    phi_max = max_flow_function(GAS)
    for M in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01, 1.2, 2.0, 4.0]:
        assert flow_function(M, GAS) < phi_max
    # strictly increasing below 1, strictly decreasing above
    sub = [flow_function(m / 20.0, GAS) for m in range(21)]
    assert all(b > a for a, b in zip(sub, sub[1:], strict=False))
    sup = [flow_function(1.0 + m / 10.0, GAS) for m in range(21)]
    assert all(b < a for a, b in zip(sup, sup[1:], strict=False))


def test_derivative_matches_finite_difference():
    h = 1e-6
    for M in [0.05, 0.2, 0.5, 0.8, 0.95, 1.05, 1.5, 3.0]:
        fd = (flow_function(M + h, GAS) - flow_function(M - h, GAS)) / (2 * h)
        assert d_flow_function(M, GAS) == pytest.approx(fd, rel=1e-8)


def test_derivative_vanishes_exactly_at_sonic():
    # The (1 - M^2) factor. This is the ill-conditioning of PLAN.md §4.4,
    # and it is a property of the physics, not of the inversion algorithm.
    assert d_flow_function(1.0, GAS) == 0.0


# ===========================================================================
# Mach inversion
# ===========================================================================


@pytest.mark.parametrize("M_true", [1e-6, 0.05, 0.2, 0.5, 0.66516, 0.9, 0.99])
def test_subsonic_inversion_roundtrip(M_true):
    phi = flow_function(M_true, GAS)
    assert mach_from_flow_function(phi, GAS) == pytest.approx(M_true, rel=1e-11)


@pytest.mark.parametrize("M_true", [1.01, 1.2, 2.0, 3.5, 6.0])
def test_supersonic_inversion_roundtrip(M_true):
    phi = flow_function(M_true, GAS)
    got = mach_from_flow_function(phi, GAS, supersonic=True)
    assert got == pytest.approx(M_true, rel=1e-10)


def test_both_branches_returned_for_the_same_flow_function():
    phi = flow_function(0.4, GAS)
    assert mach_from_flow_function(phi, GAS) < 1.0
    assert mach_from_flow_function(phi, GAS, supersonic=True) > 1.0


def test_inversion_at_and_near_choke():
    phi_max = max_flow_function(GAS)
    assert mach_from_flow_function(phi_max, GAS) == pytest.approx(1.0, rel=1e-12)
    # Phi ~ Phi_max - 0.57 (M-1)^2, so a relative perturbation d in phi moves M
    # by O(sqrt(d)). At d = 1e-12 that is ~1e-6 -- inherent, not algorithmic.
    M = mach_from_flow_function(phi_max * (1 - 1e-12), GAS)
    assert 0.999 < M < 1.0
    assert abs(flow_function(M, GAS) - phi_max * (1 - 1e-12)) < 1e-12 * phi_max


def test_inversion_rejects_infeasible_flow_function():
    with pytest.raises(InfeasibleOperatingPoint):
        mach_from_flow_function(max_flow_function(GAS) * 1.001, GAS)
    with pytest.raises(ValueError):
        mach_from_flow_function(-0.1, GAS)


def test_inversion_rejects_nan():
    """NaN compares False against everything, so it needs an explicit guard.

    Without one it slips past both range checks and the bracket collapses onto
    M = 1, returning 0.9999999999999929 for garbage input — the worst possible
    failure mode for a routine that arbitrates a PDE solver.
    """
    nan = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        mach_from_flow_function(nan, GAS)
    with pytest.raises(ValueError, match="NaN"):
        static_from_stagnation(T01, P01, nan, A1, GAS)
    with pytest.raises(ValueError, match="NaN"):
        zero_d_compressor(P01, T01, nan, PR, ETA, A1, A2, GAS)


def test_inversion_converges_at_newton_rate_not_bisection_rate():
    """Guards the *algorithm*, not just its answer.

    Bisection on [0, 1] needs ~47 iterations to reach 1e-14; Newton needs a
    handful. Capping maxiter well below the bisection requirement means a
    reversed Newton step, a disabled Newton step, or a degradation to pure
    bisection all show up as a RuntimeError rather than passing silently.
    """
    for M_true in [0.05, 0.2, 0.5, 0.6647817795154938, 0.9]:
        phi = flow_function(M_true, GAS)
        got = mach_from_flow_function(phi, GAS, maxiter=10)
        assert got == pytest.approx(M_true, rel=1e-11)
    for M_true in [1.2, 2.0, 4.0, 6.0]:
        phi = flow_function(M_true, GAS)
        got = mach_from_flow_function(phi, GAS, supersonic=True, maxiter=10)
        assert got == pytest.approx(M_true, rel=1e-10)
    # Confirm the cap is genuinely below what bisection alone would need, so
    # the test above could actually fail. M = 0.99 is the slowest subsonic
    # case (13 steps); M = 0.5 would be a useless probe since it is the
    # starting midpoint and converges immediately.
    with pytest.raises(RuntimeError, match="failed to converge"):
        mach_from_flow_function(flow_function(0.99, GAS), GAS, maxiter=4)


def test_inversion_works_for_other_gases():
    for gas in [PerfectGas(1.33, 1150.0), PerfectGas(1.667, 5193.0)]:
        for M_true in [0.15, 0.6, 0.95, 2.5]:
            phi = flow_function(M_true, gas)
            got = mach_from_flow_function(phi, gas, supersonic=M_true > 1)
            assert got == pytest.approx(M_true, rel=1e-10)


# ===========================================================================
# Station states
# ===========================================================================


@pytest.mark.parametrize("W", [0.0, 1.0, 10.0, 21.490742688146575, 24.0])
def test_static_state_is_thermodynamically_self_consistent(W):
    st = static_from_stagnation(T01, P01, W, A1, GAS)
    # stagnation definitions recovered
    assert st.T + st.u**2 / (2 * GAS.cp) == pytest.approx(T01, rel=1e-13)
    assert st.p * (T01 / st.T) ** GAS.g_over_gm1 == pytest.approx(P01, rel=1e-12)
    # equation of state and mass conservation
    assert st.p == pytest.approx(st.rho * GAS.R * st.T, rel=1e-13)
    assert st.rho * st.u * A1 == pytest.approx(W, rel=1e-11, abs=1e-14)
    # Mach consistency
    assert st.u == pytest.approx(st.M * st.c, rel=1e-14)
    assert st.c == pytest.approx(GAS.speed_of_sound(st.T), rel=1e-14)


def test_zero_flow_gives_the_stagnation_state():
    st = static_from_stagnation(T01, P01, 0.0, A1, GAS)
    assert st.p == pytest.approx(P01, rel=1e-15)
    assert st.T == pytest.approx(T01, rel=1e-15)
    assert st.u == 0.0
    assert st.M == 0.0


@pytest.mark.parametrize("T0,p0,W,A,ps_expected,v_expected", LEGACY_SETSTATIC)
def test_static_state_matches_legacy_setStatic(T0, p0, W, A, ps_expected, v_expected):
    # The legacy fixed-point iteration converges to |dT/T| < 1e-10, which
    # limits the agreement to ~1e-9 relative. Newton is exact to 1e-13.
    st = static_from_stagnation(T0, p0, W, A, GAS)
    assert st.p == pytest.approx(ps_expected, rel=1e-9)
    assert st.u == pytest.approx(v_expected, rel=1e-9)


def test_static_state_rejects_infeasible_mass_flow():
    W_max = choked_mass_flow(P01, T01, A1, GAS)
    static_from_stagnation(T01, P01, W_max, A1, GAS)  # exactly sonic is fine
    with pytest.raises(InfeasibleOperatingPoint):
        static_from_stagnation(T01, P01, W_max * 1.01, A1, GAS)


def test_choke_limits_match_review_values():
    # PLAN.md §3.1
    assert choked_mass_flow(P01, T01, A1, GAS) == pytest.approx(24.1201, abs=1e-4)
    assert choked_mass_flow(LEGACY_P02, LEGACY_T02, A2, GAS) == pytest.approx(28.1208, abs=1e-4)


def test_choked_mass_flow_matches_the_textbook_formula():
    """External check, not a round trip.

    ``test_choked_mass_flow_is_exactly_sonic`` (removed) was circular:
    ``choked_mass_flow`` multiplies by ``max_flow_function`` and
    ``static_from_stagnation`` divides it straight back out, so replacing
    ``max_flow_function`` with any other value left the test passing. Compare
    against the independent closed form instead:

        W_choke = p0*A*sqrt(gamma/(R*T0)) * (2/(gamma+1))**((gamma+1)/(2(gamma-1)))
    """
    g, R = GAS.gamma, GAS.R
    for p0, T0, A in [(P01, T01, A1), (LEGACY_P02, LEGACY_T02, A2), (60000.0, 250.0, 0.37)]:
        expected = (
            p0 * A * math.sqrt(g / (R * T0)) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        )
        assert choked_mass_flow(p0, T0, A, GAS) == pytest.approx(expected, rel=1e-14)
    # and pinned absolutely, so a change in either route is visible
    assert choked_mass_flow(P01, T01, A1, GAS) == pytest.approx(24.12007042162616, rel=1e-13)
    assert choked_mass_flow(LEGACY_P02, LEGACY_T02, A2, GAS) == pytest.approx(
        28.120755274, rel=1e-10
    )


def test_choked_mass_flow_is_exactly_sonic():
    W = choked_mass_flow(P01, T01, A1, GAS)
    st = static_from_stagnation(T01, P01, W, A1, GAS)
    assert st.M == pytest.approx(1.0, rel=1e-12)
    # pin the sonic static state externally too, so the round trip cannot be
    # satisfied by an arbitrary choking criterion
    assert st.T == pytest.approx(T01 / (1.0 + 0.5 * GAS.gm1), rel=1e-13)
    assert st.p == pytest.approx(P01 * (2.0 / GAS.gp1) ** GAS.g_over_gm1, rel=1e-13)


# ===========================================================================
# Compressor stagnation states and source terms
# ===========================================================================


def test_exit_stagnation_matches_legacy():
    T02, p02 = compressor_exit_stagnation(T01, P01, PR, ETA, GAS)
    assert T02 == pytest.approx(LEGACY_T02, rel=1e-14)
    assert p02 == pytest.approx(LEGACY_P02, rel=1e-15)


def test_exit_stagnation_rejects_bad_efficiency():
    for bad in [0.0, -0.1, 1.5]:
        with pytest.raises(ValueError):
            compressor_exit_stagnation(T01, P01, PR, bad, GAS)


@pytest.mark.parametrize("W,Fx_legacy,SWx_legacy", LEGACY_MAP_POINTS)
def test_source_terms_match_legacy_map(W, Fx_legacy, SWx_legacy):
    Fx, SWx = compressor_source_terms(T01, P01, W, PR, ETA, A1, A2, GAS)
    assert Fx == pytest.approx(Fx_legacy, rel=1e-7)
    assert SWx == pytest.approx(SWx_legacy, rel=1e-12, abs=1e-9)


def test_blade_force_at_zero_flow_is_the_pressure_difference():
    Fx, SWx = compressor_source_terms(T01, P01, 0.0, PR, ETA, A1, A2, GAS)
    assert Fx == pytest.approx((LEGACY_P02 - P01) * A1, rel=1e-14)
    assert SWx == 0.0


def test_blade_force_is_positive_everywhere_and_never_crosses_zero():
    # PLAN.md §4.3: a rejected review finding claimed the absence of an Fx
    # zero-crossing proved no equilibrium exists. Fx is the force ON the fluid;
    # a compressor always pushes. Equilibrium is a vanishing residual.
    forces = [compressor_source_terms(T01, P01, W, PR, ETA, A1, A2, GAS)[0] for W in range(0, 24)]
    assert all(f > 0 for f in forces)
    assert all(b < a for a, b in zip(forces, forces[1:], strict=False))  # monotonically decreasing


# ===========================================================================
# The 0D solution — the Phase 3 acceptance target
# ===========================================================================


@pytest.fixture
def solution() -> CompressorSolution:
    return zero_d_compressor(P01, T01, PB, PR, ETA, A1, A2, GAS)


def test_the_reference_operating_point(solution):
    # PLAN.md §3.1 -- this is the number every later phase is checked against.
    assert solution.W == pytest.approx(21.490742688146575, rel=1e-12)
    assert solution.regime == "subsonic"
    assert solution.station1.M == pytest.approx(0.6647817795154938, rel=1e-12)
    assert solution.station2.M == pytest.approx(0.5170711949922851, rel=1e-12)
    assert solution.p02 == pytest.approx(121590.0, rel=1e-14)
    assert solution.T02 == pytest.approx(305.27011981156431, rel=1e-14)
    assert solution.tau == pytest.approx(1.05941391570906, rel=1e-13)
    assert solution.Fx == pytest.approx(1731.2433536618760, rel=1e-12)
    assert solution.SWx == pytest.approx(369763.7101088717, rel=1e-12)


def test_W_ff_identity(solution):
    """The legacy 'choked flow reference' is the answer to the test case.

    ``legacy/source_map.py`` computes ``W_ff`` from the compressible flow
    function at the imposed PR and labels it an ideal *choked* mass flow. It is
    neither ideal nor choked (M2 = 0.517): it is the exact steady mass flow of
    the duct when the exit static pressure equals p01, which is precisely the
    back-pressure condition ``legacy/q1d_solver.py`` imposes.
    """
    gam1, gamma, rgas = GAS.gm1, GAS.gamma, GAS.R
    ff = (2 * gamma / gam1 * PR ** (-2 / gamma) * (1 - PR ** (-gam1 / gamma))) ** 0.5
    W_ff = ff * A2 * (P01 * PR) / (rgas * LEGACY_T02) ** 0.5
    assert W_ff == pytest.approx(LEGACY_W_FF, rel=1e-14)
    assert solution.W == pytest.approx(W_ff, rel=1e-12)
    assert solution.station2.M < 1.0  # nothing is choked


def test_exit_static_pressure_equals_the_back_pressure(solution):
    # Self-consistency of the matching: this is what makes the case closed-form.
    assert solution.station2.p == pytest.approx(PB, rel=1e-12)


def test_mass_flow_is_consistent_at_both_stations(solution):
    assert solution.station1.rho * solution.station1.u * A1 == pytest.approx(solution.W, rel=1e-11)
    assert solution.station2.rho * solution.station2.u * A2 == pytest.approx(solution.W, rel=1e-11)


def test_flow_function_relation_between_stations(solution):
    # Phi2 = Phi1 * (A1/A2) * sqrt(tau) / PR  (PLAN.md §3.2)
    expected = solution.phi1 * (A1 / A2) * math.sqrt(solution.tau) / PR
    assert solution.phi2 == pytest.approx(expected, rel=1e-12)


def test_operating_point_is_comfortably_subsonic(solution):
    assert solution.W / choked_mass_flow(P01, T01, A1, GAS) == pytest.approx(0.891, abs=1e-3)


# ===========================================================================
# The identity that makes the Phase 3 gate achievable
# ===========================================================================


def test_source_terms_reconstruct_station_2_from_station_1(solution):
    """Applying (Fx, SWx) to station 1 must *produce* station 2.

    This is the property the Q1D solver relies on: because global discrete
    conservation is exact, the jump across the smeared disk region is exactly
    (Fx, SWx), so the downstream uniform state is whatever this reconstruction
    says it is. If this identity did not hold, the 1e-8 Phase 3 gate would be
    unreachable no matter how good the discretisation.

    Solved here independently of ``analytic.py``: mass, momentum and energy
    balances are combined into a quadratic for the exit velocity.
    """
    st1, W = solution.station1, solution.W
    cp, R = GAS.cp, GAS.R

    momentum_flux_out = W * st1.u + st1.p * A1 + solution.Fx
    h02 = cp * T01 + solution.SWx / W

    # a*u^2 + b*u + c = 0, from  cp*T2 + u2^2/2 = h02  with
    # T2 = (momentum_flux_out - W*u2) * u2 / (W*R)  and  rho2 = W/(u2*A2)
    a = 0.5 - cp / R
    b = cp * momentum_flux_out / (W * R)
    c = -h02
    disc = b * b - 4 * a * c
    assert disc > 0
    roots = [(-b + s * math.sqrt(disc)) / (2 * a) for s in (1.0, -1.0)]

    subsonic = []
    for u2 in roots:
        if u2 <= 0:
            continue
        T2 = (momentum_flux_out - W * u2) * u2 / (W * R)
        if T2 <= 0:
            continue
        if u2 / math.sqrt(GAS.gamma * R * T2) < 1.0:
            subsonic.append((u2, T2))
    assert len(subsonic) == 1, "expected exactly one subsonic branch"

    u2, T2 = subsonic[0]
    rho2 = W / (u2 * A2)
    p2 = rho2 * R * T2

    assert u2 == pytest.approx(solution.station2.u, rel=1e-10)
    assert T2 == pytest.approx(solution.station2.T, rel=1e-10)
    assert p2 == pytest.approx(solution.station2.p, rel=1e-10)
    # and the recovered stagnation state is the map's
    assert T2 + u2**2 / (2 * cp) == pytest.approx(solution.T02, rel=1e-11)
    assert p2 * (1 + 0.5 * GAS.gm1 * (u2**2 / (GAS.gamma * R * T2))) ** GAS.g_over_gm1 == (
        pytest.approx(solution.p02, rel=1e-10)
    )


# ===========================================================================
# Inlet invariance (PLAN.md §3.2)
# ===========================================================================

INLET_CONDITIONS = [(101325.0, 288.15), (60000.0, 250.0), (150000.0, 320.0), (40000.0, 220.0)]


def test_dimensionless_source_terms_collapse_across_inlet_conditions():
    """F-hat and S-hat are functions of Phi1 alone for a perfect gas.

    Verified to machine precision across inlet conditions whose raw Fx differ
    by a factor of ~4. This is what makes the Phase 3 ambient sweep a test of
    the *interface* rather than a re-test of the physics -- and it is exactly
    the property that dies under NASA9 (PLAN.md §3.3), which is why the source
    terms are computed at runtime rather than tabulated.
    """
    for M1 in [0.2, 0.4, 0.6, 0.66516, 0.8]:
        phi1 = flow_function(M1, GAS)
        fhats, shats, raw = [], [], []
        for p01, T0 in INLET_CONDITIONS:
            W = phi1 * A1 * p01 / math.sqrt(GAS.R * T0)
            Fx, SWx = compressor_source_terms(T0, p01, W, PR, ETA, A1, A2, GAS)
            fhats.append(Fx / (p01 * A1))
            shats.append(SWx / (W * GAS.cp * T0))
            raw.append(Fx)
        spread = (max(fhats) - min(fhats)) / abs(sum(fhats) / len(fhats))
        assert spread < 1e-12, f"F-hat failed to collapse at M1={M1}: spread {spread:.3e}"
        assert max(shats) - min(shats) < 1e-14
        assert max(raw) / min(raw) > 3.0, "inlet conditions were not varied enough to be a test"


def test_Shat_equals_tau_minus_one(solution):
    assert solution.Shat == pytest.approx(solution.tau - 1.0, rel=1e-13)


def test_dimensionless_coefficients_have_pinned_absolute_values(solution):
    """The collapse test alone cannot see a wrong Fx.

    A sign-flipped momentum term, or Fx scaled by an arbitrary constant, still
    collapses perfectly across inlet conditions — the collapse only tests
    dimensional homogeneity. Pinning the absolute values closes that gap, and
    catches the ``p01`` vs ``p02`` and ``A1`` vs ``A2`` denominator mutations.
    """
    assert solution.Fhat == pytest.approx(0.17086043460763642, rel=1e-12)
    assert solution.Shat == pytest.approx(0.05941391570905539, rel=1e-12)
    assert solution.Fhat == pytest.approx(solution.Fx / (solution.p01 * solution.A1), rel=1e-15)


def test_momentum_source_with_unequal_areas_includes_the_wall_reaction():
    """With A1 != A2 the momentum-flux jump is not the blade force.

    Integrating ``d/dx[(rho u^2 + p)A] = p dA/dx + f_blade`` gives
    ``Delta[(rho u^2 + p)A] = int p dA + F_blade``. For a contraction the wall
    term dominates and the total goes *negative*, which is why
    ``compressor_source_terms`` refuses unequal areas unless the caller states
    how the wall term is accounted for.
    """
    with pytest.raises(ValueError, match="wall_pressure_integral"):
        compressor_source_terms(T01, P01, 10.0, PR, ETA, 0.1, 0.05, GAS)

    Fx, _ = compressor_source_terms(
        T01, P01, 10.0, PR, ETA, 0.1, 0.05, GAS, wall_pressure_integral=0.0
    )
    assert Fx == pytest.approx(-3709.9098, rel=1e-6)  # negative, and pinned

    # subtracting a wall integral shifts it exactly, and the A1/A2 roles are
    # not interchangeable (this kills the swap mutation)
    Fx_shift, _ = compressor_source_terms(
        T01, P01, 10.0, PR, ETA, 0.1, 0.05, GAS, wall_pressure_integral=500.0
    )
    assert Fx_shift == pytest.approx(Fx - 500.0, rel=1e-12)
    Fx_swapped, _ = compressor_source_terms(
        T01, P01, 10.0, PR, ETA, 0.05, 0.1, GAS, wall_pressure_integral=0.0
    )
    assert abs(Fx_swapped - Fx) > 1000.0


def test_solution_is_invariant_in_dimensionless_form():
    ref = zero_d_compressor(P01, T01, PB, PR, ETA, A1, A2, GAS)
    # scale the whole problem: same PR and same pb/p01, different absolute level
    for scale_p, T0 in [(0.6, 250.0), (1.5, 320.0)]:
        other = zero_d_compressor(P01 * scale_p, T0, PB * scale_p, PR, ETA, A1, A2, GAS)
        assert other.phi1 == pytest.approx(ref.phi1, rel=1e-12)
        assert other.station1.M == pytest.approx(ref.station1.M, rel=1e-12)
        assert other.station2.M == pytest.approx(ref.station2.M, rel=1e-12)
        assert other.Fhat == pytest.approx(ref.Fhat, rel=1e-12)
        assert other.Shat == pytest.approx(ref.Shat, rel=1e-12)


# ===========================================================================
# Regimes and failure modes
# ===========================================================================


def test_back_pressure_above_exit_stagnation_has_no_forward_flow():
    with pytest.raises(InfeasibleOperatingPoint, match="no forward flow"):
        zero_d_compressor(P01, T01, 1.5 * P01 * PR, PR, ETA, A1, A2, GAS)


def test_with_equal_areas_the_inlet_always_limits_before_the_exit():
    """A constant-area duct can never choke at the exit first.

    Exit choke requires Phi2 = Phi_max, which via
    ``Phi2 = Phi1 (A1/A2) sqrt(tau) / PR`` demands
    ``Phi1/Phi_max = (A2/A1) * PR/sqrt(tau)``. With A1 == A2 and PR = 1.2 that
    is 1.166 > 1, so station 1 saturates first. The exit can only choke first
    if ``A2/A1 < sqrt(tau)/PR = 0.8577``.
    """
    tau = compressor_exit_stagnation(T01, P01, PR, ETA, GAS)[0] / T01
    area_ratio_threshold = math.sqrt(tau) / PR
    assert area_ratio_threshold == pytest.approx(0.857732, rel=1e-5)

    # equal areas, back pressure driven right down: inlet-limited, not exit-choked
    with pytest.raises(InletChokeLimited, match="inlet choke limit"):
        zero_d_compressor(P01, T01, 0.3 * P01, PR, ETA, A1, A2, GAS)


def test_low_back_pressure_chokes_a_contracted_exit():
    # A2/A1 = 0.5 is below the 0.8577 threshold, so the exit chokes first.
    sol = zero_d_compressor(P01, T01, 0.3 * P01, PR, ETA, 0.1, 0.05, GAS)
    assert sol.regime == "exit_choked"
    assert sol.station2.M == pytest.approx(1.0, rel=1e-12)
    assert sol.station1.M < 1.0
    # exit static pressure is then set by the choke, not by pb
    assert sol.station2.p > 0.3 * P01
    assert sol.W == pytest.approx(choked_mass_flow(sol.p02, sol.T02, 0.05, GAS), rel=1e-12)


def test_the_exit_choke_regime_boundary_is_sharp():
    """Pin the M2 = 1 threshold itself, not just a point far past it.

    The only prior exit-choke test drove M2 to 1.56, so a threshold moved to
    0.95 or 1.05 went undetected — silently clamping a range of genuinely
    subsonic exits to sonic, or letting supersonic demands through.
    """
    A2_small = 0.05
    T02, p02 = compressor_exit_stagnation(T01, P01, PR, ETA, GAS)
    pb_crit = p02 * (2.0 / GAS.gp1) ** GAS.g_over_gm1  # exit static at M2 = 1

    just_subsonic = zero_d_compressor(P01, T01, pb_crit * 1.001, PR, ETA, A1, A2_small, GAS)
    assert just_subsonic.regime == "subsonic"
    assert 0.98 < just_subsonic.station2.M < 1.0

    just_choked = zero_d_compressor(P01, T01, pb_crit * 0.999, PR, ETA, A1, A2_small, GAS)
    assert just_choked.regime == "exit_choked"
    assert just_choked.station2.M == pytest.approx(1.0, rel=1e-12)

    # mass flow is continuous across the boundary and pins at the choke value
    assert just_choked.W == pytest.approx(just_subsonic.W, rel=2e-3)
    assert just_choked.W == pytest.approx(choked_mass_flow(p02, T02, A2_small, GAS), rel=1e-12)


def test_inlet_choke_limit_is_reported_with_its_boundary():
    pb_min = minimum_back_pressure(P01, T01, PR, ETA, A1, A2, GAS)
    zero_d_compressor(P01, T01, pb_min * 1.0001, PR, ETA, A1, A2, GAS)  # just inside
    with pytest.raises(InletChokeLimited) as exc:
        zero_d_compressor(P01, T01, pb_min * 0.999, PR, ETA, A1, A2, GAS)
    assert exc.value.W_choke == pytest.approx(24.12007042162616, rel=1e-12)
    assert exc.value.pb_min == pytest.approx(pb_min, rel=1e-15)
    assert pb_min == pytest.approx(93839.1, abs=1.0)


def test_high_pressure_ratio_can_make_the_inlet_infeasible():
    # A large PR with A1 == A2 demands more flow than station 1 can pass.
    with pytest.raises(InletChokeLimited, match="inlet choke limit"):
        zero_d_compressor(P01, T01, PB, 3.0, ETA, A1, A2, GAS)


def test_larger_inlet_area_rescues_the_same_pressure_ratio():
    sol = zero_d_compressor(P01, T01, PB, 3.0, ETA, 0.4, A2, GAS)
    assert sol.station1.M < 1.0
    assert sol.W > 0


def test_rejects_nonpositive_geometry_and_state():
    with pytest.raises(ValueError):
        static_from_stagnation(T01, P01, 1.0, 0.0, GAS)
    with pytest.raises(ValueError):
        static_from_stagnation(T01, -1.0, 1.0, A1, GAS)
    with pytest.raises(ValueError):
        zero_d_compressor(P01, T01, -1.0, PR, ETA, A1, A2, GAS)


# ===========================================================================
# Monotonicity — sanity of the model across an operating range
# ===========================================================================


def test_mass_flow_increases_as_back_pressure_falls():
    W_prev = 0.0
    for pb in [121000.0, 115000.0, 110000.0, 105000.0, 101325.0]:
        sol = zero_d_compressor(P01, T01, pb, PR, ETA, A1, A2, GAS)
        assert sol.W > W_prev
        W_prev = sol.W


def test_shaft_power_rises_with_pressure_ratio_at_fixed_flow():
    prev = 0.0
    for pr in [1.05, 1.1, 1.2, 1.4]:
        _, SWx = compressor_source_terms(T01, P01, 10.0, pr, ETA, A1, A2, GAS)
        assert SWx > prev
        prev = SWx


def test_lower_efficiency_costs_more_power_for_the_same_pressure_ratio():
    _, sw_good = compressor_source_terms(T01, P01, 10.0, PR, 0.9, A1, A2, GAS)
    _, sw_poor = compressor_source_terms(T01, P01, 10.0, PR, 0.7, A1, A2, GAS)
    assert sw_poor > sw_good
    assert sw_poor / sw_good == pytest.approx(0.9 / 0.7, rel=1e-12)


# -- flux inversion ---------------------------------------------------------


class TestStateFromFlux:
    """``state_from_flux`` is the inverse of the flux function."""

    @staticmethod
    def _flux(st, A, gas):
        return (
            st.rho * st.u * A,
            (st.rho * st.u**2 + st.p) * A,
            st.rho * st.u * (gas.cp * st.T + 0.5 * st.u**2) * A,
        )

    @pytest.mark.parametrize("M", [0.05, 0.2, 0.45, 0.7, 0.95])
    @pytest.mark.parametrize("A", [0.3, 1.0, 7.5])
    def test_round_trips_the_subsonic_branch(self, M, A):
        gas = GAS
        st = static_from_stagnation(
            300.0, 2.0e5, flow_function(M, gas) * A * 2.0e5 / math.sqrt(gas.R * 300.0), A, gas
        )
        got = state_from_flux(self._flux(st, A, gas), A, gas)
        assert got.rho == pytest.approx(st.rho, rel=1e-12)
        assert got.u == pytest.approx(st.u, rel=1e-12)
        assert got.p == pytest.approx(st.p, rel=1e-12)
        assert got.M == pytest.approx(M, rel=1e-10)

    @pytest.mark.parametrize("M", [1.2, 2.0, 3.5])
    def test_round_trips_the_supersonic_branch(self, M):
        gas = GAS
        A = 1.0
        st = static_from_stagnation(
            300.0,
            2.0e5,
            flow_function(M, gas) * A * 2.0e5 / math.sqrt(gas.R * 300.0),
            A,
            gas,
            supersonic=True,
        )
        got = state_from_flux(self._flux(st, A, gas), A, gas, supersonic=True)
        assert got.M == pytest.approx(M, rel=1e-9)

    def test_the_two_roots_are_the_two_branches_of_the_same_flux(self):
        """One flux vector, two states — the Rankine-Hugoniot pair.

        Both exist only where the supersonic partner has positive pressure. A
        deeply subsonic state has no conjugate: its partner would need
        ~2300 m/s and a negative static pressure, and is correctly discarded.
        """
        gas = GAS
        A = 1.0
        W = flow_function(1.5, gas) * A * 2.0e5 / math.sqrt(gas.R * 300.0)
        sup = static_from_stagnation(300.0, 2.0e5, W, A, gas, supersonic=True)
        f = self._flux(sup, A, gas)
        lo = state_from_flux(f, A, gas)
        hi = state_from_flux(f, A, gas, supersonic=True)
        assert lo.M < 1.0 < hi.M
        # same mass, momentum and energy flux, different state
        for got in (lo, hi):
            assert self._flux(got, A, gas)[0] == pytest.approx(f[0], rel=1e-12)
            assert self._flux(got, A, gas)[1] == pytest.approx(f[1], rel=1e-12)
            assert self._flux(got, A, gas)[2] == pytest.approx(f[2], rel=1e-12)

    def test_rejects_a_flux_no_state_can_produce(self):
        with pytest.raises(InfeasibleOperatingPoint):
            state_from_flux((10.0, 1.0, 1e12), 1.0, GAS)

    def test_rejects_non_positive_mass_flux_and_area(self):
        with pytest.raises(ValueError, match="mass flux"):
            state_from_flux((0.0, 1.0, 1.0), 1.0, GAS)
        with pytest.raises(ValueError, match="area"):
            state_from_flux((1.0, 1.0, 1.0), 0.0, GAS)
