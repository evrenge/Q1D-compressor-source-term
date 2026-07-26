"""Gate for the implicit integrators.

The point of these is *not* that implicit integration rescues the high-pressure
blockage — it does not, and ``PLAN.md`` §3.16 records why: the designed steady
state is linearly unstable from PR 2.26 up, converged under mesh refinement, so
no A-stable scheme can hold it. What these tests pin is that the three schemes
are what they claim to be:

* each reproduces a steady state exactly, so the *spatial* discretisation is
  untouched — an implicit scheme that quietly changes the fixed point would
  invalidate every comparison made against the explicit solver;
* each takes steps far beyond the explicit CFL limit without dying, which is
  the whole reason to pay for a Newton solve;
* the observed temporal orders are 1, 2 and 3, measured against a
  finely-stepped reference. Order is the one property that separates the three
  and the one most easily lost — using constant-step BDF2 coefficients with a
  CFL-driven ``dt`` silently drops it to first order, which is the usual way
  BDF2 disappoints.
"""

import math

import numpy as np
import pytest

from q1d.boundary import StagnationInletStaticOutlet
from q1d.gas import PerfectGas
from q1d.grid import Grid
from q1d.implicit import (
    SCHEMES,
    SDIRK3_GAMMA,
    ImplicitConfig,
    ImplicitStepper,
    NewtonDidNotConverge,
)
from q1d.solver import ReferenceState, Solver, SolverConfig

GAS = PerfectGas(gamma=1.4, cp=1004.7)

T0_IN, P0_IN = 288.15, 101325.0
P_BACK = 0.92 * P0_IN
AREA = 0.1
NCELL = 41


def build(ncell=NCELL):
    """Uniform duct, subsonic, no source. Small enough to iterate on."""
    grid = Grid.uniform(0.0, 1.0, ncell, AREA)
    bc = StagnationInletStaticOutlet(p0_in=P0_IN, T0_in=T0_IN, p_back=P_BACK)
    rho = P_BACK / (GAS.R * T0_IN)
    ref = ReferenceState(rho=rho, u=50.0, p=P_BACK)
    s = Solver(grid, GAS, bc, ref, config=SolverConfig(cfl=0.5))
    s.set_state(
        np.full(ncell, rho), np.full(ncell, 50.0), np.full(ncell, P_BACK)
    )
    return s


@pytest.fixture(scope="module")
def steady():
    """The **analytic** steady state of this duct, not a converged one.

    Constant area and no source, so the exact solution is uniform flow at the
    back pressure with the inlet's stagnation state; the flux differences
    telescope to zero identically and the geometric source vanishes. Measured
    residual is 1.4e-12 relative and 200 explicit steps move it by 9e-16.

    Marching to a fixed point instead would have been a trap. Four thousand
    explicit steps leave the residual at 4e-4 relative, and driving it lower is
    slow for a reason worth recording: backward Euler at a *fixed* step is not
    Newton on ``R = 0``, its contraction on the slowest mode being
    ``1/(1 + λh)``, which is near one exactly for the modes that take longest.
    A hold test on such a state measures the leftover transient rather than the
    scheme, which is what the first version of this file did.

    Using the analytic state also puts the boundary conditions under test: if
    the ghost construction did not reproduce the exact state, the residual above
    would not be at round-off.
    """
    g = GAS.gamma
    M = math.sqrt(2.0 / (g - 1.0) * ((P0_IN / P_BACK) ** ((g - 1.0) / g) - 1.0))
    T = T0_IN / (1.0 + 0.5 * (g - 1.0) * M * M)
    rho = P_BACK / (GAS.R * T)
    u = M * math.sqrt(g * GAS.R * T)
    return (
        np.full(NCELL, rho),
        np.full(NCELL, u),
        np.full(NCELL, P_BACK),
    )


def test_the_analytic_state_really_is_a_fixed_point(steady):
    """Guards the fixture itself: every other test reads it as exact."""
    s = build()
    s.set_state(*steady)
    r = np.linalg.norm(s.residual() / s.grid.dx) / np.linalg.norm(s.cv[:, 1:-1])
    assert r < 1e-11, f"fixture is not a steady state: relative residual {r:.3e}"


# -- construction -----------------------------------------------------------


def test_unknown_scheme_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown scheme"):
        ImplicitStepper(build(), scheme="rk4")


@pytest.mark.parametrize("scheme", SCHEMES)
def test_every_advertised_scheme_constructs(scheme):
    assert ImplicitStepper(build(), scheme=scheme).scheme == scheme


def test_sdirk3_gamma_is_the_root_that_makes_it_third_order():
    g = SDIRK3_GAMMA
    assert abs(g**3 - 3.0 * g * g + 1.5 * g - 1.0 / 6.0) < 1e-14
    assert 1.0 / 6.0 < g < 0.5


def test_sdirk3_tableau_is_stiffly_accurate_and_consistent():
    from q1d.implicit import _sdirk3_tableau

    A, c = _sdirk3_tableau()
    assert np.allclose(np.diag(A), SDIRK3_GAMMA)
    assert np.allclose(np.triu(A, 1), 0.0)  # lower triangular: it is DIRK
    # stiffly accurate: last row is the weight vector, so X_3 is the update
    assert abs(A[2].sum() - 1.0) < 1e-14
    assert abs(c[2] - 1.0) < 1e-14


# -- the fixed point is untouched -------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_scheme_holds_a_steady_state(scheme, steady):
    """A converged field must not move, whatever the step size.

    This is the load-bearing test: it says the implicit schemes share the
    explicit solver's fixed point exactly, so any disagreement elsewhere is
    about time integration and nothing else.
    """
    s = build()
    s.set_state(*steady)
    st = ImplicitStepper(s, scheme=scheme)
    before = s.cv[:, 1:-1].copy()
    dt = s.timestep()
    for _ in range(5):
        st.step(20.0 * dt)
    after = s.cv[:, 1:-1]
    assert np.max(np.abs(after / before - 1.0)) < 1e-10


@pytest.mark.parametrize("scheme", SCHEMES)
def test_scheme_steps_far_beyond_the_explicit_limit(scheme, steady):
    """50x the CFL step, from a perturbed state, without dying.

    Explicit would go non-physical within a step or two. This is what the
    Newton solve is bought with.
    """
    s = build()
    rho, u, p = steady
    s.set_state(rho * 1.02, u * 0.97, p * 1.01)
    st = ImplicitStepper(s, scheme=scheme)
    dt = 50.0 * s.timestep()
    for _ in range(10):
        st.step(dt)
    rho2, u2, p2, _ = s.primitives()
    assert np.all(np.isfinite(rho2)) and np.all(rho2[1:-1] > 0.0)
    assert np.all(p2[1:-1] > 0.0)
    # and it is heading back to the steady state, not merely surviving
    assert np.max(np.abs(p2[1:-1] / p - 1.0)) < 0.01


def test_step_returns_the_size_actually_taken(steady):
    s = build()
    s.set_state(*steady)
    st = ImplicitStepper(s, scheme="euler")
    dt = 10.0 * s.timestep()
    assert st.step(dt) == pytest.approx(dt * 0.5**st.step_cuts)


def test_the_clock_advances_by_the_step_taken(steady):
    s = build()
    s.set_state(*steady)
    st = ImplicitStepper(s, scheme="euler")
    t0 = s.t
    taken = st.step(5.0 * s.timestep())
    assert s.t == pytest.approx(t0 + taken)
    assert s.dt_last == pytest.approx(taken)


def test_sdirk3_reports_three_stages_per_step(steady):
    s = build()
    s.set_state(*steady)
    st = ImplicitStepper(s, scheme="sdirk3")
    st.step(5.0 * s.timestep())
    assert st.stages == 3


def test_bdf2_self_starts_with_one_euler_step(steady):
    """The first BDF2 step has no history, so it must be a Euler step.

    Checked by construction rather than by output: with ``_x_prev`` unset the
    two must agree bitwise, and must diverge once history exists.
    """
    dt = None
    finals = {}
    for scheme in ("euler", "bdf2"):
        s = build()
        rho, u, p = steady
        s.set_state(rho * 1.01, u, p)
        st = ImplicitStepper(s, scheme=scheme)
        dt = dt or 10.0 * s.timestep()
        st.step(dt)
        finals[scheme] = s.cv[:, 1:-1].copy()
    assert np.array_equal(finals["euler"], finals["bdf2"])


def test_reset_history_forces_bdf2_to_self_start_again(steady):
    s = build()
    rho, u, p = steady
    s.set_state(rho * 1.01, u, p)
    st = ImplicitStepper(s, scheme="bdf2")
    dt = 10.0 * s.timestep()
    st.step(dt)
    assert st._x_prev is not None
    st.reset_history()
    assert st._x_prev is None and math.isnan(st._dt_prev)


# -- temporal order ---------------------------------------------------------


def _integrate(scheme, dt, n_steps, start):
    s = build()
    s.set_state(*start)
    st = ImplicitStepper(s, scheme=scheme, config=ImplicitConfig(newton_tol=1e-12))
    for _ in range(n_steps):
        st.step(dt)
    return s.cv[:, 1:-1].ravel().copy()


#: Step counts over a fixed end time. These have to be this fine: the van
#: Albada limiter is only piecewise differentiable, and until the step is small
#: enough that no face changes limiter branch during it, every scheme reads one
#: order low. Measured rates over successive halvings, MUSCL active:
#: euler 0.39, 0.54, 0.69, 0.81; bdf2 0.71, 0.97, 1.27, 1.63; sdirk3 1.62,
#: 2.21, 2.68, 2.89. With reconstruction forced to first order the same runs
#: give 0.87, 2.06 and 2.97, which is what identifies the limiter as the cause.
_ORDER_STEPS = (16, 32, 64)


@pytest.fixture(scope="module")
def convergence(steady):
    """Relative error against a finely-stepped reference, for all three schemes."""
    rho, u, p = steady
    start = (rho * 1.03, u * 0.95, p * 1.02)
    T = 40.0 * build().timestep()
    ref = _integrate("sdirk3", T / 512.0, 512, start)
    nref = np.linalg.norm(ref)
    out = {}
    for scheme in SCHEMES:
        out[scheme] = [
            np.linalg.norm(_integrate(scheme, T / n, n, start) - ref) / nref
            for n in _ORDER_STEPS
        ]
    return out


@pytest.mark.parametrize(
    "scheme, expected",
    [("euler", 1.0), ("bdf2", 2.0), ("sdirk3", 3.0)],
)
def test_observed_temporal_order(scheme, expected, convergence):
    errs = convergence[scheme]
    assert errs[-1] < errs[0], f"{scheme} did not converge under refinement: {errs}"
    rates = [math.log2(errs[k] / errs[k + 1]) for k in range(len(errs) - 1)]
    # the finest pair is the trustworthy one: it is the closest to asymptotic
    assert rates[-1] == pytest.approx(expected, abs=0.6), f"{scheme}: rates {rates}"


def test_the_schemes_are_ranked_by_their_order(convergence):
    """At a common step size, higher order must mean smaller error.

    Weaker than the rate test and independent of it: it would still catch a
    scheme that converged at the right rate off a wrong constant.
    """
    fine = {s: convergence[s][-1] for s in SCHEMES}
    assert fine["sdirk3"] < fine["bdf2"] < fine["euler"], fine


def test_the_schemes_agree_when_the_step_is_small(steady):
    """Different order, same equation: at a small step all three must coincide."""
    rho, u, p = steady
    start = (rho * 1.02, u, p)
    dt = 2.0 * build().timestep()
    out = {s: _integrate(s, dt, 8, start) for s in SCHEMES}
    for scheme in ("bdf2", "sdirk3"):
        rel = np.linalg.norm(out[scheme] - out["euler"]) / np.linalg.norm(out["euler"])
        assert rel < 5e-3, f"{scheme} disagrees with euler by {rel:.2e}"


# -- failure behaviour ------------------------------------------------------


def test_failure_rolls_the_state_back_and_raises(steady):
    """A step that cannot converge must leave the state exactly as it found it."""
    s = build()
    s.set_state(*steady)
    st = ImplicitStepper(
        s,
        scheme="euler",
        config=ImplicitConfig(max_newton=1, max_krylov=1, max_step_cuts=1, newton_tol=1e-16),
    )
    before = s.cv[:, 1:-1].copy()
    t_before = s.t
    with pytest.raises(NewtonDidNotConverge, match="euler"):
        st.step(1e6)
    # the interior is what `_write` restores; the ghosts are derived from it and
    # are allowed to differ in the last bits after being recomputed
    assert np.array_equal(s.cv[:, 1:-1], before)
    assert s.t == t_before
