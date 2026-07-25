"""The exact Riemann solver must be trustworthy before it can arbitrate the
Phase 2 shock-tube gate.

Star-region values for the five standard Toro tests are pinned from the
published tables, so these are genuine external references rather than
self-comparison.
"""

import math

import numpy as np
import pytest

from q1d.gas import PerfectGas
from q1d.riemann import (
    SOD_LEFT,
    SOD_RIGHT,
    RiemannState,
    sample,
    sample_profile,
    solve_star,
)

GAS = PerfectGas(gamma=1.4, cp=1005.0)

# Toro, table 4.1 / 4.3: (left, right, p_star, u_star, rel_tol).
# The tolerance tracks the significant figures published for each test -- test 2
# is quoted to three, so demanding 2e-5 there would be testing round-off.
TORO_TESTS = [
    ("1 Sod", SOD_LEFT, SOD_RIGHT, 0.30313, 0.92745, 2e-5),
    (
        "2 123 problem",
        RiemannState(1.0, -2.0, 0.4),
        RiemannState(1.0, 2.0, 0.4),
        0.00189,
        0.00000,
        3e-3,
    ),
    (
        "3 left blast",
        RiemannState(1.0, 0.0, 1000.0),
        RiemannState(1.0, 0.0, 0.01),
        460.894,
        19.5975,
        2e-5,
    ),
    (
        "4 right blast",
        RiemannState(1.0, 0.0, 0.01),
        RiemannState(1.0, 0.0, 100.0),
        46.0950,
        -6.19633,
        2e-5,
    ),
    (
        "5 collision",
        RiemannState(5.99924, 19.5975, 460.894),
        RiemannState(5.99242, -6.19633, 46.0950),
        1691.64,
        8.68975,
        2e-5,
    ),
]


@pytest.mark.parametrize("name,left,right,p_star,u_star,rtol", TORO_TESTS)
def test_star_state_matches_published_values(name, left, right, p_star, u_star, rtol):
    star = solve_star(left, right, GAS)
    assert star.p == pytest.approx(p_star, rel=rtol), name
    assert star.u == pytest.approx(u_star, rel=rtol, abs=1e-5), name


def test_newton_converges_quickly():
    for _, left, right, _, _, _ in TORO_TESTS:
        assert solve_star(left, right, GAS).iterations < 15


def test_identical_states_produce_no_waves():
    s = RiemannState(1.225, 130.0, 101325.0)
    star = solve_star(s, s, GAS)
    assert star.p == pytest.approx(s.p, rel=1e-12)
    assert star.u == pytest.approx(s.u, rel=1e-12)
    for S in [-500.0, 0.0, 130.0, 500.0]:
        rho, u, p = sample(S, s, s, star, GAS)
        assert (rho, u, p) == pytest.approx((s.rho, s.u, s.p), rel=1e-12)


def test_vacuum_data_are_rejected():
    left = RiemannState(1.0, -20.0, 0.4)
    right = RiemannState(1.0, 20.0, 0.4)
    with pytest.raises(ValueError, match="vacuum"):
        solve_star(left, right, GAS)


def test_sampling_far_from_the_waves_returns_the_initial_states():
    star = solve_star(SOD_LEFT, SOD_RIGHT, GAS)
    for S in (-100.0, 100.0):
        rho, u, p = sample(S, SOD_LEFT, SOD_RIGHT, star, GAS)
        expected = SOD_LEFT if S < 0 else SOD_RIGHT
        assert (rho, u, p) == pytest.approx((expected.rho, expected.u, expected.p), rel=1e-14)


def test_sod_wave_structure():
    """Sod: left rarefaction, contact, right shock — check each speed."""
    star = solve_star(SOD_LEFT, SOD_RIGHT, GAS)
    g = GAS.gamma
    cl = SOD_LEFT.sound_speed(GAS)
    cr = SOD_RIGHT.sound_speed(GAS)

    head = SOD_LEFT.u - cl
    c_star_l = cl * (star.p / SOD_LEFT.p) ** ((g - 1) / (2 * g))
    tail = star.u - c_star_l
    shock = SOD_RIGHT.u + cr * math.sqrt(
        (g + 1) / (2 * g) * star.p / SOD_RIGHT.p + (g - 1) / (2 * g)
    )
    assert head == pytest.approx(-1.18322, rel=1e-4)
    assert tail == pytest.approx(-0.07027, abs=1e-4)
    assert shock == pytest.approx(1.75216, rel=1e-4)
    assert head < tail < star.u < shock  # ordering of the three waves

    # density is discontinuous across the contact but pressure and velocity are not
    eps = 1e-9
    rho_l, u_l, p_l = sample(star.u - eps, SOD_LEFT, SOD_RIGHT, star, GAS)
    rho_r, u_r, p_r = sample(star.u + eps, SOD_LEFT, SOD_RIGHT, star, GAS)
    assert p_l == pytest.approx(p_r, rel=1e-9)
    assert u_l == pytest.approx(u_r, rel=1e-6)
    # the standard Sod plateau densities either side of the contact
    assert rho_l == pytest.approx(0.426319, rel=1e-5)
    assert rho_r == pytest.approx(0.265574, rel=1e-5)


def test_rankine_hugoniot_holds_across_the_sod_shock():
    star = solve_star(SOD_LEFT, SOD_RIGHT, GAS)
    g = GAS.gamma
    cr = SOD_RIGHT.sound_speed(GAS)
    S = SOD_RIGHT.u + cr * math.sqrt((g + 1) / (2 * g) * star.p / SOD_RIGHT.p + (g - 1) / (2 * g))
    eps = 1e-9
    rL, uL, pL = sample(S - eps, SOD_LEFT, SOD_RIGHT, star, GAS)
    rR, uR, pR = sample(S + eps, SOD_LEFT, SOD_RIGHT, star, GAS)

    def flux(rho, u, p):
        E = p / (g - 1) + 0.5 * rho * u * u
        return np.array([rho * u, rho * u * u + p, u * (E + p)])

    def cons(rho, u, p):
        return np.array([rho, rho * u, p / (g - 1) + 0.5 * rho * u * u])

    lhs = flux(rL, uL, pL) - flux(rR, uR, pR)
    rhs = S * (cons(rL, uL, pL) - cons(rR, uR, pR))
    assert np.allclose(lhs, rhs, rtol=1e-6, atol=1e-9)


def test_entropy_is_constant_through_the_rarefaction_fan():
    star = solve_star(SOD_LEFT, SOD_RIGHT, GAS)
    s_ref = SOD_LEFT.p / SOD_LEFT.rho**GAS.gamma
    for S in np.linspace(-1.15, -0.10, 25):
        rho, _, p = sample(S, SOD_LEFT, SOD_RIGHT, star, GAS)
        assert p / rho**GAS.gamma == pytest.approx(s_ref, rel=1e-10)


def test_entropy_jumps_across_the_shock_only_in_the_correct_direction():
    star = solve_star(SOD_LEFT, SOD_RIGHT, GAS)
    s_right = SOD_RIGHT.p / SOD_RIGHT.rho**GAS.gamma
    rho, _, p = sample(1.2, SOD_LEFT, SOD_RIGHT, star, GAS)  # between contact and shock
    assert p / rho**GAS.gamma > s_right  # entropy increases through a shock


def test_profile_is_self_similar():
    x = np.linspace(0.0, 1.0, 401)
    r1, u1, p1 = sample_profile(x, 0.1, SOD_LEFT, SOD_RIGHT, GAS)
    # doubling t and stretching x about the diaphragm must give the same state
    x2 = 0.5 + (x - 0.5) * 2.0
    r2, u2, p2 = sample_profile(x2, 0.2, SOD_LEFT, SOD_RIGHT, GAS)
    assert np.allclose(r1, r2, rtol=1e-12)
    assert np.allclose(u1, u2, rtol=1e-12, atol=1e-14)
    assert np.allclose(p1, p2, rtol=1e-12)


def test_profile_rejects_nonpositive_time():
    with pytest.raises(ValueError):
        sample_profile(np.linspace(0, 1, 5), 0.0, SOD_LEFT, SOD_RIGHT, GAS)


def test_solver_is_symmetric_under_mirroring():
    """Mirroring x reverses velocities and swaps the states."""
    left = RiemannState(1.0, 0.3, 2.0)
    right = RiemannState(0.4, -0.2, 0.5)
    mirror_l = RiemannState(right.rho, -right.u, right.p)
    mirror_r = RiemannState(left.rho, -left.u, left.p)

    a = solve_star(left, right, GAS)
    b = solve_star(mirror_l, mirror_r, GAS)
    assert a.p == pytest.approx(b.p, rel=1e-12)
    assert a.u == pytest.approx(-b.u, rel=1e-12)

    for S in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        r1, u1, p1 = sample(S, left, right, a, GAS)
        r2, u2, p2 = sample(-S, mirror_l, mirror_r, b, GAS)
        assert r1 == pytest.approx(r2, rel=1e-12)
        assert u1 == pytest.approx(-u2, rel=1e-12, abs=1e-14)
        assert p1 == pytest.approx(p2, rel=1e-12)


def test_works_for_other_gamma():
    gas = PerfectGas(gamma=1.667, cp=5193.0)
    star = solve_star(SOD_LEFT, SOD_RIGHT, gas)
    assert star.p > SOD_RIGHT.p
    assert star.p < SOD_LEFT.p
    assert star.u > 0.0
    # sampling stays monotone in pressure from left to right
    S = np.linspace(-3, 3, 200)
    p = [sample(s, SOD_LEFT, SOD_RIGHT, star, gas)[2] for s in S]
    assert all(b <= a + 1e-12 for a, b in zip(p, p[1:], strict=False))
