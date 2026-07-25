"""Exact Riemann solver for the 1D Euler equations.

A verification reference, not a solver component — nothing in the time-marching
code imports this. It exists so the Phase 2 shock-tube gate compares against an
exact solution rather than against a previous run of the same code.

Follows Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
chapter 4: solve the pressure function for ``p_star`` by Newton iteration, then
sample the self-similar solution at ``S = x/t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gas import PerfectGas

__all__ = [
    "RiemannState",
    "StarState",
    "solve_star",
    "sample",
    "sample_profile",
    "SOD_LEFT",
    "SOD_RIGHT",
]


@dataclass(frozen=True)
class RiemannState:
    """Primitive state on one side of the discontinuity."""

    rho: float
    u: float
    p: float

    def sound_speed(self, gas: PerfectGas) -> float:
        return math.sqrt(gas.gamma * self.p / self.rho)


@dataclass(frozen=True)
class StarState:
    """Pressure and velocity in the star region between the two waves."""

    p: float
    u: float
    iterations: int


#: Sod's shock tube, the standard test (Toro §4.3.3, test 1).
SOD_LEFT = RiemannState(rho=1.0, u=0.0, p=1.0)
SOD_RIGHT = RiemannState(rho=0.125, u=0.0, p=0.1)


def _wave_function(p: float, side: RiemannState, gas: PerfectGas) -> tuple[float, float]:
    """Pressure function contribution from one side, and its derivative.

    Shock branch (``p > p_K``) and rarefaction branch (``p <= p_K``) per
    Toro eq. (4.6)-(4.7).
    """
    g = gas.gamma
    c = side.sound_speed(gas)

    if p > side.p:  # shock
        A = 2.0 / ((g + 1.0) * side.rho)
        B = (g - 1.0) / (g + 1.0) * side.p
        sqrt_term = math.sqrt(A / (B + p))
        f = (p - side.p) * sqrt_term
        df = sqrt_term * (1.0 - 0.5 * (p - side.p) / (B + p))
    else:  # rarefaction
        ratio = p / side.p
        f = 2.0 * c / (g - 1.0) * (ratio ** ((g - 1.0) / (2.0 * g)) - 1.0)
        df = ratio ** (-(g + 1.0) / (2.0 * g)) / (side.rho * c)
    return f, df


def _initial_pressure_guess(left: RiemannState, right: RiemannState, gas: PerfectGas) -> float:
    """Two-rarefaction approximation — positive and robust for all admissible data."""
    g = gas.gamma
    cl, cr = left.sound_speed(gas), right.sound_speed(gas)
    z = (g - 1.0) / (2.0 * g)
    num = cl + cr - 0.5 * (g - 1.0) * (right.u - left.u)
    den = cl / left.p**z + cr / right.p**z
    return max(1e-12, (num / den) ** (1.0 / z))


def solve_star(
    left: RiemannState,
    right: RiemannState,
    gas: PerfectGas,
    tol: float = 1e-14,
    maxiter: int = 100,
) -> StarState:
    """Solve for the star-region pressure and velocity.

    Raises ``ValueError`` if the initial data produce vacuum (the pressure
    positivity condition of Toro eq. 4.40 is violated).
    """
    g = gas.gamma
    cl, cr = left.sound_speed(gas), right.sound_speed(gas)
    du = right.u - left.u
    if 2.0 * (cl + cr) / (g - 1.0) <= du:
        raise ValueError("initial data generate vacuum; no star region exists")

    p = _initial_pressure_guess(left, right, gas)
    iterations = 0
    for _ in range(maxiter):
        iterations += 1
        fl, dfl = _wave_function(p, left, gas)
        fr, dfr = _wave_function(p, right, gas)
        p_new = p - (fl + fr + du) / (dfl + dfr)
        if p_new < 0.0:
            p_new = tol
        if abs(p_new - p) <= tol * max(1.0, abs(p_new)):
            p = p_new
            break
        p = p_new
    else:  # pragma: no cover - Newton on this function is reliably convergent
        raise RuntimeError("star pressure iteration did not converge")

    fl, _ = _wave_function(p, left, gas)
    fr, _ = _wave_function(p, right, gas)
    u = 0.5 * (left.u + right.u + fr - fl)
    return StarState(p=p, u=u, iterations=iterations)


def _sample_side(
    S: float,
    side: RiemannState,
    star: StarState,
    gas: PerfectGas,
    sign: float,
) -> tuple[float, float, float]:
    """Sample within the left (``sign = -1``) or right (``sign = +1``) wave family."""
    g = gas.gamma
    c = side.sound_speed(gas)
    # `sign` is -1 for the left family (waves run toward -x relative to the
    # fluid) and +1 for the right. A point lies *outside* the wave — still in
    # the undisturbed initial state — when (S - wave_speed) * sign > 0.
    if star.p > side.p:  # shock
        speed = side.u + sign * c * math.sqrt(
            (g + 1.0) / (2.0 * g) * star.p / side.p + (g - 1.0) / (2.0 * g)
        )
        if (S - speed) * sign > 0.0:
            return side.rho, side.u, side.p
        ratio = star.p / side.p
        rho = side.rho * (ratio + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * ratio + 1.0)
        return rho, star.u, star.p

    # rarefaction
    head = side.u + sign * c
    c_star = c * (star.p / side.p) ** ((g - 1.0) / (2.0 * g))
    tail = star.u + sign * c_star
    if (S - head) * sign > 0.0:
        return side.rho, side.u, side.p
    if (S - tail) * sign < 0.0:
        rho = side.rho * (star.p / side.p) ** (1.0 / g)
        return rho, star.u, star.p
    # inside the fan
    u = 2.0 / (g + 1.0) * (-sign * c + 0.5 * (g - 1.0) * side.u + S)
    c_fan = 2.0 / (g + 1.0) * (c - sign * 0.5 * (g - 1.0) * (side.u - S))
    rho = side.rho * (c_fan / c) ** (2.0 / (g - 1.0))
    p = side.p * (c_fan / c) ** (2.0 * g / (g - 1.0))
    return rho, u, p


def sample(
    S: float,
    left: RiemannState,
    right: RiemannState,
    star: StarState,
    gas: PerfectGas,
) -> tuple[float, float, float]:
    """Return ``(rho, u, p)`` at self-similar coordinate ``S = x/t``."""
    if S <= star.u:
        return _sample_side(S, left, star, gas, sign=-1.0)
    return _sample_side(S, right, star, gas, sign=+1.0)


def sample_profile(
    x: np.ndarray,
    t: float,
    left: RiemannState,
    right: RiemannState,
    gas: PerfectGas,
    x0: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact profile at time ``t`` for a diaphragm initially at ``x0``.

    Returns ``(rho, u, p)`` arrays matching ``x``.
    """
    if t <= 0.0:
        raise ValueError(f"time must be positive, got {t!r}")
    star = solve_star(left, right, gas)
    out = np.empty((3, len(x)))
    for i, xi in enumerate(np.asarray(x)):
        out[:, i] = sample((xi - x0) / t, left, right, star, gas)
    return out[0], out[1], out[2]
