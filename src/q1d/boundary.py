"""Boundary conditions.

Each condition writes the ghost-cell conservative state and pressure. The
characteristic conditions follow Blazek §8.4; the algebra was verified against
the standard derivation during review and is reproduced here unchanged, with
the supersonic branches that the legacy script omitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .gas import PerfectGas
from .grid import Grid

__all__ = ["BoundaryCondition", "GhostState", "StagnationInletStaticOutlet", "Transmissive"]


@dataclass(frozen=True)
class GhostState:
    """Primitive state to impose in a ghost cell."""

    rho: float
    u: float
    p: float


class BoundaryCondition(Protocol):
    def left(self, rho: float, u: float, p: float, c: float, gas: PerfectGas) -> GhostState: ...

    def right(self, rho: float, u: float, p: float, c: float, gas: PerfectGas) -> GhostState: ...


@dataclass(frozen=True)
class Transmissive:
    """Zero-gradient extrapolation at both ends.

    Non-reflecting only for genuinely outgoing waves; used for shock-tube tests
    run short enough that no wave reaches a boundary.
    """

    def left(self, rho: float, u: float, p: float, c: float, gas: PerfectGas) -> GhostState:
        return GhostState(rho, u, p)

    def right(self, rho: float, u: float, p: float, c: float, gas: PerfectGas) -> GhostState:
        return GhostState(rho, u, p)


@dataclass(frozen=True)
class StagnationInletStaticOutlet:
    """Stagnation state imposed at the inlet, static pressure at the outlet.

    ``p0_in``/``T0_in`` drive the left boundary; ``p_back`` is the static
    pressure imposed at the right. ``p0_back``/``T0_back`` are the stagnation
    state used if flow ever *enters* through the right boundary; they default to
    the inlet values.

    The legacy version reused ``p1`` as both the inlet stagnation pressure and
    the static back pressure for reverse flow at the left boundary, silently
    conflating the two. They are separate here.
    """

    p0_in: float
    T0_in: float
    p_back: float
    p0_back: float | None = None
    T0_back: float | None = None
    p_static_in: float | None = None

    def _subsonic_inflow(
        self, riemann: float, p0: float, T0: float, sign: float, gas: PerfectGas, gm1: float
    ) -> GhostState:
        """Solve the inflow condition for the boundary state.

        ``riemann`` is the invariant running *out* of the domain:
        ``u - 2c/(gamma-1)`` at the left, ``u + 2c/(gamma-1)`` at the right.
        ``sign`` is +1 when the inflow runs in +x (left boundary), -1 at the
        right. ``gm1`` is the ``gamma - 1`` **the caller used to form**
        ``riemann``; passing it in rather than re-deriving it is what makes the
        two halves of the condition the same equation (see below).

        Whatever ``T_b`` comes out, the ghost state is exactly on the isentrope
        through ``(p0, T0)`` and exactly satisfies ``h(T0) = h(T_b) + u_b^2/2``,
        so its *stagnation* state is the imposed one to round-off for either
        gas. ``T_b`` is the one free parameter, and the outgoing characteristic
        is what fixes it.
        """
        if hasattr(gas, "gamma"):
            g, gp1 = gas.gamma, gas.gp1
            c0_sq = g * gas.R * T0
            disc = -0.5 * gm1 + gp1 * c0_sq / (gm1 * riemann * riemann)
            if disc < 0.0:
                # The imposed stagnation state cannot sustain the outgoing
                # invariant. Clamping hides this; fall back to the stagnation
                # state itself, which is the physically closest admissible state.
                cb = math.sqrt(c0_sq)
            else:
                cb = abs(riemann) * gm1 / gp1 * (1.0 + math.sqrt(disc))
            Tb = min(T0 * cb * cb / c0_sq, T0)  # guard round-off above stagnation
        else:
            Tb = _real_inflow_temperature(riemann, T0, sign, gas, gm1)
        pb = p0 * gas.pressure_ratio_isentropic(T0, Tb)
        rhob = pb / (gas.R * Tb)
        ub = sign * math.sqrt(max(0.0, 2.0 * (gas.enthalpy(T0) - gas.enthalpy(Tb))))
        return GhostState(rhob, ub, pb)

    @staticmethod
    def _subsonic_outflow(
        rho: float, p: float, riemann: float, p_imposed: float, sign: float,
        gas: PerfectGas, gm1: float
    ) -> GhostState:
        """Impose static pressure, extrapolate entropy, close with the invariant.

        ``sign`` is +1 at the left boundary, where the outgoing invariant is
        ``J- = u - 2c/(gamma-1)`` and so ``u_b = J- + 2c_b/(gamma-1)``; and -1
        at the right, where ``J+ = u + 2c/(gamma-1)`` gives
        ``u_b = J+ - 2c_b/(gamma-1)``.

        ``gm1`` again comes from the caller, for the same reason.
        """
        T = p / (rho * gas.R)
        if hasattr(gas, "gamma"):
            rhob = rho * (p_imposed / p) ** (1.0 / gas.gamma)
            cb = math.sqrt(gas.gamma * p_imposed / rhob)
        else:
            # The constant-gamma `rho (p/p_ref)^(1/gamma)` is the power-law
            # isentrope; take the real one through the same interior state.
            Tb = gas.temperature_isentropic(T, p_imposed / p)
            rhob = p_imposed / (gas.R * Tb)
            cb = math.sqrt(gas.gamma_at(Tb) * gas.R * Tb)
        ub = riemann + sign * 2.0 * cb / gm1
        return GhostState(rhob, ub, p_imposed)

    def left(self, rho: float, u: float, p: float, c: float, gas: PerfectGas) -> GhostState:
        # gamma frozen at the state whose characteristic this is -- the interior
        # cell's. Not at T0: a uniform duct is then no longer a fixed point of
        # the condition, because the invariant would be formed with one gamma
        # and inverted with another. Measured on a NASA9 compressor, that
        # mismatch left the sampled inlet p01 1.4e-04 BELOW the imposed p0_in
        # and moved the operating point by 2.8e-04 in mass flow (PLAN.md §3.45).
        # Identical for a perfect gas, where the two gammas are the same number.
        gm1 = _gamma_local(gas, p / (rho * gas.R))[1]
        if u >= 0.0:  # inflow
            if u >= c:
                # Supersonic inflow: all three characteristics enter, so the
                # state is fully imposed and needs a third condition beyond
                # p0/T0. The legacy code had no supersonic branch at all and
                # fell through to the characteristic form regardless of Mach.
                if self.p_static_in is None:
                    raise ValueError(
                        "supersonic inflow at the left boundary needs a third condition; "
                        "set p_static_in alongside p0_in/T0_in"
                    )
                pb = self.p_static_in
                Tb = gas.temperature_isentropic(self.T0_in, pb / self.p0_in)
                ub = math.sqrt(max(0.0, 2.0 * (gas.enthalpy(self.T0_in) - gas.enthalpy(Tb))))
                cb = math.sqrt(_gamma_local(gas, Tb)[0] * gas.R * Tb)
                if ub < cb:
                    raise ValueError(
                        f"p_static_in={pb!r} with p0_in={self.p0_in!r} imposes M={ub / cb:.3f} "
                        "on a boundary classified as supersonic inflow; the imposed triple "
                        "must itself be supersonic"
                    )
                return GhostState(pb / (gas.R * Tb), ub, pb)
            return self._subsonic_inflow(
                u - 2.0 * c / gm1, self.p0_in, self.T0_in, +1.0, gas, gm1
            )
        if -u < c:  # subsonic outflow through the inlet (reverse flow)
            return self._subsonic_outflow(
                rho, p, u - 2.0 * c / gm1, self.p_back, +1.0, gas, gm1
            )
        return GhostState(rho, u, p)  # supersonic outflow: extrapolate

    def right(self, rho: float, u: float, p: float, c: float, gas: PerfectGas) -> GhostState:
        gm1 = _gamma_local(gas, p / (rho * gas.R))[1]
        if u < 0.0:  # inflow through the outlet (reverse flow)
            p0 = self.p0_back if self.p0_back is not None else self.p0_in
            T0 = self.T0_back if self.T0_back is not None else self.T0_in
            if -u >= c:
                return GhostState(rho, u, p)
            return self._subsonic_inflow(u + 2.0 * c / gm1, p0, T0, -1.0, gas, gm1)
        if u < c:  # subsonic outflow: impose the back pressure
            return self._subsonic_outflow(
                rho, p, u + 2.0 * c / gm1, self.p_back, -1.0, gas, gm1
            )
        return GhostState(rho, u, p)  # supersonic outflow: extrapolate


def _real_inflow_temperature(
    riemann: float, T0: float, sign: float, gas, gm1: float
) -> float:
    """``T_b`` on the isentrope through ``T0`` that carries the given invariant.

    The perfect-gas branch closes in one quadratic because ``T = T0 (c/c0)^2``
    and ``u = 2(c0 - c)/(gamma-1)`` are both explicit. Neither survives variable
    ``cp``, so solve the same statement directly:

    .. math::

        \\mathrm{sign}\\left(u(T_b) - \\frac{2c(T_b)}{\\gamma-1}\\right)
            = \\text{riemann},\\quad
        u(T_b) = \\sqrt{2(h(T_0) - h(T_b))},\\;
        c(T_b) = \\sqrt{\\gamma(T_b) R T_b}

    As ``T_b`` falls from ``T0`` the velocity rises from zero and the sound speed
    falls, so the left side is monotone and the root is unique on the subsonic
    bracket ``[T*, T0]``. Because ``u`` and ``c`` here are the *real* ones and
    ``gamma-1`` is the same frozen value the caller used to form ``riemann``, an
    interior state already at the imposed stagnation condition reproduces itself
    exactly — which is the property the perfect-gas branch has and the reason
    this exists.
    """
    from .analytic import _bracketed_root, _sonic_temperature

    h0 = gas.enthalpy(T0)

    def leaving(T: float) -> float:
        u = math.sqrt(max(0.0, 2.0 * (h0 - gas.enthalpy(T))))
        return sign * (u - 2.0 * math.sqrt(gas.gamma_at(T) * gas.R * T) / gm1) - riemann

    lo = _sonic_temperature(T0, gas)
    hi = T0 * (1.0 - 1e-15)
    f_lo, f_hi = leaving(lo), leaving(hi)
    if f_lo * f_hi > 0.0:
        # No subsonic state on this isentrope carries the invariant -- the
        # perfect-gas branch's negative discriminant, in the general form. Fall
        # back to the stagnation state, the closest admissible one.
        return T0
    return _bracketed_root(leaving, lo, hi, f_lo, f_hi, tol=1e-14 * T0)


def _gamma_local(gas, T: float) -> tuple[float, float, float]:
    """``(γ, γ−1, γ+1)`` at a temperature, for either gas.

    The characteristic boundary conditions integrate the Riemann invariant as
    ``u ± 2c/(γ−1)``, which is the *constant-γ* result. For a real gas the
    invariant is ``u ± ∫dp/(ρc)`` along an isentrope and has no closed form, so
    this freezes γ at the local temperature — standard practice, and accurate to
    the extent that γ varies little across the boundary cell. On air γ moves
    1.399 → 1.306 over 288–1600 K, so freezing it *locally* is a much smaller
    approximation than freezing it globally, which is what the perfect gas does.
    """
    g = gas.gamma if hasattr(gas, "gamma") else gas.gamma_at(T)
    return g, g - 1.0, g + 1.0


def apply_boundaries(
    cv: np.ndarray,
    p: np.ndarray,
    grid: Grid,
    gas: PerfectGas,
    bc: BoundaryCondition,
) -> None:
    """Write both ghost cells in place from the adjacent interior states."""
    for index, adjacent, setter in ((0, 1, "left"), (-1, -2, "right")):
        rho = cv[0, adjacent] / grid.a_cell[adjacent]
        u = cv[1, adjacent] / cv[0, adjacent]
        pi = p[adjacent]
        c = math.sqrt(_gamma_local(gas, pi / (rho * gas.R))[0] * pi / rho)
        g: GhostState = getattr(bc, setter)(rho, u, pi, c, gas)

        area = grid.a_cell[index]
        cv[0, index] = g.rho * area
        cv[1, index] = g.rho * g.u * area
        if hasattr(gas, "gm1"):
            cv[2, index] = (g.p / gas.gm1 + 0.5 * g.rho * g.u * g.u) * area
        else:
            Tg = g.p / (g.rho * gas.R)
            cv[2, index] = g.rho * (gas.enthalpy(Tg) - gas.R * Tg + 0.5 * g.u * g.u) * area
        p[index] = g.p
