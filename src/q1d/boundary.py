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
        self, riemann: float, p0: float, T0: float, sign: float, gas: PerfectGas
    ) -> GhostState:
        """Solve the inflow quadratic for the boundary sound speed.

        ``riemann`` is the invariant running *out* of the domain:
        ``u - 2c/(gamma-1)`` at the left, ``u + 2c/(gamma-1)`` at the right.
        ``sign`` is +1 when the inflow runs in +x (left boundary), -1 at the
        right.
        """
        g, gm1, gp1 = _gamma_local(gas, T0)
        c0_sq = g * gas.R * T0
        disc = -0.5 * gm1 + gp1 * c0_sq / (gm1 * riemann * riemann)
        if disc < 0.0:
            # The imposed stagnation state cannot sustain the outgoing
            # invariant. Clamping hides this; fall back to the stagnation
            # state itself, which is the physically closest admissible state.
            cb = math.sqrt(c0_sq)
        else:
            cb = abs(riemann) * gm1 / gp1 * (1.0 + math.sqrt(disc))
        Tb = T0 * cb * cb / c0_sq
        Tb = min(Tb, T0)  # guard round-off above stagnation
        pb = p0 * gas.pressure_ratio_isentropic(T0, Tb)
        rhob = pb / (gas.R * Tb)
        ub = sign * math.sqrt(max(0.0, 2.0 * (gas.enthalpy(T0) - gas.enthalpy(Tb))))
        return GhostState(rhob, ub, pb)

    @staticmethod
    def _subsonic_outflow(
        rho: float, p: float, riemann: float, p_imposed: float, sign: float, gas: PerfectGas
    ) -> GhostState:
        """Impose static pressure, extrapolate entropy, close with the invariant.

        ``sign`` is +1 at the left boundary, where the outgoing invariant is
        ``J- = u - 2c/(gamma-1)`` and so ``u_b = J- + 2c_b/(gamma-1)``; and -1
        at the right, where ``J+ = u + 2c/(gamma-1)`` gives
        ``u_b = J+ - 2c_b/(gamma-1)``.
        """
        g, gm1, _ = _gamma_local(gas, p / (rho * gas.R))
        rhob = rho * (p_imposed / p) ** (1.0 / g)
        cb = math.sqrt(g * p_imposed / rhob)
        ub = riemann + sign * 2.0 * cb / gm1
        return GhostState(rhob, ub, p_imposed)

    def left(self, rho: float, u: float, p: float, c: float, gas: PerfectGas) -> GhostState:
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
            gm1 = _gamma_local(gas, self.T0_in)[1]
            return self._subsonic_inflow(u - 2.0 * c / gm1, self.p0_in, self.T0_in, +1.0, gas)
        if -u < c:  # subsonic outflow through the inlet (reverse flow)
            gm1 = _gamma_local(gas, p / (rho * gas.R))[1]
            return self._subsonic_outflow(rho, p, u - 2.0 * c / gm1, self.p_back, +1.0, gas)
        return GhostState(rho, u, p)  # supersonic outflow: extrapolate

    def right(self, rho: float, u: float, p: float, c: float, gas: PerfectGas) -> GhostState:
        if u < 0.0:  # inflow through the outlet (reverse flow)
            p0 = self.p0_back if self.p0_back is not None else self.p0_in
            T0 = self.T0_back if self.T0_back is not None else self.T0_in
            if -u >= c:
                return GhostState(rho, u, p)
            gm1 = _gamma_local(gas, T0)[1]
            return self._subsonic_inflow(u + 2.0 * c / gm1, p0, T0, -1.0, gas)
        if u < c:  # subsonic outflow: impose the back pressure
            gm1 = _gamma_local(gas, p / (rho * gas.R))[1]
            return self._subsonic_outflow(rho, p, u + 2.0 * c / gm1, self.p_back, -1.0, gas)
        return GhostState(rho, u, p)  # supersonic outflow: extrapolate


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
