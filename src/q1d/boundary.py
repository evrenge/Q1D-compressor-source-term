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
        c0_sq = gas.gamma * gas.R * T0
        disc = -0.5 * gas.gm1 + gas.gp1 * c0_sq / (gas.gm1 * riemann * riemann)
        if disc < 0.0:
            # The imposed stagnation state cannot sustain the outgoing
            # invariant. Clamping hides this; fall back to the stagnation
            # state itself, which is the physically closest admissible state.
            cb = math.sqrt(c0_sq)
        else:
            cb = abs(riemann) * gas.gm1 / gas.gp1 * (1.0 + math.sqrt(disc))
        Tb = T0 * cb * cb / c0_sq
        Tb = min(Tb, T0)  # guard round-off above stagnation
        pb = p0 * (Tb / T0) ** gas.g_over_gm1
        rhob = pb / (gas.R * Tb)
        ub = sign * math.sqrt(max(0.0, 2.0 * gas.cp * (T0 - Tb)))
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
        rhob = rho * (p_imposed / p) ** (1.0 / gas.gamma)
        cb = math.sqrt(gas.gamma * p_imposed / rhob)
        ub = riemann + sign * 2.0 * cb / gas.gm1
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
                Tb = self.T0_in * (pb / self.p0_in) ** gas.gm1_over_g
                ub = math.sqrt(max(0.0, 2.0 * gas.cp * (self.T0_in - Tb)))
                return GhostState(pb / (gas.R * Tb), ub, pb)
            return self._subsonic_inflow(u - 2.0 * c / gas.gm1, self.p0_in, self.T0_in, +1.0, gas)
        if -u < c:  # subsonic outflow through the inlet (reverse flow)
            return self._subsonic_outflow(rho, p, u - 2.0 * c / gas.gm1, self.p_back, +1.0, gas)
        return GhostState(rho, u, p)  # supersonic outflow: extrapolate

    def right(self, rho: float, u: float, p: float, c: float, gas: PerfectGas) -> GhostState:
        if u < 0.0:  # inflow through the outlet (reverse flow)
            p0 = self.p0_back if self.p0_back is not None else self.p0_in
            T0 = self.T0_back if self.T0_back is not None else self.T0_in
            if -u >= c:
                return GhostState(rho, u, p)
            return self._subsonic_inflow(u + 2.0 * c / gas.gm1, p0, T0, -1.0, gas)
        if u < c:  # subsonic outflow: impose the back pressure
            return self._subsonic_outflow(rho, p, u + 2.0 * c / gas.gm1, self.p_back, -1.0, gas)
        return GhostState(rho, u, p)  # supersonic outflow: extrapolate


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
        c = math.sqrt(gas.gamma * pi / rho)
        g: GhostState = getattr(bc, setter)(rho, u, pi, c, gas)

        area = grid.a_cell[index]
        cv[0, index] = g.rho * area
        cv[1, index] = g.rho * g.u * area
        cv[2, index] = (g.p / gas.gm1 + 0.5 * g.rho * g.u * g.u) * area
        p[index] = g.p
