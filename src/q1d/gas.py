"""Gas models.

Phase 1 provides only :class:`PerfectGas`. It is deliberately shaped as the
seed of the ``Gas`` interface that Phase 4 will extend to NASA9 polynomials
(see ``PLAN.md`` P3): callers take a gas object rather than reaching for
module-level ``gamma``/``cpgas`` globals, so swapping the thermodynamics later
is a constructor change rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PerfectGas", "AIR_LEGACY"]


@dataclass(frozen=True)
class PerfectGas:
    """Calorically perfect gas: constant ``gamma`` and constant ``cp``.

    ``R`` is *derived* from ``gamma`` and ``cp`` rather than supplied
    independently, which keeps the three mutually consistent. Note that this
    means a gas specified with ``cp = 1005`` and ``gamma = 1.4`` has
    ``R = 287.1429``, not the standard-air 287.05.
    """

    gamma: float
    cp: float

    @property
    def R(self) -> float:
        return (self.gamma - 1.0) * self.cp / self.gamma

    @property
    def cv(self) -> float:
        return self.cp - self.R

    # --- exponents that recur throughout the isentropic relations ---

    @property
    def gm1(self) -> float:
        """``gamma - 1``"""
        return self.gamma - 1.0

    @property
    def gp1(self) -> float:
        """``gamma + 1``"""
        return self.gamma + 1.0

    @property
    def g_over_gm1(self) -> float:
        """``gamma / (gamma - 1)`` — the p/p0 exponent."""
        return self.gamma / (self.gamma - 1.0)

    @property
    def gm1_over_g(self) -> float:
        """``(gamma - 1) / gamma`` — the T-ratio-from-p-ratio exponent."""
        return (self.gamma - 1.0) / self.gamma

    def enthalpy(self, T: float) -> float:
        """Specific enthalpy. Linear here; the NASA9 override will not be.

        Present in Phase 1 so that call sites are already written in terms of
        enthalpy rather than ``cp * dT`` (``PLAN.md`` §3.3).
        """
        return self.cp * T

    def temperature_from_enthalpy(self, h: float) -> float:
        return h / self.cp

    def speed_of_sound(self, T: float) -> float:
        return (self.gamma * self.R * T) ** 0.5


#: The gas the legacy scripts use: ``cp = 1005``, ``gamma = 1.4``, so
#: ``R = 287.1429``. Self-consistent, but not standard air — retained exactly
#: so Phase 1 results are comparable with ``legacy/``.
AIR_LEGACY = PerfectGas(gamma=1.4, cp=1005.0)
