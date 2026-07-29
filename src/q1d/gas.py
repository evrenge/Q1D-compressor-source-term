"""Gas models.

Phase 1 provides only :class:`PerfectGas`. It is deliberately shaped as the
seed of the ``Gas`` interface that Phase 4 will extend to NASA9 polynomials
(see ``PLAN.md`` P3): callers take a gas object rather than reaching for
module-level ``gamma``/``cpgas`` globals, so swapping the thermodynamics later
is a constructor change rather than a rewrite.
"""

from __future__ import annotations

import bisect
import functools
import math
from dataclasses import dataclass

import numpy as np

__all__ = ["PerfectGas", "Nasa9Gas", "AIR_LEGACY"]


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
        """Specific enthalpy. Linear here; the NASA9 override is not.

        Present in Phase 1 so that call sites are already written in terms of
        enthalpy rather than ``cp * dT`` (``PLAN.md`` §3.3).
        """
        return self.cp * T

    def temperature_from_enthalpy(self, h: float) -> float:
        return h / self.cp

    def speed_of_sound(self, T: float) -> float:
        return (self.gamma * self.R * T) ** 0.5

    # --- the temperature-aware interface ----------------------------------
    #
    # A calorically perfect gas ignores ``T`` in all of these, which is exactly
    # the point: a call site written against them works unchanged when handed a
    # gas whose ``cp`` really does move. The alternative -- call sites reading
    # ``gas.cp`` directly -- silently freezes the thermodynamics at whatever the
    # constructor was given, and on air that is a **21.7%** error in ``cp``
    # between 288 K and 1600 K (``PLAN.md`` §3.45).

    def cp_at(self, T: float) -> float:
        return self.cp

    def gamma_at(self, T: float) -> float:
        return self.gamma

    def entropy_ref(self, T: float) -> float:
        """``s°(T)``, the temperature part of the entropy, in J/kg/K.

        Defined so that an isentrope satisfies ``s°(T₂) − s°(T₁) = R·ln(p₂/p₁)``.
        For constant ``cp`` that is ``cp·ln T`` up to a constant, and the constant
        cancels in every use.
        """
        return self.cp * math.log(T)

    def pressure_ratio_isentropic(self, T1: float, T2: float) -> float:
        """``p₂/p₁`` along an isentrope between two temperatures.

        The power law ``(T₂/T₁)^(γ/(γ−1))`` is only valid at constant ``γ``. The
        general statement is the entropy one, and it is what a real gas needs;
        this override keeps the perfect-gas path exact rather than routing it
        through logarithms.
        """
        return (T2 / T1) ** self.g_over_gm1

    def temperature_isentropic(self, T1: float, pressure_ratio: float) -> float:
        """The inverse: ``T₂`` reached from ``T₁`` by an isentropic ``p₂/p₁``."""
        return T1 * pressure_ratio**self.gm1_over_g


#: The gas the legacy scripts use: ``cp = 1005``, ``gamma = 1.4``, so
#: ``R = 287.1429``. Self-consistent, but not standard air — retained exactly
#: so Phase 1 results are comparable with ``legacy/``.
AIR_LEGACY = PerfectGas(gamma=1.4, cp=1005.0)


@dataclass(frozen=True)
class Nasa9Gas:
    """A mixture whose ``cp``, ``h`` and ``s°`` follow NASA9 polynomials.

    The reason to bother: on air, ``cp`` rises **21.7%** between 288 K and
    1600 K. A compressor at standard-day inlet barely notices, which is why the
    perfect-gas results in this document stand. A turbine at turbine-entry
    conditions is squarely in the range where it matters, and §3.39 runs them at
    1600 K.

    Coefficients come from Cantera and are then evaluated here, so the solver
    never calls Cantera: a lookup per cell per step through a foreign library
    would dominate a step that currently costs 1512 µs (`PLAN.md` §3.41).
    :meth:`from_cantera` reads them once at construction.

    Per species and per temperature region NASA9 gives nine numbers
    ``a₁..a₇, b₁, b₂`` with

    .. math::

        \\frac{c_p}{R} &= a_1T^{-2} + a_2T^{-1} + a_3 + a_4T + a_5T^2
                          + a_6T^3 + a_7T^4 \\\\
        \\frac{h}{RT}  &= -a_1T^{-2} + a_2\\frac{\\ln T}{T} + a_3
                          + \\frac{a_4T}{2} + \\frac{a_5T^2}{3}
                          + \\frac{a_6T^3}{4} + \\frac{a_7T^4}{5} + \\frac{b_1}{T} \\\\
        \\frac{s°}{R}  &= -\\frac{a_1}{2T^2} - \\frac{a_2}{T} + a_3\\ln T + a_4T
                          + \\frac{a_5T^2}{2} + \\frac{a_6T^3}{3}
                          + \\frac{a_7T^4}{4} + b_2

    verified against Cantera to **1.6e−09** worst case over 200–5000 K, the one
    outlier sitting exactly on a region boundary and everything else at 1e−16.

    ``gamma`` and ``cp`` are deliberately **not** attributes. Reading either as a
    scalar is the mistake this class exists to prevent, and an ``AttributeError``
    at the call site is worth more than a number that is quietly 20% wrong.
    """

    #: ``(T_low, T_high, a₁..a₇, b₁, b₂)`` per region, per species.
    regions: tuple[tuple[float, float, tuple[float, ...], float, float], ...]
    #: Mass fraction weighting already folded in, one weight per species block.
    weights: tuple[float, ...]
    #: Specific gas constant of the mixture, J/kg/K.
    R: float
    name: str = "nasa9"

    @classmethod
    def from_cantera(
        cls,
        mechanism: str = "airNASA9.yaml",
        composition: dict[str, float] | None = None,
        name: str | None = None,
    ) -> Nasa9Gas:
        """Read NASA9 coefficients for a mole-fraction mixture.

        ``airNASA9.yaml`` ships with Cantera and carries genuine NASA9
        (``Nasa9PolyMultiTempRegion``) data over 200–20000 K. Cantera's
        ``air.yaml`` is NASA7 and will be refused here — the two formats differ
        in both the number of coefficients and the low-temperature term, so
        accepting one as the other would be wrong rather than approximate.

        The default composition is dry air's two majority species renormalised.
        ``airNASA9.yaml`` carries no argon, so the ~0.93% of it in real air is
        absent; that shifts ``M`` by about 0.4% and is a stated limitation
        rather than a hidden one.
        """
        import cantera as ct

        X = dict(composition or {"N2": 0.79, "O2": 0.21})
        total = sum(X.values())
        if total <= 0.0:
            raise ValueError(f"composition must have positive total, got {X}")
        X = {k: v / total for k, v in X.items()}

        sol = ct.Solution(mechanism)
        RU = float(ct.gas_constant)  # J/kmol/K
        regions: list = []
        weights: list[float] = []
        M_mix = sum(X[s] * float(sol.molecular_weights[sol.species_index(s)]) for s in X)

        for sp, x in X.items():
            th = sol.species(sp).thermo
            if type(th).__name__ != "Nasa9PolyMultiTempRegion":
                raise ValueError(
                    f"{sp} in {mechanism} is {type(th).__name__}, not NASA9. "
                    f"Cantera's air.yaml is NASA7; use airNASA9.yaml"
                )
            c = [float(v) for v in th.coeffs]
            n = int(c[0])
            blocks = []
            for r in range(n):
                b = c[1 + r * 11 : 1 + (r + 1) * 11]
                blocks.append((b[0], b[1], tuple(b[2:9]), b[9], b[10]))
            regions.append(tuple(blocks))
            # Mole fraction times R_universal, divided by mixture mass -- so the
            # per-species molar quantities sum straight into mixture-mass ones.
            weights.append(x * RU / M_mix)

        return cls(
            regions=tuple(regions),
            weights=tuple(weights),
            R=RU / M_mix,
            name=name or f"{mechanism}:{'/'.join(f'{k}={v:.4g}' for k, v in X.items())}",
        )

    @functools.cached_property
    def _tables(self):
        """``(upper edges, coefficient rows)`` per species, built once.

        ``_coeffs`` used to rebuild both arrays from Python lists on every call,
        and it is called several times per Newton iteration per bisection step.
        Profiling a NASA9 compressor run put 933,165 calls in 100 time steps,
        with the array construction — not the polynomial — as the cost
        (``PLAN.md`` §3.45). Cached on a frozen dataclass, which works because
        ``cached_property`` writes straight into ``__dict__`` rather than
        through ``__setattr__``.
        """
        out = []
        for blocks in self.regions:
            edges = np.array([b[1] for b in blocks[:-1]])
            table = np.array([[*b[2], b[3], b[4]] for b in blocks])
            out.append((edges, table))
        return tuple(out)

    @functools.cached_property
    def _scalar_tables(self):
        """The same rows as :attr:`_tables`, as plain Python floats.

        For the scalar path. Numpy's per-call overhead is what dominates a
        single-temperature evaluation: routed through the vectorised form, one
        ``enthalpy(T)`` on a float cost **38 µs**, essentially all of it numpy
        dispatch on 18 numbers of arithmetic. The root-finds in ``analytic.py``
        are all scalar and made 138,300 such calls per 100 solver steps, which
        is how a NASA9 run came to be 83x slower than the perfect-gas one
        (``PLAN.md`` §3.45).
        """
        return tuple(
            ([b[1] for b in blocks[:-1]], [(*b[2], b[3], b[4]) for b in blocks])
            for blocks in self.regions
        )

    def _coeffs(self, T):
        """Per-species coefficient rows selected for ``T``, scalar or array.

        Vectorised because the solver evaluates these on every face of every
        cell of every step. A scalar Newton per face would make the real gas
        unusable rather than merely slower: the perfect-gas step already costs
        1512 µs at 201 cells (``PLAN.md`` §3.41).
        """
        Ta = np.asarray(T, dtype=float)
        for edges, table in self._tables:
            yield table[np.searchsorted(edges, Ta, side="left")]

    def _scalar_rows(self, t: float):
        """``(row, weight)`` per species at one temperature, no numpy."""
        for (edges, rows), w in zip(self._scalar_tables, self.weights, strict=True):
            yield rows[bisect.bisect_left(edges, t)], w

    def cp_at(self, T):
        if np.ndim(T) == 0:
            t = float(T)
            out = 0.0
            for a, w in self._scalar_rows(t):
                out += w * (
                    a[0] / t**2 + a[1] / t + a[2] + a[3] * t
                    + a[4] * t**2 + a[5] * t**3 + a[6] * t**4
                )
            return out
        Ta = np.asarray(T, dtype=float)
        out = np.zeros_like(Ta)
        for c, w in zip(self._coeffs(Ta), self.weights, strict=True):
            a = [c[..., i] for i in range(7)]
            out = out + w * (
                a[0] / Ta**2 + a[1] / Ta + a[2] + a[3] * Ta
                + a[4] * Ta**2 + a[5] * Ta**3 + a[6] * Ta**4
            )
        return out

    def enthalpy(self, T):
        if np.ndim(T) == 0:
            t = float(T)
            lnt = math.log(t)
            out = 0.0
            for a, w in self._scalar_rows(t):
                out += w * t * (
                    -a[0] / t**2 + a[1] * lnt / t + a[2] + a[3] * t / 2.0
                    + a[4] * t**2 / 3.0 + a[5] * t**3 / 4.0 + a[6] * t**4 / 5.0 + a[7] / t
                )
            return out
        Ta = np.asarray(T, dtype=float)
        lnT = np.log(Ta)
        out = np.zeros_like(Ta)
        for c, w in zip(self._coeffs(Ta), self.weights, strict=True):
            a = [c[..., i] for i in range(7)]
            b1 = c[..., 7]
            out = out + w * Ta * (
                -a[0] / Ta**2 + a[1] * lnT / Ta + a[2] + a[3] * Ta / 2.0
                + a[4] * Ta**2 / 3.0 + a[5] * Ta**3 / 4.0 + a[6] * Ta**4 / 5.0 + b1 / Ta
            )
        return out

    def entropy_ref(self, T):
        if np.ndim(T) == 0:
            t = float(T)
            lnt = math.log(t)
            out = 0.0
            for a, w in self._scalar_rows(t):
                out += w * (
                    -a[0] / (2.0 * t**2) - a[1] / t + a[2] * lnt + a[3] * t
                    + a[4] * t**2 / 2.0 + a[5] * t**3 / 3.0 + a[6] * t**4 / 4.0 + a[8]
                )
            return out
        Ta = np.asarray(T, dtype=float)
        lnT = np.log(Ta)
        out = np.zeros_like(Ta)
        for c, w in zip(self._coeffs(Ta), self.weights, strict=True):
            a = [c[..., i] for i in range(7)]
            b2 = c[..., 8]
            out = out + w * (
                -a[0] / (2.0 * Ta**2) - a[1] / Ta + a[2] * lnT + a[3] * Ta
                + a[4] * Ta**2 / 2.0 + a[5] * Ta**3 / 3.0 + a[6] * Ta**4 / 4.0 + b2
            )
        return out

    def gamma_at(self, T):
        cp = self.cp_at(T)
        return cp / (cp - self.R)

    def speed_of_sound(self, T):
        if np.ndim(T) == 0:
            return math.sqrt(self.gamma_at(T) * self.R * float(T))
        return np.sqrt(self.gamma_at(T) * self.R * np.asarray(T, dtype=float))

    def temperature_from_enthalpy_array(self, h, guess=None):
        """Vectorised Newton on ``h(T)``, for the solver's per-cell inversion."""
        ha = np.asarray(h, dtype=float)
        T = np.full_like(ha, 300.0) if guess is None else np.array(guess, dtype=float)
        for _ in range(80):
            step = (self.enthalpy(T) - ha) / self.cp_at(T)
            T = np.maximum(T - step, 1.0)
            if np.all(np.abs(step) < 1e-11 * np.maximum(T, 1.0)):
                break
        return T

    def temperature_from_enthalpy(self, h: float, guess: float = 300.0) -> float:
        """Invert ``h(T)``. Newton on a function whose derivative is ``cp``.

        ``h`` is strictly increasing in ``T`` because ``cp > 0`` everywhere in
        the tabulated range, so this converges from any physical start.
        """
        T = guess
        for _ in range(60):
            f = self.enthalpy(T) - h
            cp = self.cp_at(T)
            step = f / cp
            T -= step
            if T <= 0.0:
                T = 1.0
            if abs(step) < 1e-12 * max(T, 1.0):
                return T
        raise ValueError(f"temperature_from_enthalpy did not converge for h={h!r}")

    def pressure_ratio_isentropic(self, T1: float, T2: float) -> float:
        """``p₂/p₁ = exp((s°(T₂) − s°(T₁))/R)``.

        The general statement, valid whether or not ``cp`` moves. The
        perfect-gas power law is the special case where ``s°`` is ``cp·ln T``.
        """
        return math.exp((self.entropy_ref(T2) - self.entropy_ref(T1)) / self.R)

    def temperature_isentropic(self, T1: float, pressure_ratio: float) -> float:
        """The inverse, by Newton on ``s°`` whose derivative is ``cp/T``."""
        if pressure_ratio <= 0.0:
            raise ValueError(f"pressure ratio must be positive, got {pressure_ratio!r}")
        target = self.entropy_ref(T1) + self.R * math.log(pressure_ratio)
        T = T1 * pressure_ratio ** (self.R / self.cp_at(T1))  # perfect-gas start
        for _ in range(60):
            f = self.entropy_ref(T) - target
            step = f * T / self.cp_at(T)
            T -= step
            if T <= 0.0:
                T = 1.0
            if abs(step) < 1e-12 * max(T, 1.0):
                return T
        raise ValueError(f"temperature_isentropic did not converge for PR={pressure_ratio!r}")

    # `gamma` and `cp` are absent on purpose -- see the class docstring.
    def __getattr__(self, item: str):
        if item in ("gamma", "cp", "cv", "gm1", "gp1", "g_over_gm1", "gm1_over_g"):
            raise AttributeError(
                f"{type(self).__name__} has no constant '{item}': it varies with "
                f"temperature (cp moves 21.7% between 288 K and 1600 K on air). "
                f"Use cp_at(T), gamma_at(T), pressure_ratio_isentropic(T1, T2) or "
                f"temperature_isentropic(T1, PR)"
            )
        raise AttributeError(item)
