"""Component performance maps in β-line form.

Reads the GasTurb-style workbooks supplied with the project: sheets
``beta_lines``, ``mass_flow``, ``pressure_ratio``, ``efficiency`` (and
``surge_line`` for compressors), with corrected speed in the column headers.

Two conversions happen at load time, both measured rather than assumed
(``PLAN.md`` §3.5, §3.6):

**Efficiency becomes corrected work.** ``η`` is singular where ``PR → 1`` with
work input — on ``TranssonicCompressor`` it changes sign through a pole — and
it is non-monotonic in β on nearly every speed line. Corrected work
``CW = Δh₀/θ`` is monotonic in β on *every* speed line of *both* supplied maps
and has 2–7× lower curvature. It also needs no design speed, unlike a
coefficient normalised by ``N²``.

**Exit corrected flow is computed and becomes the lookup coordinate.**
Inverting inlet corrected flow to β is ill-posed over most of the map: the
whole β range compresses into 3–10% of ``Wc`` above 72% speed, and is not
monotonic. ``ECMF = Wc·√τ/PR`` is monotonic everywhere and spans 72–163%.

**Densification happens in (β, Nc), before the ECMF conversion** — see
:meth:`BetaMap.densify` and ``PLAN.md`` D13.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gas import PerfectGas

__all__ = ["BetaMap", "MapPoint", "load_beta_map"]

#: Standard-day reference used by the corrected parameters in these maps.
T_REF = 288.15
P_REF = 101325.0


@dataclass(frozen=True)
class MapPoint:
    """A single evaluated operating point."""

    beta: float
    corrected_speed: float
    Wc: float
    PR: float
    corrected_work: float  # Δh₀/θ  [J/kg]
    ecmf: float
    efficiency: float  # derived, for reporting only


@dataclass
class BetaMap:
    """β-line map with corrected work and ECMF derived at load time.

    Grids are ``(n_beta, n_speed)``. ``corrected_work`` and ``ecmf`` are
    computed from the supplied ``PR``/``efficiency`` using ``gas``.
    """

    name: str
    beta: np.ndarray
    corrected_speed: np.ndarray
    Wc: np.ndarray
    PR: np.ndarray
    efficiency: np.ndarray
    corrected_work: np.ndarray
    ecmf: np.ndarray
    gas: PerfectGas

    # -- diagnostics --------------------------------------------------------

    def monotonic_in_beta(self, field: str) -> list[bool]:
        """Per speed line, is ``field`` monotonic along β?

        The property that decides whether a quantity can be used as a lookup
        coordinate at all.
        """
        a = getattr(self, field)
        out = []
        for j in range(a.shape[1]):
            d = np.diff(a[:, j])
            out.append(bool(np.all(d > 0) or np.all(d < 0)))
        return out

    def span_in_beta(self, field: str) -> np.ndarray:
        """Per speed line, ``max/min - 1`` along β — the conditioning measure."""
        a = getattr(self, field)
        return a.max(axis=0) / a.min(axis=0) - 1.0

    # -- refinement ---------------------------------------------------------

    def densify(self, factor: int = 9) -> BetaMap:
        """Monotone-cubic refinement of the ``(β, Nc)`` grid, ``factor``× each way.

        Refining **before** the ECMF conversion is what makes this worth doing.
        β is not a physical dimension, but it is the *correspondence* label:
        β = 0.5 on the 80% line and β = 0.5 on the 90% line are the same
        relative position along their lines, so crossing speeds at fixed β
        follows the map's own grid. Crossing at fixed ECMF cuts across it, and
        measures worse — 2.81%/6.60% against 2.84%/5.32% on the 22 supplied
        maps (compressors/turbines, leave-one-speed-line-out).

        With a dense table, the plain linear lookup in :meth:`_speed_line`
        reproduces direct PCHIP to 0.07%, so this buys cubic accuracy at linear
        cost on a regular grid:

        =============================  ============  ==========
        leave-one-interior-line-out      linear        this
        =============================  ============  ==========
        compressors                        2.84%        1.79%
        turbines                           5.32%        2.71%
        =============================  ============  ==========

        Only ``Wc``, ``PR`` and ``corrected_work`` are interpolated; ``ecmf``
        and ``efficiency`` are *re-derived* from the refined values, exactly as
        at load. Efficiency is never interpolated — it has a pole on
        ``TranssonicCompressor`` (``PLAN.md`` §3.6) and a cubic through a pole
        is meaningless.

        Refinement adds no information and does not help extrapolation: outside
        the tabulated speed range cubic is *worse* than linear (7.7% against
        4.9%), which is why :meth:`_speed_line` clamps rather than extrapolates.
        """
        from scipy.interpolate import PchipInterpolator

        if factor < 1:
            raise ValueError(f"densify factor must be >= 1, got {factor}")
        if factor == 1:
            return self

        def refine(axis: np.ndarray) -> np.ndarray:
            if len(axis) < 2:
                return axis
            return np.concatenate(
                [np.linspace(axis[i], axis[i + 1], factor + 1)[:-1] for i in range(len(axis) - 1)]
                + [axis[-1:]]
            )

        beta = refine(self.beta)
        speed = refine(self.corrected_speed)

        def stretch(a: np.ndarray) -> np.ndarray:
            # β first (within each supplied speed line), then Nc at fixed β.
            if len(self.beta) >= 3:
                a = PchipInterpolator(self.beta, a, axis=0)(beta)
            else:
                a = np.stack([np.interp(beta, self.beta, a[:, j]) for j in range(a.shape[1])], 1)
            if len(self.corrected_speed) >= 3:
                return PchipInterpolator(self.corrected_speed, a, axis=1)(speed)
            n = self.corrected_speed
            return np.stack([np.interp(speed, n, a[i]) for i in range(len(beta))])

        Wc = stretch(self.Wc)
        PR = stretch(self.PR)
        corrected_work = stretch(self.corrected_work)

        dh0s = self.gas.cp * T_REF * (PR**self.gas.gm1_over_g - 1.0)
        tau = 1.0 + corrected_work / (self.gas.cp * T_REF)
        if np.any(tau <= 0.0):
            raise ValueError(f"{self.name}: densification produced a non-physical τ")

        return BetaMap(
            name=f"{self.name}×{factor}",
            beta=beta,
            corrected_speed=speed,
            Wc=Wc,
            PR=PR,
            efficiency=dh0s / corrected_work,
            corrected_work=corrected_work,
            ecmf=Wc * np.sqrt(tau) / PR,
            gas=self.gas,
        )

    # -- evaluation ---------------------------------------------------------

    def _speed_line(self, corrected_speed: float) -> tuple[np.ndarray, ...]:
        """Interpolate the whole map onto one corrected speed.

        Linear in ``Nc``, clamped at the ends. Returns
        ``(ecmf, Wc, PR, corrected_work, efficiency)`` as 1-D arrays over β.
        """
        n = self.corrected_speed
        if corrected_speed <= n[0]:
            j, w = 0, 0.0
        elif corrected_speed >= n[-1]:
            j, w = len(n) - 2, 1.0
        else:
            j = int(np.searchsorted(n, corrected_speed) - 1)
            w = (corrected_speed - n[j]) / (n[j + 1] - n[j])

        def lerp(a):
            return a[:, j] * (1.0 - w) + a[:, j + 1] * w

        return (
            lerp(self.ecmf),
            lerp(self.Wc),
            lerp(self.PR),
            lerp(self.corrected_work),
            lerp(self.efficiency),
        )

    def evaluate_at_ecmf(self, ecmf: float, corrected_speed: float) -> MapPoint:
        """Look up by **exit** corrected flow — the well-conditioned coordinate.

        ``ecmf`` is monotonic in β on every speed line of both supplied maps, so
        the inversion is a bracketed search with a unique root. Outside the
        tabulated range the nearest end point is returned rather than
        extrapolated; extrapolating a β map is not meaningful.
        """
        e, wc, pr, cw, eff = self._speed_line(corrected_speed)
        beta = self._invert(e, ecmf)
        return MapPoint(
            beta=beta,
            corrected_speed=corrected_speed,
            Wc=float(np.interp(beta, self.beta, wc)),
            PR=float(np.interp(beta, self.beta, pr)),
            corrected_work=float(np.interp(beta, self.beta, cw)),
            ecmf=float(np.interp(beta, self.beta, e)),
            efficiency=float(np.interp(beta, self.beta, eff)),
        )

    def evaluate_at_beta(self, beta: float, corrected_speed: float) -> MapPoint:
        e, wc, pr, cw, eff = self._speed_line(corrected_speed)
        b = float(np.clip(beta, self.beta[0], self.beta[-1]))
        return MapPoint(
            beta=b,
            corrected_speed=corrected_speed,
            Wc=float(np.interp(b, self.beta, wc)),
            PR=float(np.interp(b, self.beta, pr)),
            corrected_work=float(np.interp(b, self.beta, cw)),
            ecmf=float(np.interp(b, self.beta, e)),
            efficiency=float(np.interp(b, self.beta, eff)),
        )

    def _invert(self, values: np.ndarray, target: float) -> float:
        """β such that ``values(β) == target``, assuming monotonicity."""
        ascending = values[-1] > values[0]
        v = values if ascending else values[::-1]
        b = self.beta if ascending else self.beta[::-1]
        if target <= v[0]:
            return float(b[0])
        if target >= v[-1]:
            return float(b[-1])
        return float(np.interp(target, v, b))

    # -- feasibility --------------------------------------------------------

    def minimum_inlet_area(self, gas: PerfectGas | None = None) -> float:
        """Smallest inlet area that can pass the map's highest corrected flow.

        Below this the inlet chokes before the map's high-flow end is reachable
        and the axial Mach number becomes infeasible — the failure mode to
        check *before* running, not to debug afterwards.
        """
        from .analytic import max_flow_function

        gas = gas or self.gas
        return float(self.Wc.max() * math.sqrt(gas.R * T_REF) / (max_flow_function(gas) * P_REF))


def load_beta_map(path: str | Path, gas: PerfectGas, name: str | None = None) -> BetaMap:
    """Read a workbook and derive corrected work and ECMF.

    Corrected work is evaluated at the reference condition, where ``θ = 1``:

    .. math::

        \\Delta h_0 = \\frac{h(T_{02s}) - h(T_{ref})}{\\eta},
        \\qquad s^\\circ(T_{02s}) = s^\\circ(T_{ref}) + R\\ln PR

    For a perfect gas this reduces to ``cp·T_ref·(PR^k − 1)/η``, and the
    resulting ``CW`` is independent of ``T₀₁`` by construction. The singular
    corner where ``η ≤ 0`` is *not* special-cased: ``Δh₀`` stays finite there
    because the numerator changes sign with the denominator.
    """
    import pandas as pd

    path = Path(path)
    name = name or path.stem

    def sheet(s):
        return pd.read_excel(path, sheet_name=s)

    beta = sheet("beta_lines").iloc[:, 1].to_numpy(dtype=float)
    mf = sheet("mass_flow")
    speed = np.array([float(c) for c in mf.columns[1:]])
    Wc = mf.iloc[:, 1:].to_numpy(dtype=float)
    PR = sheet("pressure_ratio").iloc[:, 1:].to_numpy(dtype=float)
    eff = sheet("efficiency").iloc[:, 1:].to_numpy(dtype=float)

    if not (Wc.shape == PR.shape == eff.shape == (len(beta), len(speed))):
        raise ValueError(
            f"{name}: inconsistent sheet shapes — Wc {Wc.shape}, PR {PR.shape}, "
            f"eta {eff.shape}, expected {(len(beta), len(speed))}"
        )
    if np.any(np.diff(beta) <= 0):
        raise ValueError(f"{name}: beta_lines must be strictly increasing")
    if np.any(np.diff(speed) <= 0):
        raise ValueError(f"{name}: speed lines must be strictly increasing")

    # Corrected work at theta = 1. Finite even where eta <= 0, because the
    # isentropic enthalpy rise changes sign together with eta.
    dh0s = gas.cp * T_REF * (PR**gas.gm1_over_g - 1.0)
    corrected_work = dh0s / eff

    tau = 1.0 + corrected_work / (gas.cp * T_REF)
    if np.any(tau <= 0.0):
        raise ValueError(f"{name}: non-physical temperature ratio derived from the map")
    ecmf = Wc * np.sqrt(tau) / PR

    return BetaMap(
        name=name,
        beta=beta,
        corrected_speed=speed,
        Wc=Wc,
        PR=PR,
        efficiency=eff,
        corrected_work=corrected_work,
        ecmf=ecmf,
        gas=gas,
    )
