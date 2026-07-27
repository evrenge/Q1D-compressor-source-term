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
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .gas import PerfectGas

__all__ = ["BetaMap", "ECMFMap", "MapPoint", "ScaledMap", "load_beta_map"]

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
    #: ``"compressor"`` or ``"turbine"``. Sets the sign of ``corrected_work``
    #: and which isentropic relation produced it, so it is not cosmetic.
    kind: str = "compressor"
    #: ``(β, speed)`` indices whose ``η`` was non-positive in the workbook and
    #: was interpolated from neighbours along β. Empty on every compressor map
    #: supplied; non-empty on four of the six turbines.
    repaired_efficiency: tuple[tuple[int, int], ...] = ()

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

        tau = 1.0 + corrected_work / (self.gas.cp * T_REF)
        if np.any(tau <= 0.0):
            raise ValueError(f"{self.name}: densification produced a non-physical τ")

        # Re-derive with the machine's own convention, and carry `kind` through.
        # Both matter: a densified turbine that inherited the compressor forms
        # would silently become a compressor again one call after the loader
        # took care to decide otherwise.
        k = self.gas.gm1_over_g
        if self.kind == "turbine":
            efficiency = -corrected_work / (self.gas.cp * T_REF * (1.0 - PR**-k))
            ecmf = Wc * np.sqrt(tau) * PR
        else:
            efficiency = self.gas.cp * T_REF * (PR**k - 1.0) / corrected_work
            ecmf = Wc * np.sqrt(tau) / PR

        return BetaMap(
            name=f"{self.name}×{factor}",
            beta=beta,
            corrected_speed=speed,
            Wc=Wc,
            PR=PR,
            efficiency=efficiency,
            corrected_work=corrected_work,
            ecmf=ecmf,
            gas=self.gas,
            kind=self.kind,
            repaired_efficiency=self.repaired_efficiency,
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

    def evaluate_at_Wc(self, Wc: float, corrected_speed: float) -> MapPoint:
        """Look up by **inlet** corrected flow — the coordinate a solver can close on.

        ECMF is the better coordinate for a cycle code, where the exit state is
        known independently. Inside the Q1D solver it is not: the exit state is
        produced by this very source, so keying on it closes an algebraic loop
        through the source's own output. That loop's gain is
        ``-dlnPR/dlnECMF``, measured at 0.90 at design speed on
        ``SubsonicCompressor`` and **1.02–1.10** on ``TranssonicCompressor`` —
        above one, where no amount of under-relaxation converges, because
        under-relaxing a positive-gain loop gives ``|1 + r(g-1)| > 1`` for every
        ``r > 0``.

        Inlet ``Wc`` is measured upstream of the disk, where the field is clean
        to ~1e-8, and depends on the source only through the solver's own
        dynamics — which is the physical feedback, and is restoring: more flow →
        lower PR → lower exit static pressure against a fixed back pressure →
        the flow decelerates.

        The price is conditioning. ``Wc`` spans 133% of its range at 33% speed
        but only 8% at 120%, so ``dlnPR/dlnWc`` runs from −0.002 to −8.7: the
        closure amplifies a mass-flow error into a pressure-ratio error by up to
        ~9×. That is the compressor's real stiffness, not a modelling artefact.

        Raises if ``Wc`` is not monotonic on the requested speed line — on
        ``TranssonicCompressor`` above 88% speed it is not, the line being
        vertical to within the tabulation, and no inlet-only closure can pick a
        point there. That region needs the exit state and is handled separately.
        """
        e, wc, pr, cw, eff = self._speed_line(corrected_speed)
        d = np.diff(wc)
        if not (np.all(d > 0.0) or np.all(d < 0.0)):
            span = wc.max() / wc.min() - 1.0
            raise ValueError(
                f"{self.name}: inlet Wc is not monotonic in beta at Nc={corrected_speed:.4g} "
                f"(span {100 * span:.2f}%), so it cannot be inverted. The speed line is "
                f"vertical to within the tabulation — the compressor is choked there. "
                f"Use FlowMatchedCompressor, which relaxes beta on the residual and needs "
                f"no inverse (PLAN.md §3.28)"
            )
        beta = self._invert(wc, Wc)
        return MapPoint(
            beta=beta,
            corrected_speed=corrected_speed,
            Wc=float(np.interp(beta, self.beta, wc)),
            PR=float(np.interp(beta, self.beta, pr)),
            corrected_work=float(np.interp(beta, self.beta, cw)),
            ecmf=float(np.interp(beta, self.beta, e)),
            efficiency=float(np.interp(beta, self.beta, eff)),
        )

    def inlet_closure_is_invertible(self, corrected_speed: float) -> bool:
        """Can :meth:`evaluate_at_Wc` be used at this speed?"""
        d = np.diff(self._speed_line(corrected_speed)[1])
        return bool(np.all(d > 0.0) or np.all(d < 0.0))

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


@dataclass(frozen=True)
class ScaledMap:
    """The same map shape, sized for a stage with a different inlet flow.

    A multistage machine cannot use one map for every stage. With a fixed ``W``,
    stage *k* sees ``Wc = W√θ_k/δ_k``; across a stage of ``PR ≈ 2``, ``δ``
    doubles while ``√θ`` rises about 12%, so ``Wc`` roughly halves per stage and
    leaves the tabulated range after two of them. That is not a modelling
    artefact — it is why the stages of a real compressor are different machines,
    each sized to its own inlet corrected flow.

    The standard device is **stage stacking**: give every stage the same map
    shape scaled to its own inlet, which also puts every stage at the same
    relative position on its speed line, i.e. a repeating-stage machine.

    ``scale`` is this stage's design corrected flow divided by the reference
    map's, so stage 1 has ``scale = 1``. Lookups divide by it and the returned
    flow quantities are multiplied back, so ``MapPoint.Wc`` and ``.ecmf`` stay
    in the stage's own units while ``PR``, ``corrected_work``, ``efficiency``
    and ``beta`` — all dimensionless and all invariant to the sizing — pass
    through untouched.
    """

    inner: BetaMap
    scale: float

    def __post_init__(self) -> None:
        if not self.scale > 0.0:
            raise ValueError(f"scale must be positive, got {self.scale!r}")

    @property
    def name(self) -> str:
        return f"{self.inner.name}×{self.scale:.4g}"

    @property
    def beta(self) -> np.ndarray:
        return self.inner.beta

    @property
    def corrected_speed(self) -> np.ndarray:
        return self.inner.corrected_speed

    def _rescale(self, p: MapPoint) -> MapPoint:
        return replace(p, Wc=p.Wc * self.scale, ecmf=p.ecmf * self.scale)

    def _speed_line(self, corrected_speed: float) -> tuple[np.ndarray, ...]:
        e, wc, pr, cw, eff = self.inner._speed_line(corrected_speed)
        return (e * self.scale, wc * self.scale, pr, cw, eff)

    def evaluate_at_Wc(self, Wc: float, corrected_speed: float) -> MapPoint:
        return self._rescale(self.inner.evaluate_at_Wc(Wc / self.scale, corrected_speed))

    def evaluate_at_ecmf(self, ecmf: float, corrected_speed: float) -> MapPoint:
        return self._rescale(self.inner.evaluate_at_ecmf(ecmf / self.scale, corrected_speed))

    def evaluate_at_beta(self, beta: float, corrected_speed: float) -> MapPoint:
        return self._rescale(self.inner.evaluate_at_beta(beta, corrected_speed))

    def inlet_closure_is_invertible(self, corrected_speed: float) -> bool:
        return self.inner.inlet_closure_is_invertible(corrected_speed)


@dataclass(frozen=True)
class ECMFMap:
    """A map keyed on **ECMF**, with β eliminated at build time.

    β exists only because a speed line is multivalued in inlet ``Wc``. ECMF is
    monotonic in β on **135 of 135** speed lines across all twelve supplied
    compressor and fan maps, so `β ↔ ECMF` is a bijection along every line and
    ``PR(ECMF, Nc)`` and ``Δh₀(ECMF, Nc)`` are single-valued. β therefore carries
    no information the key does not, and belongs in the loader rather than in the
    runtime.

    **The order matters and is not negotiable.** Densify in ``(β, Nc)`` *first*,
    convert *after* (D13). β is the correspondence label — β = 0.5 on the 80% and
    90% lines are the same relative position — so crossing speeds at fixed β
    follows the map's own grid, at 1.79% against 2.81% for crossing at fixed
    ECMF. Converting before densifying would pay that 1.6×; converting after
    costs nothing, because the runtime blend then happens between speed lines
    already 9× closer together and the error goes with the square of the gap.

    **What it changes, measured.** *On a tabulated speed line* this is an exact
    refactor: ``PR`` and corrected work agree with
    :meth:`BetaMap.evaluate_at_ecmf` to 2.2e−16, because inverting a piecewise
    linear ECMF onto β and then interpolating a piecewise linear ``PR`` in β is
    the same map as interpolating ``PR`` against ECMF directly, on shared nodes.

    *Between* speed lines it is not, and cannot be. β is the correspondence
    label, so :class:`BetaMap` blends the whole line at fixed β and then inverts;
    without β the only option is to evaluate each bracketing line at the ECMF
    asked for and blend the results — D13's "inverting ECMF first", which it
    measures at 2.81% against 1.79% *at native speed spacing*. Densification is
    what buys that back, and the residual is small rather than absent:

    ===================  ==========  ==========  ==========  ==========
    worst, mid-interval  Sub, ×9     Sub, ×36    Trans, ×9   Trans, ×36
    ===================  ==========  ==========  ==========  ==========
    ``PR``               4.42e−05    2.80e−06    1.20e−04    7.53e−06
    ``CW``               2.60e−04    1.84e−05    2.13e−03    1.50e−04
    ===================  ==========  ==========  ==========  ==========

    — about 15× for a 4× refinement, so second order, with the worst points at
    the *lowest* speeds where the supplied lines are furthest apart. Densify
    first, convert after, and densify enough; converting a native-spacing map
    would pay the full 2.81%.

    The separate, real residual is the map's β **resolution**. Measured end to
    end on ``HighPqPCompr`` Nc 0.700 f 0.85, converged mass flow is off by
    5.54e−06, 1.57e−06 and 2.69e−07 at ``densify`` 9, 18 and 36 — observed order
    2.18, and on ``SubsonicCompressor`` the error passes through zero, so it is a
    convergent discretisation of the map rather than a bias. Densify accordingly;
    re-keying does not help it and was never going to.

    Turbines are excluded on purpose. ECMF is monotonic on only 7 of 22 turbine
    speed lines, while ``PR`` is monotonic on **22 of 22** — a turbine wants its
    own key, and :meth:`BetaMap` still serves it.
    """

    name: str
    corrected_speed: np.ndarray  # (n_speed,)
    key: np.ndarray  # (n_key, n_speed), ascending down each column
    PR: np.ndarray
    corrected_work: np.ndarray
    efficiency: np.ndarray
    gas: PerfectGas
    #: Which quantity :attr:`key` holds — ``"ecmf"`` for a compressor, ``"PR"``
    #: for a turbine. ECMF is monotonic in β on 135 of 135 compressor lines but
    #: only 17 of 64 turbine lines, where ``PR`` manages **64 of 64**, so the
    #: two machines want different keys and neither choice generalises.
    key_field: str = "ecmf"
    #: Corrected flow, tabulated rather than derived when ``PR`` is the key.
    #: ``None`` on an ECMF-keyed map, where ``Wc`` follows from the key exactly.
    Wc: np.ndarray | None = None
    #: Count of lookups whose key fell outside the tabulated range, in a
    #: one-element array so a frozen dataclass can still keep the tally.
    off_table: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))

    @property
    def ecmf(self) -> np.ndarray:
        """The key array, when the key *is* ECMF.

        Raises on a PR-keyed map rather than returning ``PR`` under the name
        ``ecmf``: a wrong label on a table that still interpolates cleanly is
        exactly the failure mode that cost §3.37 an afternoon.
        """
        if self.key_field != "ecmf":
            raise AttributeError(
                f"{self.name} is keyed on {self.key_field}, not ecmf — use .key"
            )
        return self.key

    @classmethod
    def from_beta_map(cls, m: BetaMap, n_key: int | None = None) -> ECMFMap:
        """Re-tabulate onto ECMF. ``m`` should already be densified (D13).

        By default the **native** ECMF values at the β nodes are kept, merely
        sorted — no resampling. That matters: the node values are the map's own
        numbers, so the table is exact where the map is, and the only change from
        the β path is that ``PR`` is now interpolated against ``ECMF`` instead of
        against β. Resampling onto a uniform grid instead costs 3e−06 in ``PR``
        and 3.6e−05 in corrected work, which is the same size as the
        inconsistency this class exists to remove — measured end to end, it
        pushed two cells that held with the β path back out of the gate. Pass
        ``n_key`` only if a uniform key axis is worth that.
        """
        from scipy.interpolate import PchipInterpolator

        n_speed = m.PR.shape[1]
        if n_key is not None and n_key < 2:
            raise ValueError(f"n_key must be at least 2, got {n_key}")
        key_field = "PR" if m.kind == "turbine" else "ecmf"
        src_key = m.PR if key_field == "PR" else m.ecmf
        bad = [
            float(m.corrected_speed[j])
            for j in range(n_speed)
            if not (
                np.all(np.diff(src_key[:, j]) > 0.0) or np.all(np.diff(src_key[:, j]) < 0.0)
            )
        ]
        if bad:
            raise ValueError(
                f"{m.name}: {key_field} is not monotonic in beta at Nc={bad[:4]}"
                f"{' ...' if len(bad) > 4 else ''}, so it cannot key the map"
            )

        # Column-major, because every lookup slices a *column*: `evaluate` reads
        # `a[:, col]` six times per call, and on a C-ordered array that slice is
        # strided, so numpy materialises a copy of the whole speed line before
        # it can interpolate — making the lookup O(n_key) in a memcpy rather
        # than O(log n_key) in a search. Fortran order makes the same slice a
        # contiguous view: 7.42 µs to 1.69 µs at 1009×649, and the cost stops
        # growing with densification, which is what made `densify` 72 look
        # unaffordable when it is not.
        rows = n_key or src_key.shape[0]
        key = np.empty((rows, n_speed), order="F")
        pr = np.empty((rows, n_speed), order="F")
        cw = np.empty((rows, n_speed), order="F")
        eff = np.empty((rows, n_speed), order="F")
        # Wc is carried only for a PR-keyed map. On an ECMF-keyed one it follows
        # from the key exactly -- `Wc = ECMF·PR/√τ` -- and deriving it is what
        # makes the returned point reproduce the key it was asked for to 2e-16,
        # which interpolating a fourth column would quietly give up.
        wc = np.empty((rows, n_speed), order="F") if key_field == "PR" else None
        cols = [(pr, m.PR), (cw, m.corrected_work), (eff, m.efficiency)]
        if wc is not None:
            cols.append((wc, m.Wc))
        for j in range(n_speed):
            e = src_key[:, j]
            order = np.argsort(e)
            e_s = e[order]
            if n_key is None:
                key[:, j] = e_s
                for dst, src in cols:
                    dst[:, j] = src[:, j][order]
                continue
            grid = np.linspace(e_s[0], e_s[-1], n_key)
            key[:, j] = grid
            for dst, src in cols:
                dst[:, j] = PchipInterpolator(e_s, src[:, j][order])(grid)
        return cls(
            name=f"{m.name}→{key_field.upper()}",
            corrected_speed=m.corrected_speed,
            key=key,
            PR=pr,
            corrected_work=cw,
            efficiency=eff,
            gas=m.gas,
            key_field=key_field,
            Wc=wc,
        )

    def evaluate(self, ecmf: float, corrected_speed: float) -> MapPoint:
        """``PR`` and corrected work at a measured ECMF. No β anywhere."""
        n = self.corrected_speed
        if corrected_speed <= n[0]:
            j, w = 0, 0.0
        elif corrected_speed >= n[-1]:
            j, w = len(n) - 2, 1.0
        else:
            j = int(np.searchsorted(n, corrected_speed) - 1)
            w = (corrected_speed - n[j]) / (n[j + 1] - n[j])

        def at(col: int, a: np.ndarray, extrapolate: bool = True) -> float:
            """Linear in ECMF, holding the terminal *slope* past the ends.

            §3.8 measured extrapolation as the worst error source on these maps,
            and that stays true of the map's *value*. The *gradient* is a
            separate question, and clamping gets it wrong in a way that matters.
            ``np.interp`` returns a flat characteristic outside the data, and a
            flat characteristic is exactly the zero-restoring-force condition
            §3.34 identified as surge — so a clamped lookup manufactures an
            artificial surge at the table ends and pins anything that wanders
            out there. Holding the end slope keeps the restoring force that
            pushes the point back into the data, which is where the converged
            answer lives; the extrapolated values are transient, not the answer.
            """
            x, y = self.key[:, col], a[:, col]
            if not extrapolate:
                return float(np.interp(ecmf, x, y))
            # Bounded: the slope is held for one speed-line width past each end
            # and the value is frozen beyond that. The restoring force is what
            # the near field is for, and one width is far more than any
            # converging transient uses; past that the linear model means
            # nothing and an unbounded ray would just be a different way to
            # return a wrong number confidently.
            span = float(x[-1] - x[0])
            e = min(max(ecmf, float(x[0]) - span), float(x[-1]) + span)
            if e < x[0]:
                return float(y[0] + (e - x[0]) * (y[1] - y[0]) / (x[1] - x[0]))
            if e > x[-1]:
                return float(y[-1] + (e - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]))
            return float(np.interp(e, x, y))

        # Record that the demand left the table. Clamping is the right thing to
        # DO -- the map has no information out there and extrapolating it is
        # measurably the worst option -- but doing it silently is not. An
        # operating point pinned at the end has lost its restoring force in one
        # direction, and a run that converges anyway has converged to the edge of
        # the data rather than to an answer (`PLAN.md` §3.34).
        if ecmf < self.key[0, j] or ecmf > self.key[-1, j]:
            self.off_table[0] += 1

        pr = at(j, self.PR) * (1.0 - w) + at(j + 1, self.PR) * w
        cw = at(j, self.corrected_work) * (1.0 - w) + at(j + 1, self.corrected_work) * w
        # Efficiency stays clamped: it has a pole on `TranssonicCompressor`
        # (§3.6), and a slope taken next to a pole is not a slope.
        ef = (at(j, self.efficiency, extrapolate=False) * (1.0 - w)
              + at(j + 1, self.efficiency, extrapolate=False) * w)
        tau = 1.0 + cw / (self.gas.cp * T_REF)
        lo = float(min(self.key[0, j], self.key[0, j + 1]))
        hi = float(max(self.key[-1, j], self.key[-1, j + 1]))
        e = float(np.clip(ecmf, lo, hi))
        if self.key_field == "PR":
            # PR is the key, so it is returned exactly as asked; Wc comes from
            # the table and ECMF follows. A turbine tabulates the expansion
            # ratio p₀₁/p₀₂, so ECMF = Wc·√τ·PR — the reciprocal of the
            # compressor factor, and getting it upside down is silent because
            # both are dimensionally fine (§3.38).
            pr = e
            wc = at(j, self.Wc) * (1.0 - w) + at(j + 1, self.Wc) * w
            ecmf_out = wc * math.sqrt(tau) * pr
        else:
            wc = e * pr / math.sqrt(tau)
            ecmf_out = e
        return MapPoint(
            beta=math.nan,  # deliberately absent: this map has no beta
            corrected_speed=corrected_speed,
            Wc=wc,
            PR=pr,
            corrected_work=cw,
            ecmf=ecmf_out,
            efficiency=ef,
        )


def _infer_kind(Wc: np.ndarray, PR: np.ndarray) -> str:
    """Compressor or turbine, from the sign of ``dWc/dPR`` along each line.

    A compressor throttled toward surge passes *less* flow at *more* pressure
    ratio; a turbine passes *more* flow as the expansion ratio opens up, until
    it chokes. So the sign of ``dWc/dPR`` separates them, and on the seventeen
    supplied workbooks it does so unanimously — positive on **64 of 64** turbine
    speed lines and negative on **109 of 109** compressor lines, no map mixed.

    Preferred over the other available signal, which is that compressors carry a
    ``surge_line`` sheet and turbines do not: that agrees on all seventeen too,
    but it is a filing convention and this is the machine.
    """
    rising = 0
    for j in range(PR.shape[1]):
        order = np.argsort(PR[:, j])
        d = np.diff(Wc[order, j])
        rising += int(np.sum(d > 0.0) > np.sum(d < 0.0))
    return "turbine" if rising * 2 > PR.shape[1] else "compressor"


def load_beta_map(
    path: str | Path,
    gas: PerfectGas,
    name: str | None = None,
    kind: str | None = None,
) -> BetaMap:
    """Read a workbook and derive corrected work and ECMF.

    Corrected work is evaluated at the reference condition, where ``θ = 1``, and
    is signed as the stagnation enthalpy change across the machine — positive
    into a compressor, negative out of a turbine.

    .. math::

        \\Delta h_0 = \\frac{h(T_{02s}) - h(T_{ref})}{\\eta}
        \\quad\\text{(compressor)},
        \\qquad
        \\Delta h_0 = -\\eta\\,\\bigl(h(T_{ref}) - h(T_{02s})\\bigr)
        \\quad\\text{(turbine)}

    For a perfect gas these reduce to ``cp·T_ref·(PR^k − 1)/η`` and
    ``−η·cp·T_ref·(1 − PR^−k)``, and the resulting ``CW`` is independent of
    ``T₀₁`` by construction. **Both differences are load-bearing.** The
    compressor form applied to a turbine gives a temperature *rise* — ``τ > 1``
    for every cell of every supplied turbine map, where a PR 3, η 0.9 stage must
    drop 24% — and it divides by η where a turbine multiplies, which is what
    made three of the six turbine workbooks unloadable and a fourth silently
    wrong (`PLAN.md` §3.38).

    ``kind`` is inferred by :func:`_infer_kind` when not given.

    An earlier version of this docstring claimed ``η ≤ 0`` needed no special
    case, because ``Δh₀``'s numerator changes sign with its denominator. That
    holds on **12 of 12** such compressor cells and fails on **35 of 36**
    turbine cells, where η goes negative while PR stays above 1 and nothing
    cancels. Those cells are now repaired from their neighbours along β and
    listed in :attr:`BetaMap.repaired_efficiency`.
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

    if kind is None:
        kind = _infer_kind(Wc, PR)
    if kind not in ("compressor", "turbine"):
        raise ValueError(f"{name}: kind must be 'compressor' or 'turbine', got {kind!r}")

    # eta <= 0 is repaired on TURBINES ONLY, and the asymmetry is the point.
    #
    # On a compressor those cells all have PR < 1 -- 12 of 12 across the supplied
    # maps -- so dividing by a negative eta cancels against a negative numerator
    # and yields tau > 1: work going in, temperature up, pressure down. That is
    # a stalled corner correctly modelled, it is what the loader has always done,
    # and repairing eta there would turn it into cooling. Left exactly alone.
    #
    # On a turbine eta multiplies, so a non-positive eta means no work or work
    # of the wrong sign, and 35 of 36 such cells have PR > 1 where nothing
    # cancels. Those are padding -- 29 of 36 sit on the first or last beta row --
    # so interpolate eta from the valid cells of the same speed line and record
    # what was touched. Silently dividing by -0.0339 is how `MediumPqPTurbine`
    # came to hold -153 kJ/kg against an ideal 5.2 kJ/kg (`PLAN.md` §3.38).
    repaired: tuple[tuple[int, int], ...] = ()
    if kind == "turbine":
        eff = np.array(eff, dtype=float, copy=True)
        bad = ~(eff > 0.0)
        repaired = tuple((int(i), int(j)) for i, j in zip(*np.where(bad)))
        for j in range(eff.shape[1]):
            col = bad[:, j]
            if not col.any():
                continue
            if col.all():
                raise ValueError(
                    f"{name}: speed line {speed[j]:g} has no cell with eta > 0, "
                    f"so there is nothing to interpolate from"
                )
            eff[col, j] = np.interp(beta[col], beta[~col], eff[~col, j])

    if kind == "compressor":
        # Work INTO the flow: actual rise is the isentropic rise over eta.
        corrected_work = gas.cp * T_REF * (PR**gas.gm1_over_g - 1.0) / eff
    else:
        # Work OUT of the flow. Two things differ and both matter: the machine
        # expands, so the isentropic change is a DROP of cp·T_ref·(1 − PR^−k)
        # with PR the expansion ratio, and a turbine delivers LESS than ideal,
        # so eta multiplies rather than divides. Signed as the stagnation
        # enthalpy change across the machine, which makes it negative and tau
        # less than one -- a turbine cools. Applying the compressor form here
        # gave every turbine cell a temperature RISE: +41% where a PR 3, eta
        # 0.9 stage drops 24%.
        corrected_work = -eff * gas.cp * T_REF * (1.0 - PR**-gas.gm1_over_g)

    tau = 1.0 + corrected_work / (gas.cp * T_REF)
    if np.any(tau <= 0.0):
        raise ValueError(f"{name}: non-physical temperature ratio derived from the map")
    # ECMF ≡ W·√T₀₂/p₀₂ = Wc·√τ · (p₀₁/p₀₂). A compressor tabulates PR = p₀₂/p₀₁
    # so that factor is 1/PR; a turbine tabulates the expansion ratio p₀₁/p₀₂,
    # so it is PR.
    # Written as two expressions rather than one with a reciprocal: multiplying
    # by 1/PR differs from dividing by PR in the last bit, and the compressor
    # path is meant to be untouched by this change, not almost untouched.
    if kind == "turbine":
        ecmf = Wc * np.sqrt(tau) * PR
    else:
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
        kind=kind,
        repaired_efficiency=repaired,
    )
