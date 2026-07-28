"""Design a Q1D test case *from* a map, instead of hoping a map fits a duct.

The first attempt at running the solver on a real map failed three ways in
succession — ``Fx ≈ 60,000 N`` at the disk cell driving the pressure negative,
then reverse flow at the sampling station, then ``InfeasibleOperatingPoint``
with ``Φ = 1.99`` against a sonic maximum of 1.28. All three are one mistake:
a geometry and a back pressure were chosen first and the map was asked to fit
them.

It has to run the other way. A map point fixes ``Wc``, ``PR`` and ``Δh₀/θ``;
given an inlet stagnation state those fix the mass flow and both stagnation
states. The duct area is then whatever puts station 1 at a chosen axial Mach
number, and the back pressure is whatever station 2's static pressure comes out
to be. Nothing is guessed, and infeasibility is *reported* — with the reason —
before a solver is ever constructed.

The reachability sweep is the other half: for a fixed area, which part of a
speed line can be reached at all? That is the question a throttle sweep needs
answered before it starts, not after it diverges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .analytic import (
    InfeasibleOperatingPoint,
    StaticState,
    max_flow_function,
    static_from_stagnation,
)
from .gas import PerfectGas
from .grid import Grid
from .maps import P_REF, T_REF, BetaMap, MapPoint

__all__ = [
    "DuctDesign",
    "design_from_map",
    "reachable_ecmf_range",
    "steady_profile",
    "build_case",
]

#: Fraction of the sonic flow function a station must stay below. The inversion
#: is exact up to ``Φmax`` but its derivative vanishes there, so a station sat
#: against the limit turns a small transient into a failed lookup.
CHOKE_MARGIN = 0.98


@dataclass(frozen=True)
class DuctDesign:
    """A constant-area duct and back pressure that realise one map point."""

    point: MapPoint
    gas: PerfectGas

    area: float
    T01: float
    p01: float
    p_back: float

    W: float
    T02: float
    p02: float
    station1: StaticState
    station2: StaticState

    @property
    def theta(self) -> float:
        return self.T01 / T_REF

    @property
    def delta(self) -> float:
        return self.p01 / P_REF

    @property
    def dh0(self) -> float:
        """Actual specific work at this inlet temperature [J/kg]."""
        return self.point.corrected_work * self.theta

    @property
    def Fx(self) -> float:
        """Net axial momentum source [N] — blade force plus wall reaction."""
        return (self.station2.p - self.station1.p) * self.area + self.W * (
            self.station2.u - self.station1.u
        )

    @property
    def SWx(self) -> float:
        """Shaft power into the fluid [W]."""
        return self.W * self.dh0

    def describe(self) -> str:
        p = self.point
        return (
            f"Nc={p.corrected_speed:.4g} beta={p.beta:.4g} ECMF={p.ecmf:.5g} "
            f"PR={p.PR:.5g} CW={p.corrected_work / 1e3:.5g} kJ/kg eta={p.efficiency:.5g}\n"
            f"  W={self.W:.5g} kg/s  A={self.area:.5g} m^2  "
            f"M1={self.station1.M:.4g}  M2={self.station2.M:.4g}\n"
            f"  T01={self.T01:.5g} K -> T02={self.T02:.5g} K   "
            f"p01={self.p01:.6g} Pa -> p02={self.p02:.6g} Pa   pb={self.p_back:.6g} Pa\n"
            f"  Fx={self.Fx:.6g} N   SWx={self.SWx / 1e6:.6g} MW"
        )


def _exit_stagnation(
    point: MapPoint, T01: float, p01: float, gas: PerfectGas, kind: str = "compressor"
):
    """Exit stagnation state from a map point.

    ``PR`` means different things to the two machines and both are the natural
    convention for their own map: a compressor tabulates ``p₀₂/p₀₁``, a turbine
    the expansion ratio ``p₀₁/p₀₂``, so that both read above 1. Using the
    compressor form on a turbine raises the pressure while the energy equation
    lowers the temperature — a disk that compresses and cools at once.
    """
    theta = T01 / T_REF
    T02 = T01 + point.corrected_work * theta / gas.cp
    return T02, (p01 / point.PR if kind == "turbine" else point.PR * p01)


def design_from_map(
    beta_map: BetaMap,
    corrected_speed: float,
    ecmf: float,
    *,
    inlet_mach: float = 0.45,
    T01: float = T_REF,
    p01: float = P_REF,
    gas: PerfectGas | None = None,
) -> DuctDesign:
    """Size a duct so that one map point is the steady state.

    ``inlet_mach`` is the only free choice. It sets the area, and through the
    area it sets how much throttling headroom the case has: too high and the
    inlet chokes as soon as the operating point moves up the speed line, too
    low and the disk sits in a nearly stagnant duct where the momentum source
    is a large fraction of the momentum flux.

    Raises :class:`~q1d.analytic.InfeasibleOperatingPoint` — before building
    anything — if the resulting exit station cannot pass the flow subsonically.
    The compressor exit *must* be subsonic; a supersonic exit is not a valid
    reading of any of these maps.
    """
    gas = gas or beta_map.gas
    if not 0.0 < inlet_mach < 1.0:
        raise ValueError(f"inlet_mach must be in (0, 1), got {inlet_mach!r}")
    if T01 <= 0.0 or p01 <= 0.0:
        raise ValueError(f"inlet stagnation state must be positive, got {T01!r}, {p01!r}")

    point = beta_map.evaluate_at_ecmf(ecmf, corrected_speed)
    theta, delta = T01 / T_REF, p01 / P_REF

    W = point.Wc * delta / math.sqrt(theta)
    kind = getattr(beta_map, "kind", "compressor")
    T02, p02 = _exit_stagnation(point, T01, p01, gas, kind)
    # The direction of the temperature change is the machine's definition, so
    # check it against the machine rather than assuming compression.
    if kind == "turbine" and T02 >= T01:
        raise InfeasibleOperatingPoint(
            f"map point does work {point.corrected_work:.6g} J/kg/theta, giving "
            f"T02={T02:.6g} K >= T01={T01:.6g} K — a turbine must expand and cool"
        )
    if kind != "turbine" and T02 <= T01:
        raise InfeasibleOperatingPoint(
            f"map point does work {point.corrected_work:.6g} J/kg/theta, giving "
            f"T02={T02:.6g} K <= T01={T01:.6g} K — not a compressor operating point"
        )
    if T02 <= 0.0:
        raise InfeasibleOperatingPoint(
            f"exit stagnation temperature {T02:.6g} K is non-physical: the map "
            f"extracts {-point.corrected_work * theta:.6g} J/kg from a flow entering at "
            f"{T01:.6g} K. Turbine maps are tabulated for turbine-entry conditions; "
            f"running one from a standard-day inlet asks it for more enthalpy than "
            f"the flow has"
        )

    from .analytic import flow_function

    phi1 = flow_function(inlet_mach, gas)
    phi_max = max_flow_function(gas)
    if phi1 >= phi_max * CHOKE_MARGIN:
        raise InfeasibleOperatingPoint(
            f"inlet station would sit against the choke limit: Phi1={phi1:.6g} against "
            f"{phi_max:.6g} (margin {CHOKE_MARGIN:.3g}) at M1={inlet_mach:.4g}. The "
            f"inversion is exact up to the limit but its derivative vanishes there, so a "
            f"station this close turns any transient into a failed lookup. Lower inlet_mach"
        )
    area = W * math.sqrt(gas.R * T01) / (p01 * phi1)

    # With equal areas the inlet always saturates first for a compressor:
    # exit choke would need Phi1/Phimax = PR/sqrt(tau) > 1, and PR/sqrt(tau) is
    # ~1.17 at design (PLAN.md Phase 1). This guard is therefore unreachable for
    # a well-formed compressor point and exists to catch a malformed one.
    phi2 = W * math.sqrt(gas.R * T02) / (area * p02)
    # A turbine's exit is the low-pressure station, so unlike a compressor it is
    # the one that saturates first and this guard is genuinely reachable.
    if kind != "turbine" and phi2 >= phi_max * CHOKE_MARGIN:
        raise InfeasibleOperatingPoint(
            f"exit station would choke: Phi2={phi2:.6g} against {phi_max:.6g} "
            f"(margin {CHOKE_MARGIN:.3g}) at M1={inlet_mach:.4g}. At equal areas this "
            f"needs PR/sqrt(tau) < 1, which no valid compressor point on these maps "
            f"satisfies — check the map point, not the geometry"
        )

    st1 = static_from_stagnation(T01, p01, W, area, gas)
    st2 = static_from_stagnation(T02, p02, W, area, gas)
    return DuctDesign(
        point=point,
        gas=gas,
        area=area,
        T01=T01,
        p01=p01,
        p_back=st2.p,
        W=W,
        T02=T02,
        p02=p02,
        station1=st1,
        station2=st2,
    )


def reachable_ecmf_range(
    beta_map: BetaMap,
    corrected_speed: float,
    area: float,
    *,
    T01: float = T_REF,
    p01: float = P_REF,
    gas: PerfectGas | None = None,
    samples: int = 401,
) -> tuple[float, float]:
    """Which part of a speed line a fixed duct can reach, as an ECMF interval.

    A throttle sweep walks along the speed line at fixed geometry, so the area
    that was sized for one point has to pass every other point too. Both
    stations are checked: with equal areas the exit is the limiting one
    whenever ``PR/√τ < 1``, which on these maps is most of the low-speed
    region.

    Returns the largest contiguous interval containing the feasible samples;
    raises if none are feasible.
    """
    gas = gas or beta_map.gas
    e, _, _, _, _ = beta_map._speed_line(corrected_speed)
    lo, hi = float(min(e[0], e[-1])), float(max(e[0], e[-1]))
    phi_lim = max_flow_function(gas) * CHOKE_MARGIN
    theta, delta = T01 / T_REF, p01 / P_REF

    ok = []
    for target in np.linspace(lo, hi, samples):
        pt = beta_map.evaluate_at_ecmf(float(target), corrected_speed)
        W = pt.Wc * delta / math.sqrt(theta)
        T02, p02 = _exit_stagnation(pt, T01, p01, gas, getattr(beta_map, "kind", "compressor"))
        if T02 <= 0.0 or p02 <= 0.0:
            continue
        phi1 = W * math.sqrt(gas.R * T01) / (area * p01)
        phi2 = W * math.sqrt(gas.R * T02) / (area * p02)
        if phi1 < phi_lim and phi2 < phi_lim:
            ok.append(float(target))

    if not ok:
        raise InfeasibleOperatingPoint(
            f"no point on the Nc={corrected_speed:.4g} line of {beta_map.name} is "
            f"reachable at A={area:.6g} m^2 — every sample chokes at one station "
            f"or the other. The area is too small for this speed line"
        )
    return min(ok), max(ok)


def steady_profile(
    design: DuctDesign, n_cells: int, disk_cell: int, n_smear: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The discrete steady profile of a smeared disk, from the flux balance.

    A sharp two-state seed is *not* a discrete steady state: cell ``i`` cannot
    present station 1 to its left face and station 2 to its right face at the
    same time, so a one-cell jump leaves the disk cell out of balance by the
    full source. Integrating ``F_{k+1} = F_k + q/n_smear`` across the smear
    region and inverting each face flux (:func:`~q1d.analytic.state_from_flux`)
    gives the profile the solver is actually looking for.

    Measured on ``SubsonicCompressor`` at design speed with 21 smear cells: the
    residual through the smear interior is 1e-5 to 1e-4 of the source scale,
    against ~1 for the sharp seed. What is left sits at the two kinks where the
    ramp meets the uniform regions, at 1–3%, and is the limiter reacting to a
    slope discontinuity.
    """
    from .analytic import state_from_flux

    gas, A = design.gas, design.area
    st1, st2 = design.station1, design.station2

    def flux_of(st):
        return (
            st.rho * st.u * A,
            (st.rho * st.u**2 + st.p) * A,
            st.rho * st.u * (gas.cp * st.T + 0.5 * st.u**2) * A,
        )

    rho = np.empty(n_cells)
    u = np.empty(n_cells)
    p = np.empty(n_cells)
    rho[:disk_cell], u[:disk_cell], p[:disk_cell] = st1.rho, st1.u, st1.p
    tail = slice(disk_cell + n_smear, None)
    rho[tail], u[tail], p[tail] = st2.rho, st2.u, st2.p

    F = np.array(flux_of(st1))
    q = np.array([0.0, design.Fx, design.SWx]) / n_smear
    for k in range(n_smear):
        # Cell centre carries the mean of its two face fluxes.
        st = state_from_flux(F + 0.5 * q, A, gas)
        rho[disk_cell + k], u[disk_cell + k], p[disk_cell + k] = st.rho, st.u, st.p
        F = F + q
    return rho, u, p


def build_case(
    design: DuctDesign,
    beta_map: BetaMap,
    *,
    n_cells: int = 401,
    length: float = 1.0,
    n_smear: int = 81,
    corrected_speed: float | None = None,
    sample_offset: int = 12,
    ramp_evaluations: int = 0,
    config=None,
):
    """Solver plus disk for one map point, seeded with :func:`steady_profile`.

    Seeding from the answer does not make the acceptance test circular. The
    disk still measures the field and looks the point up for itself, and the
    test is that it *stays* there — a wrong closure walks away from a correct
    seed, which is exactly how the exit-corrected-flow closure was caught.

    ``n_smear`` defaults to a fifth of the duct because a real map's source is
    large: see ``PLAN.md`` §3.9 for the measured stability limit on source
    strength per cell.
    """
    from .boundary import StagnationInletStaticOutlet
    from .compressor import InletFlowCompressor
    from .solver import ReferenceState, Solver

    if n_smear < 1 or n_smear >= n_cells:
        raise ValueError(f"n_smear must be in [1, {n_cells}), got {n_smear!r}")

    grid = Grid.uniform(0.0, length, n_cells, design.area)
    cell = (n_cells - n_smear) // 2
    if cell - sample_offset < 0 or cell + n_smear >= n_cells:
        raise ValueError(
            f"a {n_smear}-cell disk at {cell} of {n_cells} does not clear the "
            f"sampling offset {sample_offset}"
        )

    bc = StagnationInletStaticOutlet(p0_in=design.p01, T0_in=design.T01, p_back=design.p_back)
    reference = ReferenceState(
        rho=design.station1.rho,
        u=max(design.station1.u, 1.0),
        p=design.p01,
        vol=float(np.mean(grid.vol[1:-1])),
    )
    disk = InletFlowCompressor(
        cell=cell,
        beta_map=beta_map,
        corrected_speed=(
            corrected_speed if corrected_speed is not None else design.point.corrected_speed
        ),
        sample_offset=sample_offset,
        n_smear=n_smear,
        ramp_evaluations=ramp_evaluations,
    )
    solver = Solver(grid, design.gas, bc, reference, config=config, source=disk)
    solver.set_state(*steady_profile(design, n_cells, cell, n_smear))
    return solver, disk
