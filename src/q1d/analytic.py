"""Closed-form reference solutions — the 0D model.

This module is the arbiter for every later gate. It solves the compressor
duct algebraically, with no PDE and no iteration beyond a single guarded
Newton inversion, so that the Q1D solver can be checked against a number
rather than against a plausible-looking plot.

The central quantity is the **flow function**

.. math::

    \\Phi \\equiv \\frac{\\dot m \\sqrt{R T_0}}{A p_0}
            = \\sqrt{\\gamma}\\, M \\left(1 + \\tfrac{\\gamma-1}{2}M^2
              \\right)^{-\\frac{\\gamma+1}{2(\\gamma-1)}}

which is dimensionless and carries **no reference state**. That is why it is
the internal key for component maps (``PLAN.md`` §4.4): the design-vs-runtime
mismatch that corrupted the prototype's lookup is unrepresentable in terms of
:math:`\\Phi`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .gas import PerfectGas

__all__ = [
    "InfeasibleOperatingPoint",
    "StaticState",
    "CompressorSolution",
    "flow_function",
    "d_flow_function",
    "max_flow_function",
    "mach_from_flow_function",
    "static_from_stagnation",
    "choked_mass_flow",
    "compressor_exit_stagnation",
    "compressor_source_terms",
    "zero_d_compressor",
]


class InfeasibleOperatingPoint(ValueError):
    """The requested state cannot exist.

    Raised when a mass flow exceeds what an area can pass at the given
    stagnation state. Distinct from *choked*: a choked duct is a valid
    physical state, whereas this is a request for one that does not exist.
    The legacy ``setStatic`` printed ``'choked'`` for this case, which
    conflated the two (``PLAN.md`` §4.2 #14).
    """


@dataclass(frozen=True)
class StaticState:
    """Static thermodynamic state plus velocity at a station."""

    p: float
    T: float
    rho: float
    u: float
    M: float
    c: float

    @property
    def mass_flux(self) -> float:
        """``rho * u`` [kg/s/m^2]."""
        return self.rho * self.u


@dataclass(frozen=True)
class CompressorSolution:
    """Complete steady state of a constant-area duct containing a compressor."""

    W: float
    regime: str  # 'subsonic' | 'exit_choked'

    # stagnation states
    p01: float
    T01: float
    p02: float
    T02: float
    tau: float  # T02 / T01

    # station states
    station1: StaticState
    station2: StaticState

    # dimensionless coordinates
    phi1: float
    phi2: float

    # source terms
    Fx: float  # [N]   axial blade force on the fluid
    SWx: float  # [W]   shaft power into the fluid
    Fhat: float  # Fx  / (p01 * A1)
    Shat: float  # SWx / (W * cp * T01)

    # geometry / map, echoed for traceability
    A1: float
    A2: float
    PR: float
    eta: float


# ---------------------------------------------------------------------------
# Flow function and its inversion
# ---------------------------------------------------------------------------


def flow_function(M, gas: PerfectGas):
    """:math:`\\Phi(M) = \\dot m \\sqrt{R T_0} / (A p_0)`.

    Accepts scalars or numpy arrays. Monotonically increasing on ``M < 1``,
    decreasing on ``M > 1``, with a stationary maximum at ``M = 1``.
    """
    k = gas.gp1 / (2.0 * gas.gm1)
    return math.sqrt(gas.gamma) * M * (1.0 + 0.5 * gas.gm1 * M * M) ** (-k)


def d_flow_function(M, gas: PerfectGas):
    """:math:`d\\Phi/dM`, in closed form.

    Differentiating and collecting terms gives

    .. math::

        \\frac{d\\Phi}{dM} = \\sqrt{\\gamma}\\,(1 - M^2)
            \\left(1 + \\tfrac{\\gamma-1}{2}M^2\\right)^{
                -\\frac{\\gamma+1}{2(\\gamma-1)} - 1}

    The ``(1 - M^2)`` factor makes the vanishing derivative at ``M = 1``
    explicit: that is the source of the ill-conditioning near choke noted in
    ``PLAN.md`` §4.4, not an artefact of any particular solver.
    """
    k = gas.gp1 / (2.0 * gas.gm1)
    return math.sqrt(gas.gamma) * (1.0 - M * M) * (1.0 + 0.5 * gas.gm1 * M * M) ** (-k - 1.0)


def max_flow_function(gas: PerfectGas) -> float:
    """:math:`\\Phi(M=1)` — the most any area can pass per unit ``p0/\\sqrt{T_0}``."""
    return flow_function(1.0, gas)


def mach_from_flow_function(
    phi: float,
    gas: PerfectGas,
    supersonic: bool = False,
    tol: float = 1e-14,
    maxiter: int = 100,
) -> float:
    """Invert :math:`\\Phi \\to M` by bracketed Newton iteration.

    Replaces the legacy fixed-point iteration on static temperature, whose
    convergence rate is exactly :math:`M^2` — it needed 24 iterations at the
    operating point and silently hit its 100-iteration cap above
    :math:`M \\approx 0.92` (``PLAN.md`` §4.2 #5). Newton is quadratic and the
    bisection safeguard makes it unconditionally convergent.

    Accuracy near choke is limited by the physics, not the algorithm. Since
    :math:`\\Phi(M) \\approx \\Phi_{max} - \\tfrac{1}{2}|\\Phi''|(M-1)^2`, a
    relative perturbation :math:`\\delta` in ``phi`` moves ``M`` by
    :math:`O(\\sqrt{\\delta})`; at double precision that is ~1e-8 in ``M``.
    """
    phi_max = max_flow_function(gas)

    if phi < 0.0:
        raise ValueError(f"flow function must be non-negative, got {phi!r}")
    if phi > phi_max * (1.0 + 1e-12):
        raise InfeasibleOperatingPoint(
            f"flow function {phi:.12g} exceeds the sonic maximum {phi_max:.12g}; "
            "no state passes this mass flow through this area at this stagnation state"
        )

    phi = min(phi, phi_max)
    if phi == 0.0:
        if supersonic:
            raise InfeasibleOperatingPoint("zero flow function has no supersonic solution")
        return 0.0
    if phi_max - phi <= phi_max * 1e-15:
        return 1.0

    if supersonic:
        lo, hi = 1.0, 1.0
        while flow_function(hi, gas) > phi:
            hi *= 1.5
            if hi > 1e4:  # pragma: no cover - unreachable for finite phi > 0
                raise InfeasibleOperatingPoint("no supersonic bracket found")
    else:
        lo, hi = 0.0, 1.0

    # Orientation: f = Phi(M) - phi is increasing in M on the subsonic branch
    # and decreasing on the supersonic branch. Track it so the bracket update
    # is branch-agnostic.
    increasing = not supersonic

    M = 0.5 * (lo + hi)
    for _ in range(maxiter):
        f = flow_function(M, gas) - phi
        if (f > 0.0) == increasing:
            hi = M
        else:
            lo = M

        d = d_flow_function(M, gas)
        step_ok = d != 0.0
        if step_ok:
            M_new = M - f / d
            step_ok = lo < M_new < hi
        if not step_ok:
            M_new = 0.5 * (lo + hi)

        if abs(M_new - M) <= tol * max(1.0, abs(M_new)):
            return M_new
        M = M_new

    raise RuntimeError(  # pragma: no cover - the bisection safeguard prevents this
        f"Mach inversion failed to converge for phi={phi!r}"
    )


# ---------------------------------------------------------------------------
# Station states
# ---------------------------------------------------------------------------


def static_from_stagnation(
    T0: float,
    p0: float,
    W: float,
    A: float,
    gas: PerfectGas,
    supersonic: bool = False,
) -> StaticState:
    """Stagnation state plus mass flow through an area -> static state.

    All SI: ``p0`` in Pa (the legacy routine used kPa), ``T0`` in K, ``W`` in
    kg/s, ``A`` in m^2.

    Non-iterative apart from the Mach inversion: form :math:`\\Phi`, invert it
    once, then every other quantity follows algebraically.
    """
    if A <= 0.0:
        raise ValueError(f"area must be positive, got {A!r}")
    if p0 <= 0.0 or T0 <= 0.0:
        raise ValueError(f"stagnation state must be positive, got p0={p0!r}, T0={T0!r}")

    phi = W * math.sqrt(gas.R * T0) / (A * p0)
    M = mach_from_flow_function(phi, gas, supersonic=supersonic)

    T = T0 / (1.0 + 0.5 * gas.gm1 * M * M)
    p = p0 * (T / T0) ** gas.g_over_gm1
    rho = p / (gas.R * T)
    c = gas.speed_of_sound(T)
    return StaticState(p=p, T=T, rho=rho, u=M * c, M=M, c=c)


def choked_mass_flow(p0: float, T0: float, A: float, gas: PerfectGas) -> float:
    """Largest mass flow an area can pass at the given stagnation state."""
    return max_flow_function(gas) * A * p0 / math.sqrt(gas.R * T0)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


def compressor_exit_stagnation(
    T01: float, p01: float, PR: float, eta: float, gas: PerfectGas
) -> tuple[float, float]:
    """Exit stagnation state from pressure ratio and isentropic efficiency.

    .. math::

        p_{02} = PR \\cdot p_{01}, \\qquad
        \\frac{T_{02}}{T_{01}} = 1 + \\frac{PR^{(\\gamma-1)/\\gamma} - 1}{\\eta}

    Returns ``(T02, p02)``.
    """
    if PR <= 0.0:
        raise ValueError(f"pressure ratio must be positive, got {PR!r}")
    if not 0.0 < eta <= 1.0:
        raise ValueError(f"isentropic efficiency must be in (0, 1], got {eta!r}")
    tau = 1.0 + (PR**gas.gm1_over_g - 1.0) / eta
    return T01 * tau, p01 * PR


def compressor_source_terms(
    T01: float,
    p01: float,
    W: float,
    PR: float,
    eta: float,
    A1: float,
    A2: float,
    gas: PerfectGas,
) -> tuple[float, float]:
    """Axial blade force and shaft power for a given upstream state and flow.

    This is the function Phase 3 calls at runtime, from the *locally measured*
    upstream stagnation state — not a tabulated value (``PLAN.md`` P1).

    .. math::

        F_x = p_{s2}A_2 - p_{s1}A_1 + \\dot m (u_2 - u_1), \\qquad
        \\dot S_W = \\dot m (h_{02} - h_{01})

    ``Fx`` is the force **on the fluid**, so it is positive for a compressor at
    every operating point. It has no reason to vanish at equilibrium: the blade
    force is balanced by the static pressure rise it creates, which lives in
    the flux term, not the source (``PLAN.md`` §4.3).

    Returns ``(Fx [N], SWx [W])``.
    """
    T02, p02 = compressor_exit_stagnation(T01, p01, PR, eta, gas)
    st1 = static_from_stagnation(T01, p01, W, A1, gas)
    st2 = static_from_stagnation(T02, p02, W, A2, gas)

    Fx = st2.p * A2 - st1.p * A1 + W * (st2.u - st1.u)
    SWx = W * (gas.enthalpy(T02) - gas.enthalpy(T01))
    return Fx, SWx


def zero_d_compressor(
    p01: float,
    T01: float,
    pb: float,
    PR: float,
    eta: float,
    A1: float,
    A2: float,
    gas: PerfectGas,
) -> CompressorSolution:
    """Steady state of a duct containing a compressor, solved algebraically.

    The operating point is set by matching the compressor's exit stagnation
    state against the imposed exit static back pressure ``pb``. That fixes
    ``M2``, hence :math:`\\Phi_2`, hence the mass flow. Everything else follows.

    Parameters mirror the Q1D boundary conditions exactly: ``p01``/``T01`` are
    the imposed inlet stagnation state and ``pb`` the imposed exit static
    pressure.

    Raises :class:`InfeasibleOperatingPoint` if station 1 cannot pass the
    resulting mass flow.
    """
    T02, p02 = compressor_exit_stagnation(T01, p01, PR, eta, gas)
    tau = T02 / T01

    if pb <= 0.0:
        raise ValueError(f"back pressure must be positive, got {pb!r}")
    if p02 <= pb:
        raise InfeasibleOperatingPoint(
            f"exit stagnation pressure {p02:.6g} Pa does not exceed the back pressure "
            f"{pb:.6g} Pa; no forward flow"
        )

    # Exit Mach from the stagnation-to-static ratio the back pressure demands.
    M2 = math.sqrt(2.0 / gas.gm1 * ((p02 / pb) ** gas.gm1_over_g - 1.0))
    if M2 > 1.0:
        regime = "exit_choked"
        M2 = 1.0
    else:
        regime = "subsonic"

    phi2 = flow_function(M2, gas)
    W = phi2 * A2 * p02 / math.sqrt(gas.R * T02)

    phi1 = W * math.sqrt(gas.R * T01) / (A1 * p01)
    if phi1 > max_flow_function(gas):
        raise InfeasibleOperatingPoint(
            f"station 1 cannot pass {W:.6g} kg/s: required flow function {phi1:.6g} "
            f"exceeds the sonic maximum {max_flow_function(gas):.6g}. "
            f"Inlet choke limit is {choked_mass_flow(p01, T01, A1, gas):.6g} kg/s"
        )

    st1 = static_from_stagnation(T01, p01, W, A1, gas)
    st2 = static_from_stagnation(T02, p02, W, A2, gas)

    Fx = st2.p * A2 - st1.p * A1 + W * (st2.u - st1.u)
    SWx = W * (gas.enthalpy(T02) - gas.enthalpy(T01))

    return CompressorSolution(
        W=W,
        regime=regime,
        p01=p01,
        T01=T01,
        p02=p02,
        T02=T02,
        tau=tau,
        station1=st1,
        station2=st2,
        phi1=phi1,
        phi2=phi2,
        Fx=Fx,
        SWx=SWx,
        Fhat=Fx / (p01 * A1),
        Shat=SWx / (W * gas.cp * T01),
        A1=A1,
        A2=A2,
        PR=PR,
        eta=eta,
    )
