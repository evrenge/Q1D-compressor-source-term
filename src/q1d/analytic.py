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
from collections.abc import Sequence
from dataclasses import dataclass

from .gas import PerfectGas

#: Standard-day reference temperature for corrected parameters.
T_REF_STD = 288.15

#: Relative slack on feasibility checks, so a state that is sonic to round-off
#: is accepted rather than rejected. Used consistently by every feasibility
#: test in this module.
FEASIBILITY_SLACK = 1e-12

#: Upper bound of the supersonic bracket search. Flow functions below roughly
#: 1.5e-18 correspond to Mach numbers above this and are rejected.
MAX_SUPERSONIC_MACH = 1e4

__all__ = [
    "T_REF_STD",
    "FEASIBILITY_SLACK",
    "MAX_SUPERSONIC_MACH",
    "InfeasibleOperatingPoint",
    "InletChokeLimited",
    "minimum_back_pressure",
    "StaticState",
    "CompressorSolution",
    "flow_function",
    "d_flow_function",
    "max_flow_function",
    "mach_from_flow_function",
    "static_from_stagnation",
    "state_from_flux",
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


class InletChokeLimited(InfeasibleOperatingPoint):
    """The back pressure demands more flow than the inlet can pass.

    A *distinct* condition from :class:`InfeasibleOperatingPoint`: the duct is
    in a perfectly real inlet-choked state with the mass flow pinned at the
    inlet choke limit. What does not exist is the **subsonic-matched** solution
    this 0D model solves for, because with the inlet sonic the back pressure no
    longer sets the flow.

    Carries ``W_choke`` and ``pb_min`` so a caller can see the boundary rather
    than having to search for it. See ``PLAN.md`` §8 for why the inlet-choked
    branch is not modelled here.
    """

    def __init__(self, message: str, W_choke: float, pb_min: float):
        super().__init__(message)
        self.W_choke = W_choke
        self.pb_min = pb_min


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
    Fx: float  # [N] net axial momentum source: blade force + wall reaction
    SWx: float  # [W] shaft power into the fluid
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

    # NaN must be rejected explicitly. Every comparison against NaN is False,
    # so without this guard NaN slips past both range checks, past the early
    # exits, and then drives the bracket onto M = 1 by pure bisection --
    # returning a plausible near-sonic Mach number for garbage input.
    if math.isnan(phi):
        raise ValueError("flow function is NaN")
    if phi < 0.0:
        raise ValueError(f"flow function must be non-negative, got {phi!r}")
    if phi > phi_max * (1.0 + FEASIBILITY_SLACK):
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
            if hi > MAX_SUPERSONIC_MACH:
                raise InfeasibleOperatingPoint(
                    f"supersonic solution for phi={phi:.6g} lies above the "
                    f"M = {MAX_SUPERSONIC_MACH:g} search limit"
                )
    else:
        lo, hi = 0.0, 1.0

    # Newton with a bisection safeguard, in the `rtsafe` ordering (Numerical
    # Recipes §9.4). The ordering matters: an earlier version shrank the
    # bracket onto the current iterate *before* stepping from it, so once
    # Newton approached the root its own step landed on the bracket boundary,
    # was rejected as out of bounds, and convergence degraded to pure
    # bisection -- 40 iterations at M = 0.5, of which 38 were fallbacks.
    # Here the bracket is updated from the sign at the *new* point instead.
    def residual(M: float) -> float:
        return flow_function(M, gas) - phi

    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    # Orient so that `xl` is the end where the residual is negative. This makes
    # the loop identical on the subsonic branch (Phi increasing) and the
    # supersonic branch (Phi decreasing).
    xl, xh = (lo, hi) if f_lo < 0.0 else (hi, lo)

    M = 0.5 * (lo + hi)
    dx_prev = abs(hi - lo)
    dx = dx_prev
    f = residual(M)
    df = d_flow_function(M, gas)

    for _ in range(maxiter):
        # Bisect when the Newton step would leave the bracket, or when it is
        # not halving the interval fast enough.
        out_of_bracket = ((M - xh) * df - f) * ((M - xl) * df - f) > 0.0
        if out_of_bracket or abs(2.0 * f) > abs(dx_prev * df):
            dx_prev, dx = dx, 0.5 * (xh - xl)
            M = xl + dx
            # Bisection converges when the interval itself underflows -- compare
            # against the bracket end, not the previous iterate. (Comparing to
            # the previous iterate returns the midpoint on the first step,
            # because that is where the iteration starts.)
            if xl == M:
                return M
        else:
            dx_prev, dx = dx, f / df
            M_prev, M = M, M - dx
            if M_prev == M:
                return M
        if abs(dx) <= tol * max(1.0, abs(M)):
            return M

        f = residual(M)
        df = d_flow_function(M, gas)
        if f < 0.0:
            xl = M
        else:
            xh = M

    # Reachable only if maxiter is set below what bisection needs; the
    # safeguard bounds the iteration count, it does not remove this branch.
    raise RuntimeError(f"Mach inversion failed to converge for phi={phi!r} in {maxiter} iterations")


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


def state_from_flux(
    flux: Sequence[float], A: float, gas: PerfectGas, supersonic: bool = False
) -> StaticState:
    """Invert the conservative flux vector — the reverse of the flux function.

    Given ``[rho u A, (rho u^2 + p) A, rho u H A]`` recover the state that
    produces it. Needed to construct the interior profile of a *smeared*
    actuator disk: the steady flux balance ``F_{k+1} = F_k + q_k`` gives the
    face fluxes exactly, and this turns them back into cell states. A seed built
    that way has a residual of ~1e-5 of the source scale through the smear
    region, against ~1 for a sharp two-state profile — which is not a discrete
    steady state at all, because a single cell cannot simultaneously present
    station 1 to its left face and station 2 to its right face.

    With ``m = rho u`` and ``P = rho u^2 + p``, energy conservation gives a
    quadratic in ``u``::

        m (1/2 - cp/R) u^2 + (cp/R) P u - m cp T0 = 0

    whose two positive roots are the subsonic and supersonic states passing the
    same flux. The subsonic one is returned unless ``supersonic`` is set.
    """
    if A <= 0.0:
        raise ValueError(f"area must be positive, got {A!r}")
    m, P, E = flux[0] / A, flux[1] / A, flux[2] / A
    if m <= 0.0:
        raise ValueError(f"mass flux must be positive, got {m!r}")

    a = gas.g_over_gm1  # cp/R
    qa, qb, qc = m * (0.5 - a), a * P, -E
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        raise InfeasibleOperatingPoint(
            f"no state produces this flux: discriminant {disc:.6g} < 0 at "
            f"m={m:.6g}, P={P:.6g}, E={E:.6g}"
        )
    roots = ((-qb + math.sqrt(disc)) / (2.0 * qa), (-qb - math.sqrt(disc)) / (2.0 * qa))

    found = []
    for u in roots:
        if u <= 0.0:
            continue
        p = P - m * u
        if p <= 0.0:
            continue
        rho = m / u
        T = p / (rho * gas.R)
        c = gas.speed_of_sound(T)
        found.append(StaticState(p=p, T=T, rho=rho, u=u, M=u / c, c=c))
    if not found:
        raise InfeasibleOperatingPoint(
            f"no physical state produces this flux (roots {roots}) at m={m:.6g}, P={P:.6g}"
        )
    found.sort(key=lambda s: s.M)
    return found[-1] if supersonic else found[0]


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
    if PR < 1.0:
        raise ValueError(
            f"pressure ratio must be >= 1 for a compressor, got {PR!r}. "
            "Expansion needs the turbine convention tau = 1 - eta*(1 - PR**k), "
            "not this one -- applying it here would silently return the wrong "
            "sign of work"
        )
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
    wall_pressure_integral: float | None = None,
) -> tuple[float, float]:
    """Net axial momentum source and shaft power for an upstream state and flow.

    This is the function Phase 3 calls at runtime, from the *locally measured*
    upstream stagnation state — not a tabulated value (``PLAN.md`` P1).

    .. math::

        F_x = p_{s2}A_2 - p_{s1}A_1 + \\dot m (u_2 - u_1), \\qquad
        \\dot S_W = \\dot m (h_{02} - h_{01})

    **``Fx`` is the total momentum source, not the blade force alone.**
    Integrating the quasi-1D momentum equation
    ``d/dx[(rho u^2 + p)A] = p dA/dx + f_blade`` across the machine gives

    .. math::

        \\Delta\\left[(\\rho u^2 + p)A\\right]
            = \\int p\\,dA + F_\\text{blade}

    so the expression above already contains the wall pressure reaction. The
    two coincide only when ``A1 == A2``, where ``\\int p\\,dA`` vanishes.

    Consequences worth stating plainly:

    * A caller that injects ``Fx`` **must not also add** the geometric
      ``p·dA`` term in the same cell — that double-counts the wall reaction.
    * ``Fx`` is positive at every operating point *for a constant-area
      machine*. With a contraction it is routinely negative (for
      ``A1 = 0.1, A2 = 0.05`` at ``W = 10`` it is −3710 N), because the wall
      term dominates. The claim in ``PLAN.md`` §4.3 that a compressor's ``Fx``
      never crosses zero is a constant-area statement.

    A zero-thickness actuator disk has a single area by definition, so
    ``A1 != A2`` is rejected unless the caller supplies
    ``wall_pressure_integral`` explicitly, making the accounting a decision
    rather than an accident.

    Returns ``(Fx [N], SWx [W])``.
    """
    if wall_pressure_integral is None:
        if A1 != A2:
            raise ValueError(
                f"A1={A1!r} != A2={A2!r}: the returned momentum source would include "
                "the wall pressure reaction, which is not the blade force. Pass "
                "wall_pressure_integral explicitly to state how it is accounted for, "
                "or use equal areas for a zero-thickness actuator disk"
            )
        wall_pressure_integral = 0.0

    T02, p02 = compressor_exit_stagnation(T01, p01, PR, eta, gas)
    st1 = static_from_stagnation(T01, p01, W, A1, gas)
    st2 = static_from_stagnation(T02, p02, W, A2, gas)

    Fx = st2.p * A2 - st1.p * A1 + W * (st2.u - st1.u) - wall_pressure_integral
    SWx = W * (gas.enthalpy(T02) - gas.enthalpy(T01))
    return Fx, SWx


def minimum_back_pressure(
    p01: float, T01: float, PR: float, eta: float, A1: float, A2: float, gas: PerfectGas
) -> float:
    """Lowest back pressure for which a subsonic-matched solution exists.

    Below this the inlet chokes, the mass flow pins at
    :func:`choked_mass_flow`, and the back pressure stops setting the operating
    point. See :class:`InletChokeLimited`.
    """
    T02, p02 = compressor_exit_stagnation(T01, p01, PR, eta, gas)
    W_choke = choked_mass_flow(p01, T01, A1, gas)
    phi2 = W_choke * math.sqrt(gas.R * T02) / (A2 * p02)
    M2 = mach_from_flow_function(phi2, gas)
    return p02 / (1.0 + 0.5 * gas.gm1 * M2 * M2) ** gas.g_over_gm1


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
    if A1 <= 0.0 or A2 <= 0.0:
        raise ValueError(f"areas must be positive, got A1={A1!r}, A2={A2!r}")

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
    if phi1 > max_flow_function(gas) * (1.0 + FEASIBILITY_SLACK):
        W_choke = choked_mass_flow(p01, T01, A1, gas)
        pb_min = minimum_back_pressure(p01, T01, PR, eta, A1, A2, gas)
        raise InletChokeLimited(
            f"back pressure {pb:.6g} Pa demands {W:.6g} kg/s, above the inlet choke "
            f"limit {W_choke:.6g} kg/s. The duct is inlet-choked and the flow is "
            f"pinned there, but the back pressure no longer sets the operating point, "
            f"so the subsonic matching this model solves has no solution. "
            f"Subsonic matching is valid for pb >= {pb_min:.6g} Pa",
            W_choke=W_choke,
            pb_min=pb_min,
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
