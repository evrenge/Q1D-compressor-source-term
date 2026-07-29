"""Quasi-1D Euler solver: Roe flux, MUSCL reconstruction, low-storage RK.

Solves

.. math::

    \\frac{\\partial}{\\partial t}\\begin{pmatrix}\\rho A\\\\ \\rho u A\\\\
    \\rho E A\\end{pmatrix}
    + \\frac{\\partial}{\\partial x}\\begin{pmatrix}\\rho u A\\\\
    (\\rho u^2 + p)A\\\\ \\rho u H A\\end{pmatrix}
    = \\begin{pmatrix}0\\\\ p\\,dA/dx\\\\ 0\\end{pmatrix} + q_\\text{source}

A cleaned, vectorised rewrite of the frozen ``legacy/q1d_solver.py``. The
numerics are the same scheme; what changed is listed in ``BASELINE.md`` and
``PLAN.md`` Phase 2. The externally visible differences that matter:

* ``ReferenceState`` is a required argument. The legacy limiter thresholds were
  built from the *initial condition*, where ``u = 1.28e-4 m/s``, so the velocity
  limiter never disengaged (``PLAN.md`` §4.2 #7).
* Supersonic boundary branches exist rather than falling through to the
  subsonic characteristic form.
* A residual norm is available, so runs stop on convergence instead of on a
  wall-clock guess.
* Non-physical states raise instead of propagating NaN.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .boundary import BoundaryCondition, apply_boundaries
from .gas import PerfectGas
from .grid import Grid

__all__ = [
    "SolverConfig",
    "ReferenceState",
    "RunResult",
    "Solver",
    "NonPhysicalState",
    "CFL_STABILITY_LIMIT_ORDER1",
    "CFL_STABILITY_LIMIT_ORDER2",
]

#: Jameson-style 5-stage low-storage coefficients, as used by the legacy code.
#: NOTE: these are Blazek's *hybrid* set, tuned for dissipation evaluated at
#: stages 1/3/5 only. This solver, like the legacy one, evaluates dissipation
#: at every stage, so the stability limit is not the one they were tuned for.
#: `tests/test_solver.py::test_measured_cfl_stability_limit` measures it.
RK5 = (0.0695, 0.1602, 0.2898, 0.5060, 1.0)


class NonPhysicalState(RuntimeError):
    """Density or pressure went non-positive, or a NaN appeared."""


@dataclass(frozen=True)
class ReferenceState:
    """Scales for the MUSCL limiter thresholds.

    ``rho``, ``u`` and ``p`` must describe the *operating point*, not the
    initial condition. The van Albada epsilon is a smoothness threshold: it
    should sit near the square of a typical smooth-region variation so the
    limiter disengages there and engages at discontinuities. The legacy code
    took ``uref`` from an initial condition where ``u = 1.28e-4 m/s``, a factor
    1.4e6 below the operating velocity, so the velocity limiter never
    disengaged (``PLAN.md`` §4.2 #7).

    ``vol`` is a **normalising volume, not a representative cell volume**. The
    threshold is ``limfac^3 * ref^2 * (vol_cell / vol)^1.5``, and that
    ``vol_cell^1.5`` factor is the Venkatakrishnan/van Albada smoothness
    parameter ``eps^2 ~ (K*dx)^3``: it must shrink with mesh size so the
    limiter engages at discontinuities and relaxes in smooth regions. Setting
    ``vol`` to a typical cell volume defeats that — it scales the thresholds up
    by ~3e4 on a 100-cell grid and switches the limiter *off* across shocks,
    which produces negative pressures on a shock tube. Leave it at 1.0 unless
    you have measured a reason not to.
    """

    rho: float
    u: float
    p: float
    vol: float = 1.0

    def __post_init__(self) -> None:
        for name in ("rho", "u", "p", "vol"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"reference {name} must be positive, got {getattr(self, name)!r}")


#: Conservative stability bound for **order 2**. The limit is not a single
#: number: measured on a stagnant varying-area duct whose residual starts at
#: bitwise zero (so growth is pure amplification), it *falls with mesh
#: refinement* -- 2.475 at n=40, 2.452 at n=80, 2.440 at n=160, 2.435 at n=320
#: -- and a short probe hides it, since CFL 2.44 looks stable at n=160 for 400
#: steps and diverges by 5000. Order 1 is far more forgiving at ~3.13.
#:
#: An earlier constant of 2.45 was measured at (n=80, 400 steps) and presented
#: as a property of the scheme. It is not; treat this as an upper bound that
#: still needs margin, not a usable CFL.
CFL_STABILITY_LIMIT_ORDER2 = 2.43

#: Order-1 reconstruction is stable to roughly this, measured the same way.
CFL_STABILITY_LIMIT_ORDER1 = 3.12


def _has_constant_gamma(gas) -> bool:
    """Whether the closed-form perfect-gas algebra applies. See analytic.py."""
    return hasattr(gas, "gamma")


def _temperature_from_enthalpy_arr(gas, h, hint=None):
    """Vectorised inverse of ``h(T)``, warm-started from the last call."""
    ha = np.asarray(h, dtype=float)
    T = np.full_like(ha, 300.0)
    if hint is not None and np.shape(hint) == ha.shape:
        T = np.array(hint, dtype=float)
    for _ in range(80):
        step = (gas.enthalpy(T) - ha) / gas.cp_at(T)
        T = np.maximum(T - step, 1.0)
        if np.all(np.abs(step) < 1e-11 * np.maximum(T, 1.0)):
            break
    return T


def _pressure_real(gas, cv: np.ndarray, a: np.ndarray, hint=None):
    """``(p, T)`` from the conserved variables when ``cp`` is not constant.

    ``e = E/rho - u^2/2`` is internal energy; for any gas ``e = h(T) - R*T``, so
    ``T`` follows by inverting that and ``p = rho*R*T``. The inversion is a
    vectorised Newton, warm-started from the previous step's temperature when
    one is available -- during a converging run that start is within a few
    kelvin and the Newton takes two or three iterations rather than twenty.
    ``T`` comes back so the caller can keep it for that warm start; throwing it
    away and re-deriving ``p/(rho*R)`` costs nothing but leaves the next Newton
    cold.
    """
    rho = cv[0] / a
    u = cv[1] / cv[0]
    e = cv[2] / (rho * a) - 0.5 * u * u
    T = _temperature_from_internal(gas, e, hint)
    return rho * gas.R * T, T


def _temperature_from_internal(gas, e: np.ndarray, hint=None) -> np.ndarray:
    """Solve ``h(T) - R*T = e`` for ``T``. Derivative is ``cp(T) - R = cv(T)``."""
    T = np.full_like(np.asarray(e, dtype=float), 300.0) if hint is None else np.array(hint, float)
    if T.shape != np.shape(e):
        T = np.full_like(np.asarray(e, dtype=float), 300.0)
    for _ in range(80):
        f = gas.enthalpy(T) - gas.R * T - e
        step = f / (gas.cp_at(T) - gas.R)
        T = np.maximum(T - step, 1.0)
        if np.all(np.abs(step) < 1e-11 * np.maximum(T, 1.0)):
            break
    return T


@dataclass(frozen=True)
class SolverConfig:
    #: Default carries ~20% margin below the measured limit.
    cfl: float = 2.0
    order: int = 2
    limiter_factor: float = 1.5
    entropy_fix: float = 0.05
    rk_alpha: tuple[float, ...] = RK5
    local_time_stepping: bool = False


@dataclass
class RunResult:
    steps: int
    t: float
    converged: bool
    wall_time: float
    residual_history: np.ndarray = field(repr=False)
    time_history: np.ndarray = field(repr=False)

    @property
    def final_residual(self) -> np.ndarray:
        return self.residual_history[-1]


def _harten_entropy_fix(z: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Vectorised Harten correction.

    The legacy version was a Python loop over ~100 elements called three times
    per stage; this is the same formula, 4.7x cheaper (``PLAN.md`` §3.4).
    """
    return np.where(z > delta, z, 0.5 * (z * z + delta * delta) / np.maximum(delta, 1e-30))


def _van_albada(a: np.ndarray, b: np.ndarray, eps: np.ndarray) -> np.ndarray:
    return (a * (b * b + eps) + b * (a * a + eps)) / (a * a + b * b + 2.0 * eps + 1e-30)


class Solver:
    def __init__(
        self,
        grid: Grid,
        gas: PerfectGas,
        bc: BoundaryCondition,
        reference: ReferenceState,
        config: SolverConfig | None = None,
        source: Callable[[Solver], np.ndarray] | None = None,
        low_order_faces: np.ndarray | None = None,
    ) -> None:
        self.grid = grid
        self.gas = gas
        self.bc = bc
        self.reference = reference
        self.config = config or SolverConfig()
        self.source = source

        # Faces where MUSCL falls back to first order. A source term applied to
        # cell averages with no matching treatment in the reconstruction is not
        # well balanced: inside the smeared region MUSCL reads a *source*-imposed
        # profile as a solution gradient and reconstructs across it. The
        # geometric source p*dA/dx avoids this because it is written to telescope
        # against the pressure flux exactly; the disk source has no such form.
        # Measured (PLAN.md 3.13): at PR 2.0 first order holds the operating
        # point to 2.5e-14 while second order misses by 4e-4 to 4e-2 at every
        # smear width, and widening the smear makes it worse rather than better.
        # `ActuatorDisk.low_order_faces` builds the mask.
        if low_order_faces is None:
            self.low_order_faces = None
        else:
            mask = np.asarray(low_order_faces, dtype=bool)
            if mask.shape != (grid.n_interior + 1,):
                raise ValueError(
                    f"low_order_faces has shape {mask.shape}, expected "
                    f"{(grid.n_interior + 1,)} -- one entry per face"
                )
            self.low_order_faces = mask
        #: Face fluxes from the last `residual` call, for `mass_flux_at`.
        self._face_flux: np.ndarray | None = None

        n = grid.n_interior
        self._roe_T_hint = None
        self._T_hint = None
        self.cv = np.zeros((3, n + 2))
        self.p = np.zeros(n + 2)
        self.t = 0.0
        self.dt_last = 0.0

        # preallocated work arrays -- at ~100 cells, numpy per-call overhead
        # dominates the arithmetic, so avoiding fresh allocations matters
        self._du = np.zeros((3, n + 3))
        self._cvold = np.zeros((3, n + 2))
        self._prim = np.zeros((3, n + 2))

        # van Albada thresholds: limfac^3 * ref^2 * (vol/volref)^1.5
        lim3 = self.config.limiter_factor**3
        scale = (0.5 * (grid.vol[1:] + grid.vol[:-1]) / reference.vol) ** 1.5
        self._eps2 = np.array(
            [
                lim3 * reference.rho**2 * scale,
                lim3 * reference.u**2 * scale,
                lim3 * reference.p**2 * scale,
            ]
        )

    # -- state -------------------------------------------------------------

    def set_state(self, rho, u, p) -> None:
        """Initialise the interior from primitive variables (scalars or arrays)."""
        n = self.grid.n_interior
        rho = np.broadcast_to(np.asarray(rho, dtype=float), (n,))
        u = np.broadcast_to(np.asarray(u, dtype=float), (n,))
        p = np.broadcast_to(np.asarray(p, dtype=float), (n,))
        a = self.grid.a_cell[1:-1]

        self.cv[0, 1:-1] = rho * a
        self.cv[1, 1:-1] = rho * u * a
        # Total energy. For a calorically perfect gas the internal energy is
        # p/(gamma-1); in general it is rho*(h(T) - R*T), which is the same thing
        # when h = cp*T. Writing it the general way costs one enthalpy call at
        # setup and lets the same solver carry a NASA9 gas (`PLAN.md` §3.45).
        if _has_constant_gamma(self.gas):
            self.cv[2, 1:-1] = (p / self.gas.gm1 + 0.5 * rho * u * u) * a
        else:
            T = p / (rho * self.gas.R)
            self.cv[2, 1:-1] = rho * (self.gas.enthalpy(T) - self.gas.R * T
                                      + 0.5 * u * u) * a
        # Derive p from cv rather than storing the argument, so the state is
        # exactly what an RK stage would produce. Storing p directly left it
        # 1 ulp inconsistent with cv, which made the well-balancedness gate
        # measure a state the time-stepper never visits.
        #
        # Interior only: the ghost cells are still empty at this point, so the
        # whole-array `_update_pressure` would divide by a zero ghost density.
        # `_sync_boundaries` fills them from the interior immediately after.
        cv = self.cv
        if _has_constant_gamma(self.gas):
            self.p[1:-1] = (
                self.gas.gm1 / a * (cv[2, 1:-1] - 0.5 * cv[1, 1:-1] * cv[1, 1:-1] / cv[0, 1:-1])
            )
        else:
            # No hint: the interior slice has a different shape from the whole
            # array the hint is kept at, and at setup there is nothing to hint
            # with anyway.
            self.p[1:-1], _ = _pressure_real(self.gas, cv[:, 1:-1], a)
        self._sync_boundaries()

    def primitives(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(rho, u, p, c)`` over all cells including ghosts."""
        rho = self.cv[0] / self.grid.a_cell
        u = self.cv[1] / self.cv[0]
        if _has_constant_gamma(self.gas):
            return rho, u, self.p, np.sqrt(self.gas.gamma * self.p / rho)
        T = self.p / (rho * self.gas.R)
        return rho, u, self.p, np.sqrt(self.gas.gamma_at(T) * self.gas.R * T)

    def _update_pressure(self) -> None:
        cv, a = self.cv, self.grid.a_cell
        if _has_constant_gamma(self.gas):
            np.multiply(
                self.gas.gm1 / a,
                cv[2] - 0.5 * cv[1] * cv[1] / cv[0],
                out=self.p,
            )
        else:
            self.p[:], self._T_hint = _pressure_real(self.gas, cv, a, self._T_hint)
        # min() returns NaN if any element is NaN, so `not (m > 0)` catches
        # non-positive and non-finite in one reduction per array.
        if not (self.p.min() > 0.0 and cv[0].min() > 0.0):
            bad = int(np.argmin(np.where(np.isfinite(self.p), self.p, -np.inf)))
            raise NonPhysicalState(
                f"non-physical state at step t={self.t:.6g}: cell {bad} has "
                f"p={self.p[bad]!r}, rho*A={cv[0, bad]!r}"
            )

    def _sync_boundaries(self) -> None:
        apply_boundaries(self.cv, self.p, self.grid, self.gas, self.bc)

    # -- spatial discretisation --------------------------------------------

    def _reconstruct(self, rho, u):
        """Left/right face states, each ``(3, n+1)`` as ``[rho, u, p]``.

        Operates on all three variables as one stacked array rather than
        looping over them. At ~100 cells the numpy per-call overhead exceeds
        the arithmetic, so collapsing six ``_van_albada`` calls per stage into
        two is worth more than any change to the expressions themselves.
        """
        prim = self._prim
        prim[0] = rho
        prim[1] = u
        prim[2] = self.p
        if self.config.order == 1:
            return prim[:, :-1], prim[:, 1:]

        du = self._du
        np.subtract(prim[:, 1:], prim[:, :-1], out=du[:, 1:-1])
        du[:, 0] = du[:, 1]
        du[:, -1] = du[:, -2]

        eps = self._eps2
        dr = _van_albada(du[:, 2:], du[:, 1:-1], eps)
        dl = _van_albada(du[:, 1:-1], du[:, :-2], eps)
        if self.low_order_faces is not None:
            dr = np.where(self.low_order_faces, 0.0, dr)
            dl = np.where(self.low_order_faces, 0.0, dl)
        return prim[:, :-1] + 0.5 * dl, prim[:, 1:] - 0.5 * dr

    def _roe_flux(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        gas = self.gas
        rl, ul, pl = left
        rr, ur, pr = right
        perfect = _has_constant_gamma(gas)
        if perfect:
            hl = gas.g_over_gm1 * pl / rl + 0.5 * ul * ul
            hr = gas.g_over_gm1 * pr / rr + 0.5 * ur * ur
        else:
            # Total enthalpy from the polynomials. `cp/(gamma-1) * p/rho` is
            # `cp*T`, which is only `h` when `cp` is constant.
            hl = gas.enthalpy(pl / (rl * gas.R)) + 0.5 * ul * ul
            hr = gas.enthalpy(pr / (rr * gas.R)) + 0.5 * ur * ur
        qrl, qrr = ul * rl, ur * rr

        central = np.array(
            [
                qrl + qrr,
                qrl * ul + qrr * ur + pl + pr,
                qrl * hl + qrr * hr,
            ]
        )

        rav = np.sqrt(rl * rr)
        dd = rav / rl
        dd1 = 1.0 / (1.0 + dd)
        uav = (ul + dd * ur) * dd1
        hav = (hl + dd * hr) * dd1
        q2a = 0.5 * uav * uav
        if perfect:
            c2a = gas.gm1 * (hav - q2a)
            # Energy component of the entropy-wave eigenvector. See below.
            e2a = q2a
        else:
            # Real-gas Roe with the eigenstructure of a general p(rho, e), not
            # the constant-gamma one with gamma swapped for a local value. Two
            # things change and BOTH are needed; carrying only the first is what
            # blew a compressor up at step 96 with an upstream-running wave.
            #
            # `hav - q2a` is the averaged STATIC enthalpy, so it inverts for the
            # averaged temperature directly.
            Tav = _temperature_from_enthalpy_arr(gas, hav - q2a, self._roe_T_hint)
            self._roe_T_hint = Tav
            # (1) c^2 = gamma*R*T, NOT (gamma-1)*h. The two agree only when
            # h = cp*T. A NASA9 enthalpy carries the enthalpy of formation, so h
            # is NEGATIVE below about 300 K on air (-10,111 J/kg at 288 K) and
            # (gamma-1)*h would be a negative "c^2" -- the square root of which
            # is how this first appeared: NaN in cell 1 on the very first step.
            c2a = gas.gamma_at(Tav) * gas.R * Tav
            # (2) The entropy wave's energy eigenvector is `H - c^2/kappa` with
            # kappa = p_e/rho = R/cv, not `u^2/2`. The familiar `u^2/2` is what
            # that expression *collapses to* when h = cp*T: c^2/kappa = cp*T = h,
            # so H - h = u^2/2. Off that datum it does not collapse, and the gap
            # is not small -- on air at 288 K, h - cp*T is -299 kJ/kg against a
            # u^2/2 of about 11 kJ/kg, so the entropy wave's contribution to the
            # energy flux comes out with the wrong sign and 27x the magnitude.
            # The upwinding then feeds the density-error mode instead of damping
            # it, which is exactly the growing wave that was observed.
            e2a = hav - gas.cp_at(Tav) * Tav
        cav = np.sqrt(c2a)

        # One stacked Harten correction rather than three separate calls.
        delta = self.config.entropy_fix * cav
        e1, e2, e3 = _harten_entropy_fix(np.abs(np.stack((uav - cav, uav, uav + cav))), delta)

        dp = pr - pl
        h1 = rav * cav * (ur - ul)
        a2 = e1 * (dp - h1) / (2.0 * c2a)
        a3 = e2 * ((rr - rl) - dp / c2a)
        a5 = e3 * (dp + h1) / (2.0 * c2a)

        dissipation = np.array(
            [
                a2 + a3 + a5,
                a2 * (uav - cav) + a3 * uav + a5 * (uav + cav),
                a2 * (hav - cav * uav) + a3 * e2a + a5 * (hav + cav * uav),
            ]
        )
        return 0.5 * (central - dissipation) * self.grid.a_face

    def face_fluxes(self) -> np.ndarray:
        """Numerical fluxes at the ``n+1`` faces, shape ``(3, n+1)``.

        ``face_fluxes()[0]`` is the mass flux, which is the quantity the scheme
        actually conserves. At steady state it is uniform to round-off, whereas
        cell-centred ``rho*u*A(x_centre)`` is only uniform to ``O(dx^2)``
        wherever the area has curvature.
        """
        rho, u, _, _ = self.primitives()
        return self._roe_flux(*self._reconstruct(rho, u))

    def mass_flux_at(self, cell: int) -> float:
        """Mass flow through an interior cell, from the conserved numerical flux.

        The face flux is what the scheme conserves. Forming
        ``rho * u * A(x_centre)`` instead is uniform only to ``O(dx^2)`` where
        the area has curvature (see :meth:`face_fluxes`), and that biases anything
        keyed on it: on a twelve-stage tapered machine two stations disagreed by
        6e−05 with no physical error present at all (``PLAN.md`` §3.21). Since a
        compressor closure keys its *map lookup* on the sampled mass flow, such a
        bias moves the operating point, not merely a diagnostic.

        **This does not assume the mass flux is uniform.** A source that adds or
        removes mass — interstage bleed, turbine cooling air — makes the flux
        step by exactly what it takes out, and that is correct behaviour rather
        than an error. What survives is the property actually being used: the
        face flux is the conserved quantity, so it remains the right measure of
        what passes a station whatever sources exist upstream of it. Two
        consequences worth stating before bleed arrives:

        * **The cell's upstream face is used, not an average of its two faces.**
          At steady state with no mass source in the cell the two are equal, so
          this matters only to transients and to how close a station may sit to a
          disk — and there it matters a lot. Averaging reaches half a cell
          *towards* the disk and picks up its perturbation: with the average, a
          station one cell upstream of a disk read a flow function 0.03% over the
          sonic maximum during startup and raised. The upstream face is half a
          cell further away than even the cell centre, so it is the safest of the
          three readings. With a mass source *inside* the sampling cell it reads
          the flow arriving at that cell, which is the well-defined thing for a
          station placed there.
        * The *spread* of mass flux along the duct stops being a convergence
          measure once bleed exists, because it is then legitimately non-zero.
          :meth:`residual_norm` is the source-agnostic measure and should be
          preferred for that job.

        Uses the fluxes cached by the last :meth:`residual` call when there is
        one — which there is whenever a source term asks, since ``residual``
        computes them before invoking the source — and recomputes otherwise, so
        calling this from outside the solve is correct but not free.
        """
        f = self._face_flux
        if f is None:
            f = self.face_fluxes()
        return float(f[0, cell])

    def station_state_at(self, cell: int):
        """Static state at an interior cell's upstream face, from the fluxes.

        The companion to :meth:`mass_flux_at` for the rest of the state. The
        conserved fluxes are inverted back to ``(rho, u, p)`` by
        :func:`analytic.state_from_flux`, giving a station reading consistent
        with what the scheme transports rather than with a cell average of it.

        **This is what removes the "numerical boundary layer".** Phase 3 measured
        the disk contaminating the field upstream of itself, decaying ~10x every
        two to three cells, and chose ``sample_offset = 12`` to clear it. That
        contamination is not in the field — it is the error of averaging a sharp
        profile over a cell. Read from the fluxes, the stagnation state is exact
        one cell from the disk. Measured on a converged single-disk case, ``p01``
        error against the known inlet value:

        ==========  ================  ==============
        ``offset``  cell-centred      from the flux
        ==========  ================  ==============
        1           −1.933e−04        −8.9e−12
        3           −1.074e−05        −9.1e−12
        12          +3.6e−11          −9.6e−12
        ==========  ================  ==============

        Flat at every offset, and it holds at PR 1.2/1.6/2.0 with and without a
        downstream taper. The practical consequence is roughly **eleven cells per
        blade row** returned to the mesh budget, which matters once an engine has
        twenty of them (``PLAN.md`` §3.23).

        Falls back to the cell-centred reading when the flux inversion has no
        solution. Not defensive padding: during a startup transient an
        intermediate face flux can correspond to *no* physical state, and the
        inversion raises on a negative discriminant where the cell-centred
        product simply returns a number. The fallback costs nothing at
        convergence — where the flux reading is the one used, and is exact — and
        keeps a run alive through the transient that would otherwise hit it.
        """
        from .analytic import InfeasibleOperatingPoint, StaticState, state_from_flux

        f = self._face_flux
        if f is None:
            f = self.face_fluxes()
        try:
            return state_from_flux(f[:, cell], float(self.grid.a_face[cell]), self.gas)
        except InfeasibleOperatingPoint:
            rho, u, p, c = self.primitives()
            i = cell + 1  # primitives carry ghost cells
            r, uu, pp, cc = float(rho[i]), float(u[i]), float(p[i]), float(c[i])
            return StaticState(
                p=pp, T=pp / (r * self.gas.R), rho=r, u=uu, M=uu / cc, c=cc
            )

    def residual(self) -> np.ndarray:
        """``d(flux)/dx - sources`` over the interior cells, shape ``(3, n)``.

        Sign convention matches the update ``cv -= dt/dx * alpha * residual``.
        """
        rho, u, _, _ = self.primitives()
        f = self._roe_flux(*self._reconstruct(rho, u))
        # Cached before the source is called, so a source term can read the
        # conserved mass flux for free instead of forming rho*u*A itself. See
        # `mass_flux_at`.
        self._face_flux = f
        rhs = f[:, 1:] - f[:, :-1]

        # Geometric source p*dA, written as the *difference of the same two
        # products* that appear in the pressure flux rather than as p*da.
        # Floating-point multiplication does not distribute, so
        # `p*A_r - p*A_l` and `p*(A_r - A_l)` differ by ~1 ulp; using the
        # former makes a uniform stagnant field well-balanced to bitwise zero
        # instead of to round-off.
        pi = self.p[1:-1]
        rhs[1] -= pi * self.grid.a_face[1:] - pi * self.grid.a_face[:-1]

        if self.source is not None:
            rhs -= self.source(self)
        return rhs

    def residual_norm(self) -> np.ndarray:
        """RMS of ``d(cv)/dt`` per equation — the convergence measure."""
        return np.sqrt(np.mean((self.residual() / self.grid.dx) ** 2, axis=1))

    # -- time integration ---------------------------------------------------

    def timestep(self) -> np.ndarray | float:
        rho, u, _, c = self.primitives()
        lam = np.abs(u) + c
        dt_local = self.config.cfl * self.grid.dx / lam[1:-1]
        return dt_local if self.config.local_time_stepping else float(dt_local.min())

    def advance(self, dt: np.ndarray | float) -> None:
        """One time step.

        ``advance`` owns the clock. It used to be ``run`` that advanced ``t``,
        which meant a caller stepping the solver directly — every stability
        study in this project does — kept ``t = 0`` forever, and the
        ``NonPhysicalState`` message reported ``t=0`` for a failure thousands of
        steps in. Sources that carry their own state need the step size, so
        ``dt_last`` is recorded here too; with local time stepping it is the
        smallest cell's step, which is the only one that bounds a global state.
        """
        cvold = self._cvold
        np.copyto(cvold, self.cv)
        scale = np.asarray(dt) / self.grid.dx
        for alpha in self.config.rk_alpha:
            rhs = self.residual()
            self.cv[:, 1:-1] = cvold[:, 1:-1] - (scale * alpha) * rhs
            self._update_pressure()
            self._sync_boundaries()
        self.dt_last = float(np.min(dt))
        self.t += self.dt_last

    def run(
        self,
        *,
        max_steps: int = 100_000,
        tol: float | None = None,
        t_end: float | None = None,
        record_every: int = 10,
    ) -> RunResult:
        """March until converged (``tol``), until ``t_end``, or out of steps.

        ``tol`` is on the residual norm *relative to its value at step zero*, so
        it is independent of the problem's absolute scale. With neither ``tol``
        nor ``t_end`` the run is exactly ``max_steps`` steps long.
        """
        # With neither `tol` nor `t_end`, the run is exactly `max_steps` long --
        # which is what a hold test wants.
        if t_end is not None and self.config.local_time_stepping:
            raise ValueError(
                "local_time_stepping advances each cell at its own rate, so the field "
                "is not a solution at any single time and t_end is meaningless. Use it "
                "with tol for steady problems, or switch to global stepping."
            )
        if record_every < 1:
            raise ValueError(f"record_every must be >= 1, got {record_every!r}")

        start = time.perf_counter()
        residuals: list[np.ndarray] = []
        times: list[float] = []
        r0 = self.residual_norm()
        if tol is not None and not np.any(r0 > 0.0):
            # Every equation is already at bitwise zero -- a stagnant duct, say.
            # Normalising by 1.0 here would silently turn a relative tolerance
            # into an absolute one in SI units, which no run can then satisfy.
            return RunResult(
                steps=0,
                t=self.t,
                converged=True,
                wall_time=time.perf_counter() - start,
                residual_history=np.array([r0]),
                time_history=np.array([self.t]),
            )
        r0 = np.where(r0 > 0.0, r0, np.max(r0))

        converged = False
        step = 0
        while step < max_steps:
            dt = self.timestep()
            if t_end is not None:
                remaining = t_end - self.t
                if remaining <= 0.0:
                    break
                dt = np.minimum(dt, remaining)  # clip so t_end is hit exactly

            self.advance(dt)
            step += 1

            if step % record_every == 0 or step == 1:  # residual_norm is ~27% of a step
                r = self.residual_norm()
                residuals.append(r)
                times.append(self.t)
                if tol is not None and np.all(r / r0 < tol):
                    converged = True
                    break

        return RunResult(
            steps=step,
            t=self.t,
            converged=converged,
            wall_time=time.perf_counter() - start,
            residual_history=np.array(residuals) if residuals else np.array([r0]),
            time_history=np.array(times) if times else np.array([self.t]),
        )
