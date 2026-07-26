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
    ) -> None:
        self.grid = grid
        self.gas = gas
        self.bc = bc
        self.reference = reference
        self.config = config or SolverConfig()
        self.source = source

        n = grid.n_interior
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
        self.cv[2, 1:-1] = (p / self.gas.gm1 + 0.5 * rho * u * u) * a
        # Derive p from cv rather than storing the argument, so the state is
        # exactly what an RK stage would produce. Storing p directly left it
        # 1 ulp inconsistent with cv, which made the well-balancedness gate
        # measure a state the time-stepper never visits.
        #
        # Interior only: the ghost cells are still empty at this point, so the
        # whole-array `_update_pressure` would divide by a zero ghost density.
        # `_sync_boundaries` fills them from the interior immediately after.
        cv = self.cv
        self.p[1:-1] = (
            self.gas.gm1 / a * (cv[2, 1:-1] - 0.5 * cv[1, 1:-1] * cv[1, 1:-1] / cv[0, 1:-1])
        )
        self._sync_boundaries()

    def primitives(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(rho, u, p, c)`` over all cells including ghosts."""
        rho = self.cv[0] / self.grid.a_cell
        u = self.cv[1] / self.cv[0]
        return rho, u, self.p, np.sqrt(self.gas.gamma * self.p / rho)

    def _update_pressure(self) -> None:
        cv, a = self.cv, self.grid.a_cell
        np.multiply(
            self.gas.gm1 / a,
            cv[2] - 0.5 * cv[1] * cv[1] / cv[0],
            out=self.p,
        )
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
        return prim[:, :-1] + 0.5 * dl, prim[:, 1:] - 0.5 * dr

    def _roe_flux(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        gas = self.gas
        rl, ul, pl = left
        rr, ur, pr = right
        hl = gas.g_over_gm1 * pl / rl + 0.5 * ul * ul
        hr = gas.g_over_gm1 * pr / rr + 0.5 * ur * ur
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
        c2a = gas.gm1 * (hav - q2a)
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
                a2 * (hav - cav * uav) + a3 * q2a + a5 * (hav + cav * uav),
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

    def residual(self) -> np.ndarray:
        """``d(flux)/dx - sources`` over the interior cells, shape ``(3, n)``.

        Sign convention matches the update ``cv -= dt/dx * alpha * residual``.
        """
        rho, u, _, _ = self.primitives()
        f = self._roe_flux(*self._reconstruct(rho, u))
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
