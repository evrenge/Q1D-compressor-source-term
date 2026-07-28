"""Implicit time integration for the quasi-1D solver.

**Why.** The explicit scheme evaluates the source from the *previous* state. With
a compressor disk that is a stiffly coupled equilibrium: at PR 4.4 on ``HPC01``
the ``W ↔ PR`` loop grows exponentially, doubling every ~100 steps, entirely
inside the tabulated map range (``PLAN.md`` §3.15). Every parametric remedy
failed — spreading the source, first-order reconstruction, the map's own
stabilising slope, a choked outlet, a contracting duct, pinning the inlet, and
damping, whose required time constant turned out **non-monotonic**.

Solving for the state and the source *together* removes the gain condition
instead of shrinking it. That is what 1D engine codes do with iterative
component matching.

**What implicit integration does and does not buy.** It does not rescue an
equilibrium that is itself unstable: linearising about the designed steady state
on ``HPC01`` gives ``max Re(λ) > 0`` from PR 2.26 up, converged under mesh
refinement, so the fixed point is a repeller and every A-stable scheme in
existence will still walk away from it (``PLAN.md`` §3.16). What it does buy is
freedom from the explicit stability limit on the *stable* range — steps far
beyond the CFL bound, and a residual that can be driven to round-off in tens of
steps instead of tens of thousands.

**Jacobian-free.** The disk's source is **non-local**: the cells carrying it
depend on a sampling station a dozen cells upstream, so the Jacobian is not
block-tridiagonal and the usual banded solve cannot represent the very coupling
that needs to be implicit. A Krylov method needs only ``J·v``, approximated by a
directional difference of the residual, which captures the off-band term exactly
and costs one residual evaluation.

**Source state during a Newton solve.** Sources that carry their own state
(``compressor.InletFilter``) advance on ``solver.t``, which does not move during
the iterations, so they hold still and the Newton problem is well posed. They
then advance once when the step is accepted — an IMEX treatment of the filter,
which is what it is for: a slow physical lag, not part of the stiff coupling.
The same holds across Runge-Kutta stages: all stages of one step see one filter
state, so the filter runs at the step rate rather than the stage rate.

**The three schemes.** Every one of them reduces to the same nonlinear solve,

.. math::

    G(x) = x - x_\\text{ref} + c\\,h\\,R(x)/\\Delta x = 0,

and differs only in what fills ``x_ref`` and ``c``:

===========  ======  =============================================  ============
scheme       order   ``x_ref``                                      ``c``
===========  ======  =============================================  ============
``euler``    1       ``x_n``                                        1
``bdf2``     2       ``α₀x_n + α₁x_{n−1}``                          ``β h``
``sdirk3``   3       ``x_n + h Σ_{j<i} a_ij f_j`` (per stage)       ``γ``
===========  ======  =============================================  ============

``bdf2`` carries the variable-step coefficients, because a CFL-driven ``dt``
does not repeat, and starts itself with one backward-Euler step. ``sdirk3`` is
Alexander's three-stage, L-stable, stiffly accurate scheme, so its last stage
*is* the new state and no separate update is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["ImplicitConfig", "ImplicitStepper", "NewtonDidNotConverge", "SCHEMES"]

#: Alexander's 3-stage SDIRK: the root of ``γ³ − 3γ² + 3γ/2 − 1/6`` in
#: ``(1/6, 1/2)``, which is what makes the scheme third order and L-stable.
SDIRK3_GAMMA = 0.43586652150845899941601945119356

SCHEMES = ("euler", "bdf2", "sdirk3")


class NewtonDidNotConverge(RuntimeError):
    """The nonlinear solve failed to reach tolerance within its iteration cap."""


def _sdirk3_tableau() -> tuple[np.ndarray, np.ndarray]:
    """Lower-triangular ``A`` and abscissae ``c`` for Alexander's SDIRK3.

    Stiffly accurate: the last row of ``A`` equals the weights ``b``, so
    ``x_{n+1} = X_3`` and the stage solve *is* the update. That also makes the
    scheme L-stable, which matters here — the fast acoustic modes are two
    decades stiffer than anything being resolved and must be annihilated, not
    merely damped.
    """
    g = SDIRK3_GAMMA
    b1 = -(6.0 * g * g - 16.0 * g + 1.0) / 4.0
    b2 = (6.0 * g * g - 20.0 * g + 5.0) / 4.0
    A = np.array(
        [
            [g, 0.0, 0.0],
            [(1.0 - g) / 2.0, g, 0.0],
            [b1, b2, g],
        ]
    )
    return A, A.sum(axis=1)


@dataclass
class ImplicitConfig:
    #: Relative drop in the nonlinear residual that counts as converged. Not
    #: tighter than the residual is smooth: the van Albada limiter and Harten's
    #: entropy fix make ``R`` only piecewise differentiable, so Newton stalls
    #: below roughly 1e-9 no matter how good the linear solve is.
    newton_tol: float = 1e-8
    #: Absolute floor on the same test, as a fraction of the state norm. Without
    #: it a purely relative test is unsatisfiable exactly when there is nothing
    #: to do: starting from a converged field the initial defect is already at
    #: round-off, and demanding a further 1e-8 of that fails every step.
    newton_atol: float = 1e-13
    max_newton: int = 25
    #: Krylov tolerance, relative to the current nonlinear residual. Loose is
    #: fine and cheaper: an inexact Newton step still converges quadratically
    #: near the root as long as the forcing term shrinks with the residual.
    krylov_tol: float = 1e-3
    max_krylov: int = 200
    #: Directional-difference step for ``J·v``, scaled by the state norm.
    jfnk_eps: float = 1e-7
    #: Halve the step and retry this many times if Newton fails or the state
    #: goes non-physical. Zero disables, and the failure propagates.
    max_step_cuts: int = 6


@dataclass
class ImplicitStepper:
    """Implicit time integration of ``dU/dt = -R(U)/dx`` by Newton-Krylov.

    ``R`` is the solver's own residual, so the *spatial* discretisation is
    unchanged — same Roe flux, same limiter, same source. Only the time
    integration differs, which is what isolates whether a blockage is the
    explicit treatment of the coupling.

    ``scheme`` is one of :data:`SCHEMES`.
    """

    solver: object
    scheme: str = "euler"
    config: ImplicitConfig = field(default_factory=ImplicitConfig)

    #: Diagnostics from the last accepted step.
    newton_iterations: int = field(init=False, default=0)
    krylov_iterations: int = field(init=False, default=0)
    step_cuts: int = field(init=False, default=0)
    #: Nonlinear solves performed in the last step: 1 for the linear multistep
    #: schemes, 3 for SDIRK3.
    stages: int = field(init=False, default=0)

    _x_prev: np.ndarray | None = field(init=False, repr=False, default=None)
    _dt_prev: float = field(init=False, repr=False, default=math.nan)

    def __post_init__(self) -> None:
        if self.scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {self.scheme!r}; expected one of {SCHEMES}")
        self._A, self._c = _sdirk3_tableau()

    # -- state plumbing -----------------------------------------------------

    def reset_history(self) -> None:
        """Forget the multistep history, so the next BDF2 step self-starts.

        Call after changing the state behind the stepper's back; a BDF2 step
        built on a stale ``x_{n-1}`` is not merely inaccurate, it is wrong.
        """
        self._x_prev = None
        self._dt_prev = math.nan

    def _interior(self) -> np.ndarray:
        return self.solver.cv[:, 1:-1].ravel().copy()

    def _write(self, x: np.ndarray) -> bool:
        """Put a flat interior state back and refresh pressure and ghosts.

        Returns False if the state is non-physical; Newton must be able to probe
        a bad state without the run dying, so this reports rather than raises.
        """
        s = self.solver
        n = s.grid.n_interior
        s.cv[:, 1:-1] = x.reshape(3, n)
        a = s.grid.a_cell
        rho = s.cv[0] / a
        if np.any(rho[1:-1] <= 0.0) or not np.all(np.isfinite(x)):
            return False
        if hasattr(s.gas, "gm1"):
            p = s.gas.gm1 * (s.cv[2] / a - 0.5 * s.cv[1] * s.cv[1] / (s.cv[0] * a))
        else:
            from .solver import _pressure_real

            p = _pressure_real(s.gas, s.cv, a)
        if np.any(p[1:-1] <= 0.0) or not np.all(np.isfinite(p[1:-1])):
            return False
        s.p[:] = p
        s._sync_boundaries()
        return True

    def _rhs(self, x: np.ndarray) -> np.ndarray | None:
        """``f(x) = -R(x)/dx``, the semi-discrete right-hand side."""
        if not self._write(x):
            return None
        return -(self.solver.residual() / self.solver.grid.dx).ravel()

    def _G(self, x: np.ndarray, x_ref: np.ndarray, coef: float) -> np.ndarray | None:
        """Defect ``x - x_ref - coef f(x)``. None if the state is unphysical."""
        f = self._rhs(x)
        if f is None:
            return None
        return x - x_ref - coef * f

    # -- linear solve -------------------------------------------------------

    def _preconditioner(self, coef: float):
        """Diagonal approximation of ``I + coef * dR/dU``.

        The dominant diagonal of the first-order upwind Jacobian is the local
        wave speed over the cell width, so ``1 + coef(|u|+c)/dx`` per cell,
        repeated for the three variables. Cheap, and worth having, but modest:
        measured on ``HPC01`` at Nc 0.5 it takes the Krylov count from ~1530 to
        ~1200 per step, not to the few dozen a good preconditioner would give.
        A block-tridiagonal ILU of an assembled first-order Jacobian is the
        obvious next step if the cost ever matters.
        """
        from scipy.sparse.linalg import LinearOperator

        s = self.solver
        _, u, _, c = s.primitives()
        lam = (np.abs(u[1:-1]) + c[1:-1]) / s.grid.dx
        d = np.tile(1.0 + coef * lam, 3)
        inv = 1.0 / d
        n = d.size
        return LinearOperator((n, n), matvec=lambda v: inv * v, dtype=float)

    def _gmres(self, x, x_ref, coef, g0, tol):
        """Solve ``J dx = -g0`` with restarted, preconditioned matrix-free GMRES."""
        from scipy.sparse.linalg import LinearOperator, gmres

        nrm = np.linalg.norm(x)
        base = self.config.jfnk_eps * (nrm if nrm > 0.0 else 1.0)
        failures = []

        def matvec(v):
            vn = np.linalg.norm(v)
            if vn == 0.0:
                return np.zeros_like(v)
            eps = base / vn
            gp = self._G(x + eps * v, x_ref, coef)
            if gp is None:
                failures.append(True)
                return v  # identity keeps GMRES well defined; step gets cut
            return (gp - g0) / eps

        n = x.size
        op = LinearOperator((n, n), matvec=matvec, dtype=float)
        pre = self._preconditioner(coef)
        count = [0]

        def tally(_):
            count[0] += 1

        try:
            dx, _ = gmres(
                op, -g0, rtol=tol, restart=30, M=pre, maxiter=self.config.max_krylov, callback=tally
            )
        except TypeError:  # scipy < 1.12 spells it `tol`
            dx, _ = gmres(
                op, -g0, tol=tol, restart=30, M=pre, maxiter=self.config.max_krylov, callback=tally
            )
        self.krylov_iterations += count[0]
        return dx, bool(failures)

    # -- one nonlinear solve ------------------------------------------------

    def _solve(self, x_ref: np.ndarray, coef: float, guess: np.ndarray):
        """Newton-Krylov on ``x - x_ref - coef f(x) = 0``. Returns ``(x, ok)``."""
        self.stages += 1
        x = guess.copy()
        g = self._G(x, x_ref, coef)
        if g is None:
            return x, False
        g_norm0 = np.linalg.norm(g)
        target = self.config.newton_tol * g_norm0 + self.config.newton_atol * np.linalg.norm(x)
        if g_norm0 <= target:
            return x, True  # already there; a converged field must cost nothing

        for _ in range(self.config.max_newton):
            self.newton_iterations += 1
            forcing = max(self.config.krylov_tol, min(0.1, np.linalg.norm(g) / g_norm0))
            dx, probe_failed = self._gmres(x, x_ref, coef, g, forcing)
            if probe_failed or not np.all(np.isfinite(dx)):
                return x, False
            # damped update: back off until the state is physical and the defect
            # does not grow
            lam = 1.0
            accepted = False
            for _ in range(12):
                trial = x + lam * dx
                gt = self._G(trial, x_ref, coef)
                if gt is not None and np.linalg.norm(gt) < np.linalg.norm(g):
                    x, g = trial, gt
                    accepted = True
                    break
                lam *= 0.5
            # A stalled line search at tolerance is success, not failure: near
            # round-off no step can reduce the defect further.
            if np.linalg.norm(g) <= target:
                return x, True
            if not accepted:
                return x, False
        return x, False

    # -- the schemes --------------------------------------------------------

    def _attempt_euler(self, x_old, h):
        return self._solve(x_old, h, x_old)

    def _attempt_bdf2(self, x_old, h):
        """Variable-step BDF2, self-starting with one backward-Euler step.

        With ``ω = h/h_prev`` the two-step formula is

        .. math::

            x_{n+1} = \\frac{(1+ω)^2}{1+2ω} x_n - \\frac{ω^2}{1+2ω} x_{n-1}
                      + \\frac{1+ω}{1+2ω} h f(x_{n+1}),

        which collapses to the familiar ``4/3, -1/3, 2/3`` at ``ω = 1``. Using
        the constant-step coefficients with a CFL-driven ``dt`` would quietly
        drop the scheme to first order, which is the usual way BDF2 disappoints.
        """
        if self._x_prev is None or not math.isfinite(self._dt_prev):
            return self._attempt_euler(x_old, h)
        w = h / self._dt_prev
        den = 1.0 + 2.0 * w
        x_ref = ((1.0 + w) ** 2 / den) * x_old - (w * w / den) * self._x_prev
        return self._solve(x_ref, (1.0 + w) / den * h, x_old)

    def _attempt_sdirk3(self, x_old, h):
        A = self._A
        f = [None, None, None]
        x = x_old
        for i in range(3):
            x_ref = x_old.copy()
            for j in range(i):
                x_ref += h * A[i, j] * f[j]
            x, ok = self._solve(x_ref, h * A[i, i], x)
            if not ok:
                return x, False
            if i < 2:
                fi = self._rhs(x)
                if fi is None:
                    return x, False
                f[i] = fi
        # stiffly accurate: the final stage is the new state
        return x, True

    # -- the step -----------------------------------------------------------

    def step(self, dt: float) -> float:
        """Advance one step, cutting ``dt`` if the nonlinear solve struggles.

        Returns the step size actually taken, which may be smaller than asked.
        """
        s = self.solver
        x_old = self._interior()
        self.newton_iterations = self.krylov_iterations = self.step_cuts = 0
        self.stages = 0
        attempt = getattr(self, f"_attempt_{self.scheme}")

        for cut in range(self.config.max_step_cuts + 1):
            h = dt * 0.5**cut
            x, ok = attempt(x_old, h)
            if ok and self._write(x):
                self.step_cuts = cut
                self._x_prev, self._dt_prev = x_old, h
                s.dt_last = h
                s.t += h
                return h
            self._write(x_old)  # roll back before retrying smaller

        self._write(x_old)
        raise NewtonDidNotConverge(
            f"{self.scheme} failed after {self.config.max_step_cuts} step cuts "
            f"from dt={dt:.6g}; smallest tried {dt * 0.5**self.config.max_step_cuts:.6g}"
        )
