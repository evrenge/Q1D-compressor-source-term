"""Compressor as a momentum and energy source term.

This is the component the whole project exists for. The design follows
``PLAN.md`` P1 and P2:

* **Nothing dimensional is tabulated.** The map returns ``PR`` and ``eta`` as
  functions of the dimensionless inlet flow function ``Phi1``; the force and
  power are computed at runtime from the *locally measured* upstream stagnation
  state. A table of ``Fx`` [N] against a scalar is only valid at the inlet
  condition it was generated at, which is the systematic error this project
  started from.
* **The lookup samples upstream.** The prototype sampled one cell *downstream*
  of the disk, inside the region smeared by the single-cell injection, which
  cost 1e-4 in the converged mass flow (``BASELINE.md``) and closed a feedback
  loop from the source onto its own input. Upstream of the disk the field is
  uniform to ~1e-8.
* **The source is a source.** A normalised distribution over ``n_smear`` cells,
  defaulting to one. No internal boundary conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .analytic import compressor_source_terms, flow_function, max_flow_function
from .gas import PerfectGas

if TYPE_CHECKING:  # pragma: no cover
    from .solver import Solver

__all__ = [
    "CompressorMap",
    "ConstantCompressorMap",
    "DiskState",
    "ActuatorDisk",
    "MappedCompressor",
    "MappedDiskState",
    "InletFlowCompressor",
    "UnsteadyMappedCompressor",
]


class CompressorMap(Protocol):
    """Maps inlet flow function to pressure ratio and isentropic efficiency."""

    def evaluate(self, phi1: float) -> tuple[float, float]: ...


@dataclass(frozen=True)
class ConstantCompressorMap:
    """Fixed ``PR`` and ``eta`` regardless of flow — the Phase 3 test case.

    Deliberately degenerate: with ``dPR/dPhi1 = 0`` there is no aerodynamic
    stiffness, so surge dynamics are structurally unrepresentable. That is
    fine here, because the point of this phase is that the *interface* is
    correct, and a constant map is the only case with a closed-form answer to
    check against. Real speedlines arrive in Phase 5.
    """

    pressure_ratio: float
    efficiency: float

    def evaluate(self, phi1: float) -> tuple[float, float]:
        return self.pressure_ratio, self.efficiency


@dataclass
class DiskState:
    """What the disk measured and applied on its last evaluation.

    Recorded for diagnostics and for tests that need to check the disk saw the
    state it was supposed to see, rather than inferring it from the converged
    field.
    """

    phi1: float = math.nan
    W: float = math.nan
    T01: float = math.nan
    p01: float = math.nan
    PR: float = math.nan
    eta: float = math.nan
    Fx: float = math.nan
    SWx: float = math.nan


@dataclass
class ActuatorDisk:
    """Compressor source term for a single-cell (or smeared) actuator disk.

    Call it with the solver; it returns the ``(3, n)`` source array the solver
    subtracts from its residual.

    Parameters
    ----------
    cell
        Interior cell index carrying the disk (0-based over interior cells).
    compressor_map
        Supplies ``PR`` and ``eta`` from the inlet flow function.
    sample_offset
        How many cells upstream of the disk to measure the inlet state.

        **Measured, not guessed** (``PLAN.md`` §7 Q1). The disk creates a
        numerical boundary layer that extends *upstream* as well as downstream,
        and its influence decays by roughly a factor of ten every two to three
        cells. Error in the converged mass flow against the 0D reference:

        ===========  ===========
        offset       W error
        ===========  ===========
        1            -3.3e-04
        3            -1.8e-05
        5            -1.0e-06
        8            +2.4e-08
        12           +7.5e-11
        16           +2.3e-13
        ===========  ===========

        The default of 12 clears the 1e-8 gate by three orders. An earlier
        default of 3 sat *inside* the layer and made the disk read
        ``p01 = 101318`` instead of ``101325`` — the same class of defect as the
        prototype's downstream sampling, on the other side.

        The unit is cells rather than length because the contamination comes
        from the reconstruction stencil, so it does not shrink with mesh
        refinement.
    n_smear
        Number of consecutive cells to spread the source over, starting at
        ``cell``. One is a zero-thickness disk.
    """

    cell: int
    compressor_map: CompressorMap
    sample_offset: int = 12
    n_smear: int = 1
    last: DiskState = field(default_factory=DiskState)

    _weights: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.n_smear < 1:
            raise ValueError(f"n_smear must be >= 1, got {self.n_smear!r}")
        if self.sample_offset < 1:
            raise ValueError(
                f"sample_offset must be >= 1, got {self.sample_offset!r}: sampling the disk "
                "cell itself would read a state the source is in the middle of creating"
            )
        self._weights = np.full(self.n_smear, 1.0 / self.n_smear)

    # -- geometry -----------------------------------------------------------

    def _validate(self, solver: Solver) -> float:
        """Check placement and return the disk area.

        A zero-thickness actuator disk has one area by definition. If the area
        changed across the disk cells, the momentum source would have to carry
        the wall reaction ``int p dA`` as well as the blade force, and the two
        are not separable at this level of modelling (``PLAN.md`` §4.5).
        """
        grid = solver.grid
        n = grid.n_interior
        last = self.cell + self.n_smear - 1
        if self.cell - self.sample_offset < 0 or last >= n:
            raise ValueError(
                f"disk at cell {self.cell} spanning {self.n_smear} cells with "
                f"sample_offset {self.sample_offset} does not fit in {n} interior cells"
            )
        areas = grid.a_face[self.cell : last + 2]
        if not np.all(areas == areas[0]):
            raise ValueError(
                f"area varies across the disk cells ({areas.min():.6g} to {areas.max():.6g}); "
                "the momentum source would then include the wall pressure reaction, which is "
                "not the blade force. Place the disk in a constant-area section"
            )
        return float(areas[0])

    # -- evaluation ---------------------------------------------------------

    def __call__(self, solver: Solver) -> np.ndarray:
        gas: PerfectGas = solver.gas
        area = self._validate(solver)
        grid = solver.grid

        # Measured upstream, where the field is clean. `i` indexes the full
        # cell array (ghost at 0), hence the +1 on the interior index.
        i = self.cell - self.sample_offset + 1
        rho, u, p, c = solver.primitives()
        rho_i, u_i, p_i = float(rho[i]), float(u[i]), float(p[i])

        T = p_i / (rho_i * gas.R)
        mach = u_i / float(c[i])
        T01 = T * (1.0 + 0.5 * gas.gm1 * mach * mach)
        p01 = p_i * (T01 / T) ** gas.g_over_gm1
        W = rho_i * u_i * float(grid.a_cell[i])

        phi1 = W * math.sqrt(gas.R * T01) / (area * p01)
        phi_max = max_flow_function(gas)
        if phi1 > phi_max:
            raise ValueError(
                f"disk at cell {self.cell} sampled a choked inlet: flow function {phi1:.6g} "
                f"exceeds the sonic maximum {phi_max:.6g} at W={W:.6g} kg/s, "
                f"p01={p01:.6g} Pa, T01={T01:.6g} K. The map cannot be evaluated there"
            )

        PR, eta = self.compressor_map.evaluate(phi1)
        Fx, SWx = compressor_source_terms(T01, p01, W, PR, eta, area, area, gas)

        self.last = DiskState(phi1=phi1, W=W, T01=T01, p01=p01, PR=PR, eta=eta, Fx=Fx, SWx=SWx)

        q = np.zeros((3, grid.n_interior))
        span = slice(self.cell, self.cell + self.n_smear)
        q[1, span] = Fx * self._weights
        q[2, span] = SWx * self._weights
        return q

    # -- convenience --------------------------------------------------------

    def design_flow_function(self, gas: PerfectGas, W: float, T01: float, p01: float, area: float):
        """``Phi1`` for a given operating point — for choosing a map abscissa."""
        return W * math.sqrt(gas.R * T01) / (area * p01)

    @staticmethod
    def flow_function_at_mach(mach: float, gas: PerfectGas) -> float:
        return flow_function(mach, gas)


@dataclass
class MappedDiskState(DiskState):
    """Adds the map coordinates to the recorded evaluation."""

    beta: float = math.nan
    corrected_speed: float = math.nan
    ecmf: float = math.nan
    Wc: float = math.nan
    corrected_work: float = math.nan
    M2: float = math.nan
    T02: float = math.nan


@dataclass
class MappedCompressor:
    """Compressor driven by a real β map, keyed on **exit** corrected flow.

    The inlet state is measured upstream and the exit state downstream, both
    outside the region smeared by the injection. ECMF is formed from the two
    and inverted to β, which is well posed: ICMF compresses the whole β range
    into 3–10% of mass flow above 72% speed and is not monotonic, whereas ECMF
    is monotonic on every speed line of both supplied maps (``PLAN.md`` §3.5).

    Using the downstream state is a *measurement*, not a prediction — but it
    does close a feedback loop through the source's own output, so the looked-up
    values are under-relaxed. ``relaxation = 1.0`` disables that.

    The map supplies ``PR`` and **corrected work** ``Δh₀/θ`` rather than
    efficiency (``PLAN.md`` §3.6), so the energy source is
    ``SWx = W·Δh₀`` with no isentropic inversion anywhere in the path.
    """

    cell: int
    beta_map: object  # BetaMap; annotated loosely to avoid a circular import
    corrected_speed: float
    sample_offset: int = 12
    downstream_offset: int = 12
    n_smear: int = 1
    relaxation: float = 0.3
    ramp_evaluations: int = 0
    last: MappedDiskState = field(default_factory=MappedDiskState)

    _weights: np.ndarray = field(init=False, repr=False, default=None)
    _pr: float = field(init=False, repr=False, default=math.nan)
    _cw: float = field(init=False, repr=False, default=math.nan)
    _calls: int = field(init=False, repr=False, default=0)
    _reversals: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        if self.n_smear < 1:
            raise ValueError(f"n_smear must be >= 1, got {self.n_smear!r}")
        if self.sample_offset < 1 or self.downstream_offset < 1:
            raise ValueError("sample offsets must be >= 1 so the disk cell itself is not read")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError(f"relaxation must be in (0, 1], got {self.relaxation!r}")
        self._weights = np.full(self.n_smear, 1.0 / self.n_smear)

    @property
    def reversals(self) -> int:
        """Evaluations skipped because the sampled station had reverse flow."""
        return self._reversals

    @staticmethod
    def _stagnation(solver: Solver, i: int) -> tuple[float, float, float]:
        """``(T0, p0, W)`` at full-array cell index ``i``."""
        gas = solver.gas
        rho, u, p, c = solver.primitives()
        T = float(p[i]) / (float(rho[i]) * gas.R)
        mach = float(u[i]) / float(c[i])
        T0 = T * (1.0 + 0.5 * gas.gm1 * mach * mach)
        p0 = float(p[i]) * (T0 / T) ** gas.g_over_gm1
        return T0, p0, float(rho[i]) * float(u[i]) * float(solver.grid.a_cell[i])

    def __call__(self, solver: Solver) -> np.ndarray:
        from .analytic import static_from_stagnation

        gas = solver.gas
        grid = solver.grid
        n = grid.n_interior
        last_cell = self.cell + self.n_smear - 1
        if self.cell - self.sample_offset < 0 or last_cell + self.downstream_offset >= n:
            raise ValueError(
                f"disk at cell {self.cell} with offsets "
                f"({self.sample_offset}, {self.downstream_offset}) does not fit in {n} cells"
            )
        area = float(grid.a_face[self.cell])
        if not np.all(grid.a_face[self.cell : last_cell + 2] == area):
            raise ValueError("area varies across the disk cells; see PLAN.md §4.5")

        i_up = self.cell - self.sample_offset + 1
        i_dn = last_cell + self.downstream_offset + 1
        T01, p01, W = self._stagnation(solver, i_up)
        T02m, p02m, _ = self._stagnation(solver, i_dn)

        theta = T01 / 288.15
        delta = p01 / 101325.0

        # During a startup transient the field can momentarily reverse at the
        # sampling station. The map has no meaning at negative flow, so the
        # source is switched off for that evaluation rather than the lookup
        # being fed a negative flow function. This is counted, and a run that
        # converges with reversals recorded should not be trusted.
        if W <= 0.0 or T02m <= 0.0 or p02m <= 0.0:
            self._reversals += 1
            self.last = MappedDiskState(W=W, T01=T01, p01=p01)
            return np.zeros((3, n))

        ecmf = W * math.sqrt(T02m / 288.15) / (p02m / 101325.0)

        point = self.beta_map.evaluate_at_ecmf(ecmf, self.corrected_speed)

        # Under-relax the feedback path: the lookup coordinate depends on the
        # downstream state this source produces.
        if math.isnan(self._pr):
            self._pr, self._cw = point.PR, point.corrected_work
        else:
            r = self.relaxation
            self._pr += r * (point.PR - self._pr)
            self._cw += r * (point.corrected_work - self._cw)

        dh0 = self._cw * theta
        T02 = T01 + dh0 / gas.cp
        p02 = self._pr * p01

        st1 = static_from_stagnation(T01, p01, W, area, gas)
        st2 = static_from_stagnation(T02, p02, W, area, gas)
        Fx = st2.p * area - st1.p * area + W * (st2.u - st1.u)
        SWx = W * dh0

        self.last = MappedDiskState(
            phi1=W * math.sqrt(gas.R * T01) / (area * p01),
            W=W,
            T01=T01,
            p01=p01,
            PR=self._pr,
            eta=point.efficiency,
            Fx=Fx,
            SWx=SWx,
            beta=point.beta,
            corrected_speed=self.corrected_speed,
            ecmf=ecmf,
            Wc=W * math.sqrt(theta) / delta,
            corrected_work=self._cw,
            M2=st2.M,
            T02=T02,
        )

        # Continuation ramp. A real map at design speed asks for tens of
        # kilonewtons and ~100 kJ/kg; applied instantaneously to one cell of a
        # duct at rest that drives the pressure negative within a few steps.
        # This is a numerical device for reaching steady state, not physics --
        # it scales to 1 well before convergence and has no effect on the
        # converged answer, which the hold test confirms.
        self._calls += 1
        ramp = 1.0
        if self.ramp_evaluations > 0:
            ramp = min(1.0, self._calls / float(self.ramp_evaluations))

        q = np.zeros((3, n))
        span = slice(self.cell, self.cell + self.n_smear)
        q[1, span] = ramp * Fx * self._weights
        q[2, span] = ramp * SWx * self._weights
        return q


@dataclass
class InletFlowCompressor:
    """Compressor driven by a real β map, keyed on **inlet** corrected flow.

    This is the closure the Q1D solver uses. :class:`MappedCompressor` keys on
    exit corrected flow, which is the right coordinate for a cycle code and the
    wrong one here: the exit state is produced by this source, so keying on it
    closes an algebraic loop through the source's own output. Measured loop gain
    ``-dlnPR/dlnECMF`` is 0.90 at design speed on ``SubsonicCompressor`` and
    1.02–1.10 across the top of ``TranssonicCompressor``. Above one the loop
    diverges for *every* under-relaxation factor, and it does: the ECMF closure
    oscillates 16 → 32 → 24 → 35 → 16 in ECMF at CFL 0.2, 0.1 and 0.05 alike,
    which is what rules out a timestep explanation.

    Here the only measurement is upstream, where the field is clean to ~1e-8
    (``PLAN.md`` Phase 3). There is no algebraic feedback at all — the source
    depends on the solver's state only through the physical dynamics, which are
    restoring: more flow → lower PR → lower exit static pressure against a fixed
    back pressure → the flow decelerates.

    The map supplies ``PR`` and **corrected work** ``Δh₀/θ``, so the energy
    source is ``SWx = W·Δh₀`` with no isentropic inversion in the path.
    """

    cell: int
    beta_map: object  # BetaMap; annotated loosely to avoid a circular import
    corrected_speed: float
    sample_offset: int = 12
    n_smear: int = 1
    ramp_evaluations: int = 0
    last: MappedDiskState = field(default_factory=MappedDiskState)

    _weights: np.ndarray = field(init=False, repr=False, default=None)
    _calls: int = field(init=False, repr=False, default=0)
    _reversals: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        if self.n_smear < 1:
            raise ValueError(f"n_smear must be >= 1, got {self.n_smear!r}")
        if self.sample_offset < 1:
            raise ValueError("sample_offset must be >= 1 so the disk cell itself is not read")
        if not self.beta_map.inlet_closure_is_invertible(self.corrected_speed):
            # Fail at construction, not a thousand steps into a run.
            self.beta_map.evaluate_at_Wc(1.0, self.corrected_speed)
        self._weights = np.full(self.n_smear, 1.0 / self.n_smear)

    @property
    def reversals(self) -> int:
        """Evaluations skipped because the sampled station had reverse flow."""
        return self._reversals

    def __call__(self, solver: Solver) -> np.ndarray:
        from .analytic import static_from_stagnation
        from .maps import P_REF, T_REF

        gas = solver.gas
        grid = solver.grid
        n = grid.n_interior
        last_cell = self.cell + self.n_smear - 1
        if self.cell - self.sample_offset < 0 or last_cell >= n:
            raise ValueError(
                f"disk at cell {self.cell} spanning {self.n_smear} cells with sample_offset "
                f"{self.sample_offset} does not fit in {n} interior cells"
            )
        area = float(grid.a_face[self.cell])
        if not np.all(grid.a_face[self.cell : last_cell + 2] == area):
            raise ValueError("area varies across the disk cells; see PLAN.md §4.5")

        i = self.cell - self.sample_offset + 1
        rho, u, p, c = solver.primitives()
        rho_i, u_i, p_i = float(rho[i]), float(u[i]), float(p[i])
        T = p_i / (rho_i * gas.R)
        mach = u_i / float(c[i])
        T01 = T * (1.0 + 0.5 * gas.gm1 * mach * mach)
        p01 = p_i * (T01 / T) ** gas.g_over_gm1
        W = rho_i * u_i * float(grid.a_cell[i])

        # A startup transient can momentarily reverse the flow at the sampling
        # station. The map has no meaning there, so the source is switched off
        # for that evaluation and the event counted; a run that converges with
        # reversals recorded is not to be trusted without inspecting them.
        if W <= 0.0:
            self._reversals += 1
            self.last = MappedDiskState(W=W, T01=T01, p01=p01)
            return np.zeros((3, n))

        theta, delta = T01 / T_REF, p01 / P_REF
        Wc = W * math.sqrt(theta) / delta
        point = self.beta_map.evaluate_at_Wc(Wc, self.corrected_speed)

        dh0 = point.corrected_work * theta
        T02 = T01 + dh0 / gas.cp
        p02 = point.PR * p01

        st1 = static_from_stagnation(T01, p01, W, area, gas)
        st2 = static_from_stagnation(T02, p02, W, area, gas)
        Fx = (st2.p - st1.p) * area + W * (st2.u - st1.u)
        SWx = W * dh0

        self.last = MappedDiskState(
            phi1=W * math.sqrt(gas.R * T01) / (area * p01),
            W=W,
            T01=T01,
            p01=p01,
            PR=point.PR,
            eta=point.efficiency,
            Fx=Fx,
            SWx=SWx,
            beta=point.beta,
            corrected_speed=self.corrected_speed,
            ecmf=point.ecmf,
            Wc=Wc,
            corrected_work=point.corrected_work,
            M2=st2.M,
            T02=T02,
        )

        # Continuation ramp: a numerical device for reaching steady state, not
        # physics. It reaches 1 well before convergence, so the converged answer
        # does not depend on it -- which the hold test confirms.
        self._calls += 1
        ramp = 1.0
        if self.ramp_evaluations > 0:
            ramp = min(1.0, self._calls / float(self.ramp_evaluations))

        q = np.zeros((3, n))
        span = slice(self.cell, self.cell + self.n_smear)
        q[1, span] = ramp * Fx * self._weights
        q[2, span] = ramp * SWx * self._weights
        return q


@dataclass
class UnsteadyMappedCompressor:
    """Compressor whose position on the map is a **state**, not a function.

    Every earlier disk in this module slides instantly along the speed line: a
    mass-flow perturbation immediately produces the full steady ``dPR/dW``. That
    is the one physically false step in the formulation, and it is what makes
    the disk an acoustic amplifier. A real blade row cannot reorganise its
    loading faster than the flow passes through it.

    Here ``beta`` obeys

    .. math::

        \\tau \\frac{d\\beta}{dt} = \\beta_\\text{map}(W_c) - \\beta

    and ``PR`` and ``Δh₀/θ`` are read at the *current* ``beta``. The momentum
    flux term is **not** lagged — that is instantaneous gas dynamics, and an
    earlier attempt that low-passed the whole applied source delayed it too,
    which damped the wrong thing.

    Why this matters, measured across 160 operating points on 14 maps
    (``PLAN.md`` §3.10). ``Z = |dFx/dW|/c`` compares the disk's resistance with
    the duct's characteristic impedance, and predicts the observed stability to
    96%. Sliding along the map it reaches 19,608; at frozen ``beta`` the same
    points give a median of 0.149 and a maximum of 0.391, against a measured
    stability threshold of 0.82. Fast waves see the frozen-``beta`` disk; the
    map slope enters only through the slow ``beta`` equation.

    ``tau`` defaults to the blade row's own through-flow time, ``ℓ_row / u``,
    which is reduced frequency one and needs no data the maps do not carry.
    Pass ``tau`` explicitly for a rotor-period estimate (``1/N``) if the design
    speed is known.

    The steady answer does not depend on ``tau``: at convergence
    ``beta = beta_map`` exactly, whatever ``tau`` was.
    """

    cell: int
    beta_map: object  # BetaMap; annotated loosely to avoid a circular import
    corrected_speed: float
    sample_offset: int = 12
    n_smear: int = 21
    tau: float | None = None  # None -> blade-row through-flow time
    last: MappedDiskState = field(default_factory=MappedDiskState)

    _weights: np.ndarray = field(init=False, repr=False, default=None)
    _beta: float = field(init=False, repr=False, default=math.nan)
    _t_prev: float = field(init=False, repr=False, default=math.nan)
    _tau: float = field(init=False, repr=False, default=math.nan)
    _reversals: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        if self.n_smear < 1:
            raise ValueError(f"n_smear must be >= 1, got {self.n_smear!r}")
        if self.sample_offset < 1:
            raise ValueError("sample_offset must be >= 1 so the disk cell itself is not read")
        if self.tau is not None and self.tau <= 0.0:
            raise ValueError(f"tau must be positive, got {self.tau!r}")
        if not self.beta_map.inlet_closure_is_invertible(self.corrected_speed):
            self.beta_map.evaluate_at_Wc(1.0, self.corrected_speed)
        self._weights = np.full(self.n_smear, 1.0 / self.n_smear)

    @property
    def reversals(self) -> int:
        return self._reversals

    @property
    def beta(self) -> float:
        """Current map position — the disk's internal state."""
        return self._beta

    @property
    def response_time(self) -> float:
        """``tau`` actually in use [s], including the derived default."""
        return self._tau

    def __call__(self, solver: Solver) -> np.ndarray:
        from .analytic import static_from_stagnation
        from .maps import P_REF, T_REF

        gas = solver.gas
        grid = solver.grid
        n = grid.n_interior
        last_cell = self.cell + self.n_smear - 1
        if self.cell - self.sample_offset < 0 or last_cell >= n:
            raise ValueError(
                f"disk at cell {self.cell} spanning {self.n_smear} cells with sample_offset "
                f"{self.sample_offset} does not fit in {n} interior cells"
            )
        area = float(grid.a_face[self.cell])
        if not np.all(grid.a_face[self.cell : last_cell + 2] == area):
            raise ValueError("area varies across the disk cells; see PLAN.md §4.5")

        i = self.cell - self.sample_offset + 1
        rho, u, p, c = solver.primitives()
        rho_i, u_i, p_i = float(rho[i]), float(u[i]), float(p[i])
        T = p_i / (rho_i * gas.R)
        mach = u_i / float(c[i])
        T01 = T * (1.0 + 0.5 * gas.gm1 * mach * mach)
        p01 = p_i * (T01 / T) ** gas.g_over_gm1
        W = rho_i * u_i * float(grid.a_cell[i])

        if W <= 0.0:
            self._reversals += 1
            self.last = MappedDiskState(W=W, T01=T01, p01=p01)
            return np.zeros((3, n))

        theta, delta = T01 / T_REF, p01 / P_REF
        Wc = W * math.sqrt(theta) / delta
        target = self.beta_map.evaluate_at_Wc(Wc, self.corrected_speed).beta

        if math.isnan(self._beta):
            # Start on the map, so a converged seed is not disturbed by the lag.
            self._beta = target
            self._t_prev = solver.t
            row_length = float(grid.x_face[last_cell + 1] - grid.x_face[self.cell])
            self._tau = self.tau if self.tau is not None else row_length / max(u_i, 1e-9)
        elif solver.t > self._t_prev:
            # Advance once per *step*, not once per Runge-Kutta stage.
            dt = solver.t - self._t_prev
            self._t_prev = solver.t
            self._beta += (1.0 - math.exp(-dt / self._tau)) * (target - self._beta)

        point = self.beta_map.evaluate_at_beta(self._beta, self.corrected_speed)

        dh0 = point.corrected_work * theta
        T02 = T01 + dh0 / gas.cp
        p02 = point.PR * p01

        st1 = static_from_stagnation(T01, p01, W, area, gas)
        st2 = static_from_stagnation(T02, p02, W, area, gas)
        Fx = (st2.p - st1.p) * area + W * (st2.u - st1.u)
        SWx = W * dh0

        self.last = MappedDiskState(
            phi1=W * math.sqrt(gas.R * T01) / (area * p01),
            W=W,
            T01=T01,
            p01=p01,
            PR=point.PR,
            eta=point.efficiency,
            Fx=Fx,
            SWx=SWx,
            beta=self._beta,
            corrected_speed=self.corrected_speed,
            ecmf=point.ecmf,
            Wc=Wc,
            corrected_work=point.corrected_work,
            M2=st2.M,
            T02=T02,
        )

        q = np.zeros((3, n))
        span = slice(self.cell, self.cell + self.n_smear)
        q[1, span] = Fx * self._weights
        q[2, span] = SWx * self._weights
        return q
