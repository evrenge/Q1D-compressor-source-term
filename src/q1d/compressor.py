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
    "FlowMatchedCompressor",
    "EcmfCompressor",
    "CompositeSource",
    "InletFilter",
    "rotor_period",
]


@dataclass
class InletFilter:
    """First-order lag on the inlet stagnation state a source term reads.

    **Why this exists.** The prototype tabulated ``Fx`` once at its design inlet
    state, which is the systematic error this project was built to remove — a
    table in newtons is only valid at the condition it was generated at. Phase 3
    replaced it with runtime evaluation from the locally measured state, which is
    correct for varying inlet conditions and which also opened an acoustic
    feedback loop (``PLAN.md`` §3.11):

        a wave raises ``p01`` at the sampling station
          → ``Fx ≈ (PR·p01 − p1)·A`` rises
          → a stronger wave is launched,     loop gain ≈ ``Fx/(p01·A)``

    The gain grows with pressure ratio, so the disk holds its point at PR 1.2
    and fails from 1.4 — the prototype's bug had been suppressing it, and Phase
    3's gate at 1.2 sat just inside the stable region.

    The filter tracks genuine inlet changes but not the disk's own echo. It has
    **unit DC gain**, so the converged answer does not depend on ``tau``:
    measured on the constant-PR case at PR 1.4, W error is 2.044e-10, 2.050e-10,
    2.050e-10, 2.049e-10 and 2.047e-10 at ``tau`` = 1e-3, 3e-3, 1e-2, 3e-2 and
    1e-1 s — invariant over a hundredfold range, against −6.1e-03 unfiltered.

    **Choosing tau.** It must exceed the acoustic transit of the disk's
    surroundings; the blade row's own through-flow time (~7e-5 s here) is far too
    short and does not work. A rotor period, ``tau ≈ 1/N``, does: 1e-3 s is about
    a sixth of a revolution at 10,000 rpm and is what the measurements above
    used. :func:`rotor_period` derives it when the shaft speed is known.

    **The mass flow is lagged too.** With only ``(T₀, p₀)`` filtered, PR 1.4 and
    1.6 hold but 2.0 does not: the sampling station's ``W`` still responds
    instantly to a passing wave and ``Fx`` depends on it, so one feedback path
    stays open. Lagging ``W`` on the same time constant closes it — at PR 2.0,
    ``τ`` = 3e-2 gives 2.161e-10 against −2.380e-02 with the stagnation state
    alone (``PLAN.md`` §3.13).

    ``tau = 0`` disables the filter and restores the unfiltered behaviour, which
    is correct only below PR ≈ 1.3.
    """

    tau: float = 0.0

    _T0: float = field(init=False, repr=False, default=math.nan)
    _p0: float = field(init=False, repr=False, default=math.nan)
    _W: float = field(init=False, repr=False, default=math.nan)
    _t: float = field(init=False, repr=False, default=math.nan)

    def __post_init__(self) -> None:
        if self.tau < 0.0:
            raise ValueError(f"inlet filter tau must be non-negative, got {self.tau!r}")

    @property
    def state(self) -> tuple[float, float, float]:
        """The filtered ``(T0, p0, W)`` currently in use."""
        return self._T0, self._p0, self._W

    def reset(self) -> None:
        self._T0 = self._p0 = self._W = self._t = math.nan

    def update(self, t: float, T0: float, p0: float, W: float) -> tuple[float, float, float]:
        """Advance the filter to time ``t`` and return the state to use.

        Advances **once per step**, not once per Runge-Kutta stage: the solver
        calls a source term at every stage, and integrating the filter at each
        of them would run it at five times the physical rate. ``Solver.advance``
        owns the clock, so a stage repeat is detected by ``t`` not having moved.

        The first call seeds the filter with the measured state, so starting
        from a converged field is not disturbed.
        """
        if self.tau <= 0.0:
            return T0, p0, W
        if math.isnan(self._T0):
            self._T0, self._p0, self._W, self._t = T0, p0, W, t
        elif t > self._t:
            alpha = 1.0 - math.exp(-(t - self._t) / self.tau)
            self._t = t
            self._T0 += alpha * (T0 - self._T0)
            self._p0 += alpha * (p0 - self._p0)
            self._W += alpha * (W - self._W)
        return self._T0, self._p0, self._W


@dataclass
class LocalLevelFilter:
    """First-order lag on the **local** static pressure and mass flux per cell.

    Its only job is to supply a reference *level* for the similarity scaling in
    :class:`InletFlowCompressor`. The source is injected as

        ``Fx · p_k / p̄_k``   and   ``SWx · m_k / m̄_k``

    with ``p̄`` and ``m̄`` the lagged values from here, so both ratios are
    **exactly one at any steady state**, whatever the discrete profile happens to
    be. That last property is why the reference is a lag of the field itself
    rather than an analytic prediction of it: an earlier version integrated the
    expected profile across the smear from the map point, which is right to the
    accuracy of the flux inversion but not to the accuracy of the discretisation,
    and the leftover mismatch biased the converged mass flow by −5.5e−04 — three
    orders outside the Phase 3 gate. Referencing the field against its own past
    cannot drift, because at convergence past and present are the same field.

    Dynamically it is what makes the source scale with the pressure *level* it
    actually sits in, on the same time constant as the operating point. Frozen
    within a step, exactly as :class:`InletFilter` is, so a Newton or Runge-Kutta
    stage sees one reference.

    ``tau = 0`` disables it and returns the field itself, giving ratios of one
    and the unscaled fixed-force injection.
    """

    tau: float = 0.0

    _p: np.ndarray | None = field(init=False, repr=False, default=None)
    _m: np.ndarray | None = field(init=False, repr=False, default=None)
    _t: float = field(init=False, repr=False, default=math.nan)

    def reset(self) -> None:
        self._p = self._m = None
        self._t = math.nan

    def update(self, t: float, p: np.ndarray, m: np.ndarray):
        if self.tau <= 0.0:
            return p, m
        if self._p is None:
            self._p, self._m, self._t = p.copy(), m.copy(), t
        elif t > self._t:
            alpha = 1.0 - math.exp(-(t - self._t) / self.tau)
            self._t = t
            self._p += alpha * (p - self._p)
            self._m += alpha * (m - self._m)
        return self._p, self._m


def rotor_period(rpm: float) -> float:
    """``tau = 1/N`` in seconds — a defensible blade-row response time.

    The row's own through-flow time is too short to break the acoustic loop
    (``PLAN.md`` §3.11); one shaft revolution is the next physical scale up and
    is long enough. Use it when the design speed is known rather than tuning
    ``tau`` until the tests pass — ``tau`` is a transient-response parameter and
    the project exists to model transients.
    """
    if rpm <= 0.0:
        raise ValueError(f"rpm must be positive, got {rpm!r}")
    return 60.0 / rpm


@dataclass
class CompositeSource:
    """Several sources on one duct, summed.

    The endgame is a whole engine on a single mesh — a multistage compressor,
    then a burner, then a turbine, each a source on its own cells. The solver
    takes exactly one ``source``, so the composition lives here.

    Summation is the right operation and not merely a convenient one: the
    right-hand side of the quasi-1D system is a sum of independent volumetric
    contributions, and each component writes to its own disjoint cells. It is
    *not* checked that the spans are disjoint — a bleed port and the blade row
    it bleeds from may legitimately share cells.

    Each member is called once per evaluation, in order, and each therefore sees
    the same solver state. Members that carry internal state advance it on their
    own ``solver.t`` detection, so the ordering here does not affect the result.
    """

    members: tuple

    def __init__(self, *members) -> None:
        if len(members) == 1 and not callable(members[0]):
            members = tuple(members[0])
        if not members:
            raise ValueError("CompositeSource needs at least one member")
        self.members = tuple(members)

    def __call__(self, solver: Solver) -> np.ndarray:
        out = self.members[0](solver)
        for m in self.members[1:]:
            out = out + m(solver)
        return out


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
    inlet_lag: float = 0.0
    #: See :class:`InletFlowCompressor` for what this is. **Off by default
    #: here**, unlike there, and the reason is history rather than principle:
    #: the narrow-smear penalty that motivated switching it off (§3.18) was
    #: caused by reading the level from the forced cell, and §3.20 fixed that by
    #: shifting the reference one cell upstream — after which a single-cell disk
    #: carries PR 5.040 and holds it to 8.6e−12. The staged train has **not**
    #: been re-measured with the shifted form, and this class is what the staged
    #: results were taken with, so the default stays off until it has been.
    #: Turning it on is reasonable; do it with a measurement, not on faith.
    similarity_scaling: bool = False
    last: DiskState = field(default_factory=DiskState)

    _weights: np.ndarray = field(init=False, repr=False, default=None)
    _filter: InletFilter = field(init=False, repr=False, default=None)
    _levels: LocalLevelFilter = field(init=False, repr=False, default=None)
    _chokes: int = field(init=False, repr=False, default=0)

    @property
    def chokes(self) -> int:
        """Evaluations where the sampled station exceeded the sonic flow function.

        Clamped to the sonic point rather than raised, so a startup transient does
        not kill a run. Non-zero on a *converged* run means the station is placed
        where the map cannot be read and the result should not be trusted.
        """
        return self._chokes

    def __post_init__(self) -> None:
        self._filter = InletFilter(self.inlet_lag)
        self._levels = LocalLevelFilter(self.inlet_lag)
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

    # -- reconstruction -----------------------------------------------------

    def low_order_faces(self, grid, margin: int = 2) -> np.ndarray:
        """Faces where MUSCL should drop to first order — pass to ``Solver``.

        The disk source is added to cell averages with nothing matching it in
        the reconstruction, so MUSCL reads the source-imposed profile inside the
        smeared region as a solution gradient. Measured at PR 2.0 (``PLAN.md``
        §3.13): first order holds the operating point to 2.5e-14 while second
        order misses by 4e-4 with a one-cell disk and by 4e-2 with 21 cells —
        widening the smear makes it *worse*, which rules out "the gradient is
        simply too steep".

        Restricting the fallback to the disk's own neighbourhood keeps the far
        field second order, and costs nothing physical: an actuator disk is a
        *model* of a blade row, not resolved geometry, so there is no sub-cell
        structure there to resolve accurately in the first place.

        ``margin`` extra cells each side cover the reconstruction stencil, which
        reaches one cell beyond the face it feeds.
        """
        n = grid.n_interior
        mask = np.zeros(n + 1, dtype=bool)
        lo = max(0, self.cell - margin)
        hi = min(n, self.cell + self.n_smear + margin)
        mask[lo : hi + 1] = True
        return mask

    # -- evaluation ---------------------------------------------------------

    def __call__(self, solver: Solver) -> np.ndarray:
        gas: PerfectGas = solver.gas
        area = self._validate(solver)
        grid = solver.grid

        # Measured upstream, where the field is clean. `i` indexes the full
        # cell array (ghost at 0), hence the +1 on the interior index.
        # Read the station from the conserved fluxes, not from cell-centred
        # primitives. The "numerical boundary layer" Phase 3 measured upstream of
        # the disk is the error of averaging a sharp profile over a cell, not a
        # property of the field: read this way the stagnation state is exact one
        # cell from the disk, where the cell-centred reading is off by 1.9e-04
        # and needs twelve cells to recover (`PLAN.md` 3.23). That is ~11 cells
        # per blade row of mesh budget.
        idx = self.cell - self.sample_offset
        st = solver.station_state_at(idx)
        T = st.p / (st.rho * gas.R)
        mach = st.u / math.sqrt(gas.gamma * gas.R * T)
        T01 = T * (1.0 + 0.5 * gas.gm1 * mach * mach)
        p01 = st.p * (T01 / T) ** gas.g_over_gm1
        # The conserved mass flux, not rho*u*A(x_centre). The cell-centred
        # product is uniform only to O(dx^2) where the area has curvature, and
        # that bias feeds the map lookup, so it moves the operating point rather
        # than just a diagnostic (`PLAN.md` 3.21). Correct with mass sources
        # too -- bleed makes the flux step by what it removes, which is what a
        # station downstream of it should read. Free here: `residual` caches the
        # fluxes before calling the source.
        W = solver.mass_flux_at(self.cell - self.sample_offset)

        # The blade row tracks its inlet condition, not its own acoustic echo
        # (PLAN.md 3.11, 3.13). Unit DC gain, so the converged answer is
        # unchanged. W is filtered too, or one feedback path stays open.
        T01, p01, W = self._filter.update(solver.t, T01, p01, W)

        phi_max = max_flow_function(gas)
        w_max = phi_max * area * p01 / math.sqrt(gas.R * T01)
        if W > w_max:
            # Clamped and counted, not raised — the same contract the reverse-flow
            # guard above uses, and for the same reason. A startup transient can
            # graze the sonic limit at a station placed close to the disk: at
            # sample_offset 1 the sampled p01 dips to 95.2 kPa against an inlet
            # 101.3 kPa and the flow function lands 0.005% over. Killing a run for
            # that is brittle, and an engine transient will graze these limits
            # routinely.
            #
            # The clamp is on W rather than on the flow function because
            # `compressor_source_terms` re-derives the station states from W and
            # would reject the same point again a few lines later. Clamping the
            # mass flow says the physical thing: the station cannot pass more
            # than sonic, so this is the nearest state it can actually be in.
            #
            # A run that records chokes is not to be trusted without looking at
            # them; `chokes` is public so a caller can assert on it, and a
            # *converged* run should show none.
            self._chokes += 1
            W = w_max
        phi1 = W * math.sqrt(gas.R * T01) / (area * p01)

        PR, eta = self.compressor_map.evaluate(phi1)
        Fx, SWx = compressor_source_terms(T01, p01, W, PR, eta, area, area, gas)

        self.last = DiskState(phi1=phi1, W=W, T01=T01, p01=p01, PR=PR, eta=eta, Fx=Fx, SWx=SWx)

        q = np.zeros((3, grid.n_interior))
        span = slice(self.cell, self.cell + self.n_smear)
        if self.similarity_scaling:
            # Same formulation as InletFlowCompressor: the level is read one
            # cell upstream of each forced cell, never from the forced cell
            # itself (`PLAN.md` §3.20).
            ref = slice(self.cell - 1, self.cell + self.n_smear - 1)
            p_now = solver.p[1:-1][ref]
            m_now = solver.cv[1, 1:-1][ref]
            p_ref, m_ref = self._levels.update(solver.t, p_now, m_now)
            q[1, span] = Fx * self._weights * (p_now / p_ref)
            q[2, span] = SWx * self._weights * (m_now / m_ref)
        else:
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

    **Similarity scaling** (``similarity_scaling``, default on) is what lets this
    hold above PR 2.3, and it is a correctness fix rather than a stabiliser.
    ``Fx`` and ``SWx`` are computed from the *lagged* sample, so injecting them
    as a fixed force in newtons and a fixed heat rate in watts freezes the
    dimensional **level** along with the operating point. That contradicts the
    map: a map asserts a pressure *ratio* and a corrected work, both invariant to
    the absolute pressure level and to the mass flow. A device holding newtons
    fixed is not a compressor — it is a rising branch from PR 2.26, and a device
    holding watts fixed runs away because ``Δh₀ = Ẇ/W`` grows as the flow falls
    (``PLAN.md`` §3.16).

    The repair is to lag the *dimensionless* operating point and never the
    dimensional scale factors, so the source is injected as

    .. math::

        q_\\rho u(k) = \\frac{F_x}{n}\\,\\frac{p_k}{p_k^\\text{expected}},
        \\qquad
        q_{\\rho E}(k) = \\frac{\\dot S_{Wx}}{\\dot m}\\,
                         \\frac{(\\rho u A)_k}{n},

    both evaluated from the **local** state of the cell being forced. Local
    matters: the same corrections driven from the upstream probe make every
    eigenvalue worse, because that path carries a transport delay (§3.16).

    Both ratios are one at the design point, so the operating point and every
    Phase 3 gate are unchanged; only ``dR/dU`` moves. Measured on ``HPC01``, the
    largest eigenvalue of the linearised steady state goes from +7.0, +44.1,
    +77.3, +99.9, +114.4 to **−69.8, −53.9, −39.0, −27.5, −19.1** at Nc 0.6
    through 1.0, i.e. PR 2.26 through 7.49.

    **The reference is one cell upstream, and that matters.** Reading the level
    from the forced cell is self-referential: a force raises that cell's own
    pressure through its own momentum equation, giving a loop of gain of order
    ``(PR−1)/(2·n_smear)`` — fine for a 21-cell smear, above unity for a single
    cell. Shifting the reference one cell upstream removes the diagonal term
    while keeping the proxy local. Measured on ``HPC01``, largest eigenvalue at
    PR 5.040:

    ==========  ==============  ==============
    ``n_smear`` forced cell     one cell up
    ==========  ==============  ==============
    1           **+88.3**       **−92.3**
    3           +8.1            −76.5
    7           −14.3           −77.8
    21          −34.6           −64.8
    ==========  ==============  ==============

    So the shift is not a refinement, it is what lets the whole pressure ratio
    go into **one node** — and it roughly doubles the margin at wide smears too.

    ``similarity_scaling=False`` restores the fixed-force, fixed-rate injection.
    It exists so the regression test can assert that the old form *fails*, and
    is not a supported configuration above PR 2.3.
    """

    cell: int
    beta_map: object  # BetaMap; annotated loosely to avoid a circular import
    corrected_speed: float
    sample_offset: int = 12
    n_smear: int = 1
    inlet_lag: float = 0.0
    similarity_scaling: bool = True
    ramp_evaluations: int = 0
    last: MappedDiskState = field(default_factory=MappedDiskState)

    _weights: np.ndarray = field(init=False, repr=False, default=None)
    _filter: InletFilter = field(init=False, repr=False, default=None)
    _calls: int = field(init=False, repr=False, default=0)
    _reversals: int = field(init=False, repr=False, default=0)
    _levels: LocalLevelFilter = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._filter = InletFilter(self.inlet_lag)
        self._levels = LocalLevelFilter(self.inlet_lag)
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

        # Read the station from the conserved fluxes rather than from
        # cell-centred primitives -- see `Solver.station_state_at` and
        # `PLAN.md` 3.23. Exact one cell from the disk, where the cell-centred
        # reading is off by 1.9e-04.
        idx = self.cell - self.sample_offset
        st = solver.station_state_at(idx)
        T = st.p / (st.rho * gas.R)
        mach = st.u / math.sqrt(gas.gamma * gas.R * T)
        T01 = T * (1.0 + 0.5 * gas.gm1 * mach * mach)
        p01 = st.p * (T01 / T) ** gas.g_over_gm1
        # The conserved mass flux, not rho*u*A(x_centre). The cell-centred
        # product is uniform only to O(dx^2) where the area has curvature, and
        # that bias feeds the map lookup, so it moves the operating point rather
        # than just a diagnostic (`PLAN.md` 3.21). Correct with mass sources
        # too -- bleed makes the flux step by what it removes, which is what a
        # station downstream of it should read. Free here: `residual` caches the
        # fluxes before calling the source.
        W = solver.mass_flux_at(self.cell - self.sample_offset)

        # A startup transient can momentarily reverse the flow at the sampling
        # station. The map has no meaning there, so the source is switched off
        # for that evaluation and the event counted; a run that converges with
        # reversals recorded is not to be trusted without inspecting them.
        if W <= 0.0:
            self._reversals += 1
            self.last = MappedDiskState(W=W, T01=T01, p01=p01)
            return np.zeros((3, n))

        T01, p01, W = self._filter.update(solver.t, T01, p01, W)
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
        if self.similarity_scaling:
            # Read the level one cell UPSTREAM of each forced cell. Reading it
            # from the forced cell itself is self-referential — a force raises
            # that cell's own pressure through its own momentum equation — and
            # the resulting gain, of order (PR−1)/(2·n_smear), exceeds one for a
            # narrow smear. Shifting by one removes the diagonal term while
            # keeping the proxy local, which is what makes a single-cell disk
            # work at all (`PLAN.md` §3.20).
            ref = slice(self.cell - 1, self.cell + self.n_smear - 1)
            p_now = solver.p[1:-1][ref]
            m_now = solver.cv[1, 1:-1][ref]
            p_ref, m_ref = self._levels.update(solver.t, p_now, m_now)
            q[1, span] = ramp * Fx * self._weights * (p_now / p_ref)
            q[2, span] = ramp * SWx * self._weights * (m_now / m_ref)
        else:
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

        # Read the station from the conserved fluxes rather than from
        # cell-centred primitives -- see `Solver.station_state_at` and
        # `PLAN.md` 3.23. Exact one cell from the disk, where the cell-centred
        # reading is off by 1.9e-04.
        idx = self.cell - self.sample_offset
        st = solver.station_state_at(idx)
        T = st.p / (st.rho * gas.R)
        mach = st.u / math.sqrt(gas.gamma * gas.R * T)
        T01 = T * (1.0 + 0.5 * gas.gm1 * mach * mach)
        p01 = st.p * (T01 / T) ** gas.g_over_gm1
        # The conserved mass flux, not rho*u*A(x_centre). The cell-centred
        # product is uniform only to O(dx^2) where the area has curvature, and
        # that bias feeds the map lookup, so it moves the operating point rather
        # than just a diagnostic (`PLAN.md` 3.21). Correct with mass sources
        # too -- bleed makes the flux step by what it removes, which is what a
        # station downstream of it should read. Free here: `residual` caches the
        # fluxes before calling the source.
        W = solver.mass_flux_at(self.cell - self.sample_offset)

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
            self._tau = self.tau if self.tau is not None else row_length / max(st.u, 1e-9)
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


#: Smallest β change the secant estimate will divide by. Below this the
#: difference quotient is noise, and the map slope is used instead.
SECANT_MIN_DBETA = 1e-9

#: How far the measured ``dR/dβ`` may depart from the map's estimate before it is
#: rejected as spurious. Generous on purpose: the whole point is that the map
#: slope is wrong by a factor near choke — measured 0.091 against 0.60 on
#: ``HighPqPCompr`` Nc 0.700 near the choke end, a ratio of 6.6.
SECANT_MAX_RATIO = 50.0


@dataclass
class FlowMatchedCompressor:
    """Compressor whose map position is driven by the flow **residual**.

    :class:`InletFlowCompressor` inverts the map: it measures ``Wc`` upstream and
    asks the speed line which β has that flow. That inverse does not exist on
    29% of the tabulated speed lines — 13 of 45 across the four supplied
    compressor maps, always the top ones — and the reason is rank deficiency
    rather than conditioning. On ``TranssonicCompressor`` at Nc 1.144 the whole
    β range spans 9.97e−03 in ``Wc`` while ``PR`` spans 4.94e−01: the speed line
    is vertical to within the tabulation, so no measurement of the inlet flow can
    select a point on it (``PLAN.md`` §3.26).

    Nothing in the physics needs that inverse. The steady operating point is the
    intersection of two curves in ``(W, PR)`` — the map's speed line, falling,
    and the duct with a fixed back pressure, rising — and a vertical line still
    crosses a rising one transversally. Only the *algorithm* was ill-posed.

    So β becomes a state driven by the residual and never by the inverse:

    .. math::

        \\tau \\frac{d\\beta}{dt}
            = K\\,\\frac{W_c^\\text{meas}/W_c^\\text{map}(\\beta) - 1}
                       {\\partial \\ln ECMF/\\partial \\beta}

    Four things make this work where the inverse does not.

    **The scale factor is the ECMF slope, and only ECMF will do.** The step is a
    damped Newton step on the residual, so the denominator wants to be
    ``∂R/∂β ≈ c\\,∂lnPR/∂β - ∂lnWc/∂β`` — the duct's flow response to pressure,
    less the map's own flow slope. Taking the duct constant ``c`` as one makes
    that ``-∂lnECMF/∂β``, and ECMF is monotonic in β on every speed line of
    every supplied map. That is not a convenience, it is the documented reason
    ECMF exists (:mod:`q1d.maps`), and it is the only one of the three
    candidates that survives: measured over all 45 tabulated lines of the four
    compressor maps, ``|∂lnECMF/∂β|`` never falls below **0.225** and never
    changes sign, whereas ``∂lnWc/∂β`` reverses on every refused line — which is
    exactly what stops :meth:`BetaMap.evaluate_at_Wc` inverting it — and
    ``∂lnPR/∂β`` reverses too, at the **surge peak**, on 20 of the 45. A pressure
    scale therefore drives β the wrong way past the peak, and did: it killed
    ``TwoStgRadialCompr`` Nc 0.600 near surge, a point the inverse holds.

    ECMF enters here as a *slope of the tabulated map at the current* ``β``, not
    as a measurement. That is the distinction §3.9 turns on: keying on a measured
    exit ECMF closes an algebraic loop of gain 0.90–1.10 through the source's own
    output, while reading the map's own derivative closes nothing.

    **The measurement stays upstream.** ``Wc`` is read at the same station as
    :class:`InletFlowCompressor`, so §3.14's property survives: the disk never
    reads its own output. Keying on the exit pressure instead would be worse than
    ill-conditioned, it would be *degenerate* — ``p₀₂`` is what this source
    injects, so ``PR_meas ≡ PR(β)`` to within the scheme's dissipation and the
    residual carries no information about β at all.

    **The sign is restoring, and both halves of it point the same way.**
    ``Wc_meas > Wc_map(β)`` means the duct is passing more than the map allows at
    this position, and the step moves β toward choke. That closes the gap twice
    over: ``Wc_map`` rises to meet the measurement, *and* ``PR`` falls, which
    raises the inlet static pressure against a fixed inlet total and brings the
    actual flow down. ``∂lnECMF/∂β`` carries the sign of the map's orientation,
    so no explicit sign convention for β is baked in — a map tabulated the other
    way round works unchanged.

    **The fixed point is the map, exactly.** At steady state ``dβ/dt = 0`` forces
    ``Wc_meas = Wc_map(β)``, which is the same equation :class:`InletFlowCompressor`
    solves — so this is a different *solver* for the identical closure, not a
    different machine. Measured on ``SubsonicCompressor`` at Nc 1.0, converged β
    lands on the design β to 1e−10 at f = 0.15, 0.5 and 0.85, and seeding β 0.2
    away in either direction walks back to the same value to 1.4e−10.

    ``tau`` and ``gain`` set only how fast it gets there, never where. ``tau``
    defaults to the blade row's through-flow time, as in
    :class:`UnsteadyMappedCompressor`, which is reduced frequency one and needs
    no data the maps do not carry.
    """

    cell: int
    beta_map: object  # BetaMap or ScaledMap; loose to avoid a circular import
    corrected_speed: float
    sample_offset: int = 2
    n_smear: int = 1
    inlet_lag: float = 0.0
    similarity_scaling: bool = True
    gain: float = 1.0
    tau: float | None = None  # None -> blade-row through-flow time
    beta0: float | None = None  # None -> start at mid-line
    secant: bool = False  # default flips once the sweep justifies it
    exit_offset: int | None = None  # not None -> key on measured exit ECMF
    direct: bool = False  # with exit_offset: plain lookup, no relaxation at all
    last: MappedDiskState = field(default_factory=MappedDiskState)

    _weights: np.ndarray = field(init=False, repr=False, default=None)
    _filter: InletFilter = field(init=False, repr=False, default=None)
    _levels: LocalLevelFilter = field(init=False, repr=False, default=None)
    _beta: float = field(init=False, repr=False, default=math.nan)
    _t_prev: float = field(init=False, repr=False, default=math.nan)
    _tau: float = field(init=False, repr=False, default=math.nan)
    _reversals: int = field(init=False, repr=False, default=0)
    _clamps: int = field(init=False, repr=False, default=0)
    _r_prev: float = field(init=False, repr=False, default=math.nan)
    _b_prev: float = field(init=False, repr=False, default=math.nan)
    _secant_uses: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        if self.n_smear < 1:
            raise ValueError(f"n_smear must be >= 1, got {self.n_smear!r}")
        if self.sample_offset < 1:
            raise ValueError("sample_offset must be >= 1 so the disk cell itself is not read")
        if self.tau is not None and self.tau <= 0.0:
            raise ValueError(f"tau must be positive, got {self.tau!r}")
        if not self.gain > 0.0:
            raise ValueError(f"gain must be positive, got {self.gain!r}")
        if self.exit_offset is not None and self.exit_offset < 1:
            raise ValueError("exit_offset must be >= 1 so the station is clear of the disk")
        if self.direct and self.exit_offset is None:
            raise ValueError("direct lookup needs exit_offset -- there is nothing to look up on")
        self._filter = InletFilter(self.inlet_lag)
        self._levels = LocalLevelFilter(self.inlet_lag)
        self._weights = np.full(self.n_smear, 1.0 / self.n_smear)
        if self.beta0 is not None:
            self._beta = float(np.clip(self.beta0, 0.0, 1.0))
        # Cache the speed line and its log-ECMF slope once. Neither depends on
        # the solver state, and `np.gradient` on the densified β grid is the same
        # piecewise-linear derivative the lookups themselves use.
        ecmf, wc, _, _, _ = self.beta_map._speed_line(self.corrected_speed)
        self._beta_grid = np.asarray(self.beta_map.beta, float)
        self._wc_line = wc
        self._ecmf_line = ecmf
        self._dlnecmf = np.gradient(np.log(ecmf), self._beta_grid)
        if np.abs(self._dlnecmf).min() <= 0.0:
            raise ValueError(
                f"{getattr(self.beta_map, 'name', 'map')}: ECMF is stationary in beta at "
                f"Nc={self.corrected_speed:.4g}, so it cannot scale the update"
            )

    @property
    def beta(self) -> float:
        """Current map position — the disk's internal state."""
        return self._beta

    @property
    def reversals(self) -> int:
        """Evaluations skipped because the sampled station had reverse flow."""
        return self._reversals

    @property
    def clamps(self) -> int:
        """Steps whose β update had to be clipped back into ``[0, 1]``.

        Non-zero means the closure asked for a point off the end of the speed
        line. A handful during startup is ordinary; a count that keeps rising at
        convergence means the demanded operating point is not on the map.
        """
        return self._clamps

    @property
    def response_time(self) -> float:
        """``tau`` actually in use [s], including the derived default."""
        return self._tau

    @property
    def secant_uses(self) -> int:
        """Steps whose ``dR/dβ`` came from the run rather than from the map.

        Zero on a run that never moves — a converged seed — and that is correct:
        with nothing to measure, the map slope is the only estimate there is.
        """
        return self._secant_uses

    def _measure_exit_ecmf(self, solver, gas, T01, p01, theta, delta) -> float:
        """``ECMF = Wc·√τ/PR`` with ``τ`` and ``PR`` read from the field.

        The station sits ``exit_offset`` cells past the last forced cell, read
        from the conserved fluxes like every other station here, so a mass source
        between the disk and the station would be accounted for rather than
        silently violating the reading.
        """
        idx = self.cell + self.n_smear - 1 + self.exit_offset
        if idx >= solver.grid.n_interior:
            raise ValueError(
                f"exit station at interior cell {idx} is outside the "
                f"{solver.grid.n_interior}-cell duct"
            )
        try:
            st = solver.station_state_at(idx)
        except Exception:  # noqa: BLE001 -- a transient can make the flux infeasible
            return math.nan
        w = solver.mass_flux_at(idx)
        if w <= 0.0 or st.p <= 0.0 or st.rho <= 0.0:
            return math.nan
        t = st.p / (st.rho * gas.R)
        m2 = st.u * st.u / (gas.gamma * gas.R * t)
        t02 = t * (1.0 + 0.5 * gas.gm1 * m2)
        p02 = st.p * (t02 / t) ** gas.g_over_gm1
        pr, tau = p02 / p01, t02 / T01
        if pr <= 0.0 or tau <= 0.0:
            return math.nan
        return (w * math.sqrt(theta) / delta) * math.sqrt(tau) / pr

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

        idx = self.cell - self.sample_offset
        st = solver.station_state_at(idx)
        T = st.p / (st.rho * gas.R)
        mach = st.u / math.sqrt(gas.gamma * gas.R * T)
        T01 = T * (1.0 + 0.5 * gas.gm1 * mach * mach)
        p01 = st.p * (T01 / T) ** gas.g_over_gm1
        W = solver.mass_flux_at(idx)

        if W <= 0.0:
            self._reversals += 1
            self.last = MappedDiskState(W=W, T01=T01, p01=p01)
            return np.zeros((3, n))

        T01, p01, W = self._filter.update(solver.t, T01, p01, W)
        theta, delta = T01 / T_REF, p01 / P_REF
        Wc = W * math.sqrt(theta) / delta

        if math.isnan(self._t_prev):
            if math.isnan(self._beta):
                self._beta = 0.5
            self._t_prev = solver.t
            row = float(grid.x_face[last_cell + 1] - grid.x_face[self.cell])
            self._tau = self.tau if self.tau is not None else row / max(st.u, 1e-9)
        elif solver.t > self._t_prev:
            # Advance once per *step*, not once per Runge-Kutta stage.
            dt = solver.t - self._t_prev
            self._t_prev = solver.t
            if self.exit_offset is None:
                wc_b = float(np.interp(self._beta, self._beta_grid, self._wc_line))
                residual = Wc / wc_b - 1.0
            else:
                # Key on ECMF formed from the FIELD downstream of the disk, using
                # the state as it stands at the start of this step -- one step
                # behind the beta about to be computed, so no algebraic loop is
                # closed. At convergence this is the same root as the inlet form
                # (`PLAN.md` §3.26); on the way in it is not the same path,
                # because the measured p02 carries the duct's actual response
                # rather than the disk's assertion of it.
                e_meas = self._measure_exit_ecmf(solver, gas, T01, p01, theta, delta)
                if math.isnan(e_meas):
                    residual = 0.0
                elif self.direct:
                    # No relaxation and no Newton step: read the map at the
                    # measured ECMF and take that as β. The relaxation existed to
                    # tame an algebraic loop, and reading the field at the START
                    # of the step already removes it — §3.9's loop-gain objection
                    # applies within a step, not across one.
                    self._beta = self.beta_map.evaluate_at_ecmf(
                        e_meas, self.corrected_speed
                    ).beta
                    residual = math.nan
                else:
                    e_b = float(np.interp(self._beta, self._beta_grid, self._ecmf_line))
                    residual = e_meas / e_b - 1.0
            # The map slope is the fallback estimate of dR/dbeta; it assumes the
            # duct constant c = 1. Where the run has moved far enough to say
            # otherwise, believe the run (see `secant`).
            deriv = -float(np.interp(self._beta, self._beta_grid, self._dlnecmf))
            if self.secant and not math.isnan(self._r_prev):
                db = self._beta - self._b_prev
                if abs(db) > SECANT_MIN_DBETA:
                    measured = (residual - self._r_prev) / db
                    lo, hi = deriv / SECANT_MAX_RATIO, deriv * SECANT_MAX_RATIO
                    if lo <= measured <= hi:
                        deriv = measured
                        self._secant_uses += 1
            if not math.isnan(residual):
                self._r_prev, self._b_prev = residual, self._beta
                nb = self._beta - (
                    1.0 - math.exp(-dt / self._tau)
                ) * self.gain * residual / deriv
                if nb < 0.0 or nb > 1.0:
                    self._clamps += 1
                self._beta = min(1.0, max(0.0, nb))

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
        if self.similarity_scaling:
            # See `InletFlowCompressor`: lag the dimensionless operating point,
            # never the dimensional level, and read the level one cell upstream
            # of each forced cell (`PLAN.md` §3.18, §3.20).
            ref = slice(self.cell - 1, self.cell + self.n_smear - 1)
            p_now = solver.p[1:-1][ref]
            m_now = solver.cv[1, 1:-1][ref]
            p_ref, m_ref = self._levels.update(solver.t, p_now, m_now)
            q[1, span] = Fx * self._weights * (p_now / p_ref)
            q[2, span] = SWx * self._weights * (m_now / m_ref)
        else:
            q[1, span] = Fx * self._weights
            q[2, span] = SWx * self._weights
        return q


@dataclass
class EcmfCompressor:
    """Compressor keyed on **exit ECMF**, read one step behind. No β anywhere.

    This is the closure the project ended up with, and it is smaller than
    everything it replaces. Each step:

    .. math::

        PR,\\ \\Delta h_0/\\theta \;=\; f\\bigl(ECMF(t-1),\\ N_c\\bigr)

    a single table read. No operating-point state, no time constant, no gain, no
    Newton step, no clamping. Contrast :class:`FlowMatchedCompressor`, which
    needed all of those to make an inlet-``Wc`` closure survive the lines where
    ``Wc`` is rank-deficient (``PLAN.md`` §3.26).

    **Why keying on the exit is not circular here.** §3.9 rejected exit-ECMF
    keying because the source would read its own output, with measured loop gain
    ``−dlnPR/dlnECMF`` of 0.90–1.10 — above one, where no relaxation converges.
    That loop exists only *within* a step. Reading the field at the **start** of
    the step uses a value produced by the previous step's operating point, so
    there is no algebraic loop to have a gain at all.

    The distinction that took longest to see: ECMF *reconstructed* as
    ``PR(β)·p₀₁`` is degenerate — the disk agreeing with itself, carrying no
    information. ECMF *measured from the field* is the duct's actual response,
    and the two coincide only at the fixed point. The cells that fail do not fail
    at the fixed point; they fail on the way to it.

    **Why ECMF and not inlet ``Wc``.** ECMF is monotonic in β on 135 of 135
    compressor and fan speed lines; inlet ``Wc`` is monotonic on far fewer —
    ``SingleStgRadialCompr`` 0/11, ``MediumPqPCompr`` 4/14, ``HighPqPCompr``
    5/10. Keying on the degenerate coordinate is what made 13 of 45 speed lines
    unrunnable.

    Measured against the inlet-``Wc`` closure, converged mass-flow error:

    ==================================  ==============  ==============
    case                                inlet ``Wc``    this
    ==================================  ==============  ==============
    ``HighPqPCompr`` 0.700 f 0.85       −1.90e−02       **+5.5e−06**
    ``TranssonicCompressor`` 1.000 f .5 −1.50e−01       **+1.3e−06**
    ``SubsonicCompressor`` 1.200 f 0.5  +1.76e−02       **+1.7e−06**
    ``SubsonicCompressor`` 1.000 f 0.5  +6.7e−12        +5.1e−07
    ==================================  ==============  ==============

    with zero clamped steps everywhere, against tens of thousands for the
    inlet closure. The residual ~1e−06 is the **map layer**, not this: it is
    ``BetaMap``'s O(Δβ²) inversion inconsistency, converges at order 2.18 with
    map densification, and vanishes entirely against an
    :class:`~q1d.maps.ECMFMap`, which reproduces its own key to 2.2e−16.

    **Station placement — ``exit_offset`` 2 or 3, never 1.** ``station_state_at``
    reads a cell's *upstream* face. At ``exit_offset`` 1 that face sits between
    the last forced cell and the first unforced one, and those two states differ
    by one cell's share of the source — at PR 25 with ``n_smear`` 7 a **58%
    pressure jump across a single face**. Roe's dissipation term scales with the
    state jump, and it corrupts the *mass flux* there: measured on the exact
    seeded design field, ``W`` reads **36% low** at PR 25, which carries straight
    into ECMF because stagnation pressure and temperature are right to 1–2%.

    At ``exit_offset`` ≥ 2 both cells straddling the face carry the same state,
    the jump is zero, and the flux is exact. Reading against the design value:

    ======  ========  ========  ========
    PR      offset 1  offset 2  offset 3
    ======  ========  ========  ========
    3.95    0.9539    0.9983    **1.0000**
    15.88   0.7681    0.9970    **1.0000**
    21.85   0.6881    0.9972    **1.0000**
    25.13   0.6460    0.9973    **1.0000**
    ======  ========  ========  ========

    The consequence of getting this wrong is not a small bias. The ECMF table is
    only ~38% wide, so above PR ≈ 16 the very first reading falls *below the
    table*, the disk applies the speed line's **maximum** PR — 26.885 against a
    design 21.846 — and the resulting 23% over-pressure drives the flow
    supersonic and kills the run (``PLAN.md`` §3.35, §3.36).

    Do not go past 3: §3.30 measured offsets 4 and 8 as failing to converge,
    because the transport delay from disk to station enters the t−1 path. Two and
    three are the usable values and they agree to five digits.

    **Use ``inlet_lag = key_lag = 1e-2``. Smaller is not faster and is not
    safe.** Both filters have unit DC gain, so ``tau`` cannot move the answer —
    measured over a hundredfold range on ``SubsonicCompressor`` Nc 1.0, the
    converged mass flow moves by 2e−10 relative (+4.0038e−07 at 1e−2 against
    +3.9953e−07 at 1e−4). What it does change is speed and robustness, and
    neither favours a smaller value:

    ==========  ==================  =============================
    ``tau``     control, steps      ``HighPqPCompr`` Nc 0.950
    ==========  ==================  =============================
    1e−2        9 000               **HELD** (PR 15.883)
    3e−3        9 000               died at 1164
    1e−3        8 000               died at 1122
    3e−4        6 500               HELD, 35 500 steps
    1e−4        14 000              died at 533
    ==========  ==================  =============================

    The speedup on offer is 1.4×, not the order of magnitude the 500-steps-per-
    time-constant arithmetic suggests: below ~1e−2 convergence is no longer
    filter-limited but set by the duct's own acoustic and convective settling,
    and at 1e−4 it gets slower again. Meanwhile high pressure ratio becomes
    erratic — dying at 3e−3 and 1e−3, surviving at 3e−4, dying at 1e−4. That is
    marginal stability, not a threshold, and not a region to operate in.
    """

    cell: int
    ecmf_map: object  # ECMFMap; annotated loosely to avoid a circular import
    corrected_speed: float
    sample_offset: int = 2
    exit_offset: int = 2
    n_smear: int = 1
    inlet_lag: float = 0.0
    key_lag: float | None = None  # None -> same as inlet_lag
    similarity_scaling: bool = True
    last: MappedDiskState = field(default_factory=MappedDiskState)

    _weights: np.ndarray = field(init=False, repr=False, default=None)
    _filter: InletFilter = field(init=False, repr=False, default=None)
    _levels: LocalLevelFilter = field(init=False, repr=False, default=None)
    _key: float = field(init=False, repr=False, default=math.nan)
    _point: object = field(init=False, repr=False, default=None)
    _t_prev: float = field(init=False, repr=False, default=math.nan)
    _reversals: int = field(init=False, repr=False, default=0)
    _stalls: int = field(init=False, repr=False, default=0)
    _off_table: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        if self.n_smear < 1:
            raise ValueError(f"n_smear must be >= 1, got {self.n_smear!r}")
        if self.sample_offset < 1:
            raise ValueError("sample_offset must be >= 1 so the disk cell itself is not read")
        if self.exit_offset < 1:
            raise ValueError("exit_offset must be >= 1 so the station is clear of the disk")
        if self.key_lag is not None and self.key_lag < 0.0:
            raise ValueError(f"key_lag must be non-negative, got {self.key_lag!r}")
        self._filter = InletFilter(self.inlet_lag)
        self._levels = LocalLevelFilter(self.inlet_lag)
        self._weights = np.full(self.n_smear, 1.0 / self.n_smear)

    @property
    def reversals(self) -> int:
        """Evaluations skipped because the sampled station had reverse flow."""
        return self._reversals

    @property
    def stalls(self) -> int:
        """Steps whose exit reading was unusable, so the last point was held.

        Non-zero during a violent startup is ordinary. Non-zero at convergence
        means the exit station is not seeing a physical state and the run should
        not be trusted.
        """
        return self._stalls

    @property
    def point(self):
        """The map point currently applied — frozen within a step."""
        return self._point

    @property
    def off_table(self) -> int:
        """Steps whose demanded ECMF fell outside the tabulated speed line.

        **A non-zero count is not by itself a failure — a persistent one is.**
        Startup routinely goes off-table and recovers: ``HighPqPCompr`` Nc 1.000
        f = 0.15 logs 824 excursions and still holds at +5.6e−08. What matters is
        whether the count is still rising once the flow has settled, because the
        map holds no information beyond its ends and the lookup clamps there —
        the right thing to do, since §3.8 measures extrapolation as the worst
        error source on these maps, but it leaves the operating point pinned with
        no restoring force outward.

        Measured on ``HighPqPCompr`` Nc 0.950 (``PLAN.md`` §3.34): a design point
        placed *exactly* on the end of the ECMF range gives −7.54e−02 and never
        converges, while the same line **one percent** inside holds at +4.14e−08.
        The transition is a step, not a slope, and it is invisible without this
        counter.
        """
        return self._off_table

    def _map_off_table(self) -> int:
        """The map's own off-table tally, or 0 for a map that does not keep one.

        Read either side of the lookup so each disk attributes only its own
        excursions — an engine shares one map object between many components.
        """
        counter = getattr(self.ecmf_map, "off_table", None)
        return 0 if counter is None else int(counter[0])

    def _exit_ecmf(self, solver, gas, T01, p01, theta, delta) -> float:
        idx = self.cell + self.n_smear - 1 + self.exit_offset
        if idx >= solver.grid.n_interior:
            raise ValueError(
                f"exit station at interior cell {idx} is outside the "
                f"{solver.grid.n_interior}-cell duct"
            )
        try:
            st = solver.station_state_at(idx)
        except Exception:  # noqa: BLE001 -- a transient can make the flux infeasible
            return math.nan
        w = solver.mass_flux_at(idx)
        if w <= 0.0 or st.p <= 0.0 or st.rho <= 0.0:
            return math.nan
        t = st.p / (st.rho * gas.R)
        m2 = st.u * st.u / (gas.gamma * gas.R * t)
        t02 = t * (1.0 + 0.5 * gas.gm1 * m2)
        p02 = st.p * (t02 / t) ** gas.g_over_gm1
        pr, tau = p02 / p01, t02 / T01
        if pr <= 0.0 or tau <= 0.0:
            return math.nan
        return (w * math.sqrt(theta) / delta) * math.sqrt(tau) / pr

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

        idx = self.cell - self.sample_offset
        st = solver.station_state_at(idx)
        T = st.p / (st.rho * gas.R)
        mach = st.u / math.sqrt(gas.gamma * gas.R * T)
        T01 = T * (1.0 + 0.5 * gas.gm1 * mach * mach)
        p01 = st.p * (T01 / T) ** gas.g_over_gm1
        W = solver.mass_flux_at(idx)

        if W <= 0.0:
            self._reversals += 1
            self.last = MappedDiskState(W=W, T01=T01, p01=p01)
            return np.zeros((3, n))

        T01, p01, W = self._filter.update(solver.t, T01, p01, W)
        theta, delta = T01 / T_REF, p01 / P_REF

        # Refresh the operating point once per STEP, from the field as it stands
        # at the start of it. Refreshing per Runge-Kutta stage would reintroduce
        # the algebraic loop this design exists to avoid.
        if self._point is None or solver.t > self._t_prev:
            t_prev = self._t_prev if not math.isnan(self._t_prev) else solver.t
            self._t_prev = solver.t
            e = self._exit_ecmf(solver, gas, T01, p01, theta, delta)
            if math.isnan(e):
                if self._point is None:
                    raise ValueError(
                        "the exit station is not physical on the first evaluation; "
                        "seed the duct with a steady profile before marching"
                    )
                self._stalls += 1
            else:
                # Lag the key, for the reason §3.11-§3.13 lag the inlet state: an
                # unfiltered input lets the disk hear its own acoustic echo, and
                # that loop's gain grows with pressure ratio. Unit DC gain, so the
                # converged answer does not depend on `key_lag`.
                tau = self.inlet_lag if self.key_lag is None else self.key_lag
                if tau <= 0.0 or math.isnan(self._key):
                    self._key = e
                else:
                    self._key += (1.0 - math.exp(-(solver.t - t_prev) / tau)) * (
                        e - self._key
                    )
                before = self._map_off_table()
                self._point = self.ecmf_map.evaluate(self._key, self.corrected_speed)
                self._off_table += self._map_off_table() - before

        point = self._point
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
            beta=math.nan,
            corrected_speed=self.corrected_speed,
            ecmf=point.ecmf,
            Wc=W * math.sqrt(theta) / delta,
            corrected_work=point.corrected_work,
            M2=st2.M,
            T02=T02,
        )

        q = np.zeros((3, n))
        span = slice(self.cell, self.cell + self.n_smear)
        if self.similarity_scaling:
            # See `InletFlowCompressor`: lag the dimensionless operating point,
            # never the dimensional level, and read the level one cell upstream
            # of each forced cell (`PLAN.md` §3.18, §3.20).
            ref = slice(self.cell - 1, self.cell + self.n_smear - 1)
            p_now = solver.p[1:-1][ref]
            m_now = solver.cv[1, 1:-1][ref]
            p_ref, m_ref = self._levels.update(solver.t, p_now, m_now)
            q[1, span] = Fx * self._weights * (p_now / p_ref)
            q[2, span] = SWx * self._weights * (m_now / m_ref)
        else:
            q[1, span] = Fx * self._weights
            q[2, span] = SWx * self._weights
        return q
