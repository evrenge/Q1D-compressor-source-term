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
