"""The real-gas branch of the Roe flux, against the two things that pin it.

The perfect-gas branch has been exercised for the whole project; the real-gas
branch is new, and its failure mode is not a wrong answer but a *slowly growing
one*. Both gates here are exact identities rather than tolerances, and both were
written after the second of the two real-gas eigenstructure errors was found by
watching a compressor blow up 96 steps in:

1. **Branch consistency.** Handed a gas whose ``cp`` happens to be constant, the
   real-gas branch must reproduce the perfect-gas branch bit for bit-ish. This
   catches anything that is simply mis-transcribed.
2. **Datum invariance.** Adding a constant to the enthalpy zero is a gauge
   choice: with fixed composition and no mass source it shifts ``E`` by
   ``rho*h0`` and the energy flux by ``(rho*u)*h0``, both of which telescope
   against mass conservation. So the numerical energy flux must shift by exactly
   ``h0`` times the numerical *mass* flux, and nothing else may move at all.

   Gate 1 alone does not catch the eigenstructure error, because on a constant
   ``cp`` gas with ``h = cp*T`` the wrong expression and the right one coincide.
   That is precisely the trap: NASA9 enthalpies are referenced to the enthalpy of
   formation, which on air is about -299 kJ/kg away from ``cp*T``, and the
   entropy wave's energy eigenvector is ``H - c^2/kappa`` -- which collapses to
   the familiar ``u^2/2`` only on the ``h = cp*T`` datum.
"""

import numpy as np
import pytest

from q1d.boundary import Transmissive
from q1d.gas import PerfectGas
from q1d.grid import Grid
from q1d.solver import ReferenceState, Solver, SolverConfig

PERFECT = PerfectGas(gamma=1.4, cp=1005.0)


class _ConstantCpReal:
    """``cp`` really is constant, but the class does not advertise ``gamma``.

    Which is the only thing the solver dispatches on, so this takes the
    *real-gas* path while describing a gas whose exact answer is known from the
    perfect-gas path. ``h0`` moves the enthalpy datum.
    """

    def __init__(self, cp=1005.0, gamma=1.4, h0=0.0):
        self._cp = cp
        self._gamma = gamma
        self.h0 = h0
        self.R = (gamma - 1.0) * cp / gamma

    def cp_at(self, T):
        return np.full_like(np.asarray(T, dtype=float), self._cp)

    def gamma_at(self, T):
        return np.full_like(np.asarray(T, dtype=float), self._gamma)

    def entropy_ref(self, T):
        return self._cp * np.log(np.asarray(T, dtype=float))

    def pressure_ratio_isentropic(self, T1, T2):
        return np.exp((self.entropy_ref(T2) - self.entropy_ref(T1)) / self.R)

    def temperature_isentropic(self, T1, pressure_ratio):
        return T1 * np.exp(self.R * np.log(pressure_ratio) / self._cp)

    def enthalpy(self, T):
        return self._cp * np.asarray(T, dtype=float) + self.h0

    def temperature_from_enthalpy(self, h, guess=300.0):
        return (h - self.h0) / self._cp

    def speed_of_sound(self, T):
        return np.sqrt(self._gamma * self.R * np.asarray(T, dtype=float))


class _ShiftedDatum:
    """``inner`` with its enthalpy zero moved by ``h0``. Everything else same."""

    def __init__(self, inner, h0):
        self._inner = inner
        self.h0 = h0
        self.R = inner.R

    def enthalpy(self, T):
        return self._inner.enthalpy(T) + self.h0

    def temperature_from_enthalpy(self, h, guess=300.0):
        return self._inner.temperature_from_enthalpy(h - self.h0, guess)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _solver(gas):
    grid = Grid.uniform(0.0, 1.0, 20, lambda x: 1.0 + 0.2 * x)
    return Solver(
        grid,
        gas,
        Transmissive(),
        ReferenceState(rho=1.2, u=150.0, p=101325.0),
        config=SolverConfig(order=2),
    )


def _face_states(seed=0):
    """A pair of left/right face states with genuine jumps in all three."""
    rng = np.random.default_rng(seed)
    n = 21
    rho = 1.2 * (1.0 + 0.25 * rng.standard_normal(n))
    u = 150.0 + 60.0 * rng.standard_normal(n)
    p = 101325.0 * (1.0 + 0.25 * rng.standard_normal(n))
    left = np.array([rho, u, p])
    rho2 = 1.2 * (1.0 + 0.25 * rng.standard_normal(n))
    u2 = 150.0 + 60.0 * rng.standard_normal(n)
    p2 = 101325.0 * (1.0 + 0.25 * rng.standard_normal(n))
    return left, np.array([rho2, u2, p2])


class TestTheBranchesAgree:
    def test_a_constant_cp_gas_gives_the_perfect_gas_flux(self):
        left, right = _face_states()
        exact = _solver(PERFECT)._roe_flux(left, right)
        viareal = _solver(_ConstantCpReal())._roe_flux(left, right)
        assert viareal == pytest.approx(exact, rel=1e-12)

    def test_the_two_branches_reach_the_same_steady_state(self):
        """Not just the flux: a whole nozzle solve, marched to convergence."""
        from q1d.boundary import StagnationInletStaticOutlet

        def area(x):
            return 1.0 - 0.4 * np.sin(np.pi * x)

        results = []
        for gas in (PERFECT, _ConstantCpReal()):
            grid = Grid.uniform(0.0, 1.0, 101, area)
            bc = StagnationInletStaticOutlet(p0_in=101325.0, T0_in=288.15, p_back=95000.0)
            s = Solver(
                grid,
                gas,
                bc,
                ReferenceState(rho=1.2, u=100.0, p=101325.0),
                config=SolverConfig(cfl=1.5),
            )
            s.set_state(1.2, 100.0, 101325.0)
            s.run(max_steps=4000, tol=1e-10)
            results.append(s.face_fluxes()[0].mean())
        assert results[1] == pytest.approx(results[0], rel=1e-9)


class TestTheEnthalpyDatumIsAGauge:
    """Move the enthalpy zero; only the energy flux may move, and by exactly
    ``h0`` times the mass flux."""

    @pytest.mark.parametrize("h0", [-2.99e5, +5.0e5])
    def test_on_a_constant_cp_gas(self, h0):
        left, right = _face_states(seed=3)
        base = _solver(_ConstantCpReal())._roe_flux(left, right)
        moved = _solver(_ConstantCpReal(h0=h0))._roe_flux(left, right)
        assert moved[0] == pytest.approx(base[0], rel=1e-13)
        assert moved[1] == pytest.approx(base[1], rel=1e-13)
        assert moved[2] == pytest.approx(base[2] + h0 * base[0], rel=1e-11)

    @pytest.mark.parametrize("h0", [-2.99e5, +5.0e5])
    def test_on_the_nasa9_gas(self, h0):
        """The case that matters: ``h`` genuinely is not ``cp*T`` here."""
        pytest.importorskip("cantera")
        from q1d.gas import Nasa9Gas

        gas = Nasa9Gas.from_cantera()
        left, right = _face_states(seed=5)
        base = _solver(gas)._roe_flux(left, right)
        moved = _solver(_ShiftedDatum(gas, h0))._roe_flux(left, right)
        assert moved[0] == pytest.approx(base[0], rel=1e-12)
        assert moved[1] == pytest.approx(base[1], rel=1e-12)
        assert moved[2] == pytest.approx(base[2] + h0 * base[0], rel=1e-10)

    def test_the_premise_holds_on_air(self):
        """``h - cp*T`` is large on NASA9 air, so the gauge really is being
        exercised rather than being a no-op dressed as a test."""
        pytest.importorskip("cantera")
        from q1d.gas import Nasa9Gas

        gas = Nasa9Gas.from_cantera()
        gap = gas.enthalpy(288.15) - gas.cp_at(288.15) * 288.15
        assert gap == pytest.approx(-2.99e5, rel=0.02)
