"""The NASA9 gas driving the actual machines, not just the thermodynamics.

``test_nasa9.py`` checks the polynomials against Cantera and ``test_realgas_flux``
checks the Roe branch. Neither would have caught the two defects this file
exists for, because both live in the *coupling* between the map, the design
point and the marching solver — and both are invisible on a calorically perfect
gas, which is why the 1400-cell perfect-gas library sweep says nothing about
them.

**The disk and the design point must read the map the same way.** ``Δh₀ = CW·θ``
is the map's similarity scaling and it is exact only at constant ``cp``; the real
gas takes ``PR`` and ``η`` to the actual ``T₀₁`` instead. Those two agree
identically at ``θ = 1`` — a compressor at standard day — and diverge as ``θ``
moves away from it. A turbine at 1600 K runs at ``θ = 5.55``, and there the disk
was imposing one exit state while the duct had been sized for another: **-1.7e-02
and -3.0e-02** in held mass flow on the two turbines.

**The inlet characteristic must be one equation.** It formed the outgoing
invariant with ``γ`` at the interior temperature and inverted it with ``γ`` at
``T₀`` through the constant-``γ`` relation ``T = T₀(c/c₀)²``. For a perfect gas
those are the same number and a uniform duct is a fixed point of the condition.
For a real gas they are not, the fixed point is lost, and the sampled inlet
``p₀₁`` sits **1.4e-04 below the imposed ``p0_in``** — which moves the operating
point, because the disk keys on a corrected quantity built from it.

**A measured ``τ`` is not a tabulated ``τ``.** The ECMF key axis is built from
``τ`` at ``T_ref``; the disk measures ``T₀₂/T₀₁`` at the running temperature. For
a perfect gas those are the same number, which is what corrected parameters *are*.
On NASA9 air at a turbine's 1600 K they differ by 3.5%, shifting the key by
1.7e-02.

With all three fixed, the two supplied turbines hold to +1.4e-06 and -4.6e-07
against a perfect-gas +9.2e-07 and -1.7e-06 — the same interpolation floor,
reached from -1.7e-02 and -3.0e-02.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from q1d.analytic import (
    exit_stagnation_from_map,
    reference_tau,
    stagnation_from_static,
    state_from_flux,
)
from q1d.boundary import StagnationInletStaticOutlet
from q1d.compressor import EcmfCompressor, _exit_from_point, _reference_tau
from q1d.design import _exit_stagnation, design_from_map
from q1d.gas import PerfectGas
from q1d.grid import Grid
from q1d.maps import P_REF, T_REF, ECMFMap, load_beta_map
from q1d.solver import NonPhysicalState, ReferenceState, Solver, SolverConfig

pytest.importorskip("cantera")
from q1d.gas import Nasa9Gas  # noqa: E402

PERFECT = PerfectGas(1.4, 1004.7)
DATA = Path(__file__).resolve().parents[1] / "data" / "maps"
COMPRESSOR = DATA / "SubsonicCompressor.xlsx"

#: The gate every hold test in this project uses: mass flow held to 1e-6.
HOLD = 1e-6


@pytest.fixture(scope="module")
def nasa9():
    return Nasa9Gas.from_cantera()


def turbine_workbook(path):
    """A turbine map in the supplied workbooks' layout, written in memory.

    The vendor turbine maps live outside the repository, so the marching turbine
    case builds its own. Expansion ratio and corrected flow both rise with β,
    which is what ``_infer_kind`` reads as a turbine.
    """
    import pandas as pd

    beta = np.linspace(0.0, 1.0, 11)
    speed = (0.7, 0.8, 0.9, 1.0)
    PR = np.linspace(1.6, 3.2, 11)[:, None] * np.ones((1, 4))
    Wc = np.linspace(9.0, 12.5, 11)[:, None] * np.linspace(1.0, 1.12, 4)
    eff = np.full((11, 4), 0.90)
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame({"i": np.arange(len(beta)), "beta": beta}).to_excel(
            xl, sheet_name="beta_lines", index=False
        )
        for sheet, a in (("mass_flow", Wc), ("pressure_ratio", PR), ("efficiency", eff)):
            df = pd.DataFrame(a, columns=[str(s) for s in speed])
            df.insert(0, "beta", beta)
            df.to_excel(xl, sheet_name=sheet, index=False)
    return path


# ---------------------------------------------------------------------------
# The two paths into a map point must agree — the cheap gate on the same bug
# ---------------------------------------------------------------------------


class TestTheDiskAndTheDesignReadTheMapAlike:
    """``design.py`` sizes the duct; the disk imposes the source. If they take
    different routes from ``PR``/``η``/``CW`` to ``(T₀₂, p₀₂)`` the duct is
    solving a different machine from the one it was built for, and no amount of
    marching converges that away."""

    @pytest.mark.parametrize("T01", [288.15, 500.0, 1600.0])
    def test_they_agree_on_a_compressor_at_any_inlet_temperature(self, nasa9, T01):
        m = load_beta_map(COMPRESSOR, nasa9)
        point = m.evaluate_at_beta(0.5, 1.0)
        p01 = 101325.0
        want = _exit_stagnation(point, T01, p01, nasa9, "compressor")
        got = _exit_from_point(point, T01, p01, T01 / T_REF, nasa9, "compressor")
        assert got[0] == pytest.approx(want[0], rel=1e-12)
        assert got[1] == pytest.approx(want[1], rel=1e-12)

    @pytest.mark.parametrize("T01", [288.15, 1600.0])
    def test_they_agree_on_a_turbine_at_turbine_entry(self, nasa9, tmp_path, T01):
        m = load_beta_map(turbine_workbook(tmp_path / "t.xlsx"), nasa9)
        point = m.evaluate_at_beta(0.5, 0.9)
        p01 = 1.2e6
        want = _exit_stagnation(point, T01, p01, nasa9, "turbine")
        got = _exit_from_point(point, T01, p01, T01 / T_REF, nasa9, "turbine")
        assert got[0] == pytest.approx(want[0], rel=1e-12)
        assert got[1] == pytest.approx(want[1], rel=1e-12)

    def test_the_perfect_gas_route_is_untouched(self):
        """``CW·θ`` stays exact where it is exact, so the perfect-gas library
        results are not being quietly re-derived by this change."""
        m = load_beta_map(COMPRESSOR, PERFECT)
        point = m.evaluate_at_beta(0.5, 1.0)
        T02, p02, dh0 = _exit_from_point(point, 400.0, 101325.0, 400.0 / T_REF, PERFECT,
                                         "compressor")
        assert dh0 == pytest.approx(point.corrected_work * 400.0 / T_REF, rel=1e-15)
        assert T02 == pytest.approx(400.0 + dh0 / PERFECT.cp, rel=1e-15)
        assert p02 == pytest.approx(point.PR * 101325.0, rel=1e-15)

    def test_the_two_routes_would_disagree_if_the_gas_were_real_and_theta_were_not_one(
        self, nasa9
    ):
        """The premise. Without this the parametrised tests above could pass by
        both routes having quietly become the same code."""
        m = load_beta_map(COMPRESSOR, nasa9)
        point = m.evaluate_at_beta(0.5, 1.0)
        T01 = 1600.0
        naive = T01 + point.corrected_work * (T01 / T_REF) / nasa9.cp_at(T_REF)
        real = _exit_from_point(point, T01, 101325.0, T01 / T_REF, nasa9, "compressor")[0]
        assert abs(naive - real) > 20.0, "premise gone: CW*theta now agrees at theta=5.55"


# ---------------------------------------------------------------------------
# The key the disk measures must be in the units the map's axis is built in
# ---------------------------------------------------------------------------


class TestTheMeasuredKeyIsCorrectedToTheReference:
    """`ECMF = Wc·√τ/PR` with `τ` at ``T_ref``. A disk reads ``T₀₂/T₀₁`` off the
    field at the running temperature. Those coincide only at constant ``cp``."""

    @staticmethod
    def _map_tau(point, gas):
        """``τ`` exactly as the loader derived it for the key axis."""
        if hasattr(gas, "cp"):
            return 1.0 + point.corrected_work / (gas.cp * T_REF)
        return gas.temperature_from_enthalpy(
            gas.enthalpy(T_REF) + point.corrected_work
        ) / T_REF

    @staticmethod
    def _at_a_grid_node(m, i=10, j=1):
        """A point where ``PR``, ``η`` and ``CW`` are the loader's own triple.

        Off a node they are interpolated *independently*, so they stop being
        mutually consistent and ``τ`` from ``CW`` no longer equals ``τ`` from
        ``PR``/``η`` — by 8.8e-06 on the raw compressor map, which is the map's
        resolution and has nothing to do with what is being tested here.
        """
        return m.evaluate_at_beta(float(m.beta[i]), float(m.corrected_speed[j]))

    def test_a_compressor_measurement_corrects_back_to_the_map(self, nasa9):
        m = load_beta_map(COMPRESSOR, nasa9)
        point = self._at_a_grid_node(m)
        T02, p02, _ = exit_stagnation_from_map(
            900.0, 4.0e5, point.PR, point.efficiency, nasa9, "compressor"
        )
        got = reference_tau(900.0, T02, p02 / 4.0e5, nasa9, T_REF)
        assert got == pytest.approx(self._map_tau(point, nasa9), rel=1e-10)

    def test_a_turbine_measurement_corrects_back_to_the_map(self, nasa9, tmp_path):
        m = load_beta_map(turbine_workbook(tmp_path / "t.xlsx"), nasa9)
        point = self._at_a_grid_node(m, i=5, j=2)
        T02, p02, _ = exit_stagnation_from_map(
            1600.0, 1.2e6, point.PR, point.efficiency, nasa9, "turbine"
        )
        got = reference_tau(1600.0, T02, p02 / 1.2e6, nasa9, T_REF)
        assert got == pytest.approx(self._map_tau(point, nasa9), rel=1e-10)

    @pytest.mark.parametrize("kind,pr,p01", [("compressor", 2.2, 101325.0),
                                             ("turbine", 2.5, 1.2e6)])
    def test_the_corrected_reading_does_not_depend_on_where_it_was_taken(
        self, nasa9, kind, pr, p01
    ):
        """The similarity statement itself, and the one that does not lean on a
        map at all: the same machine read at 300 K, 900 K and 1600 K must give
        one corrected ``τ``. That is what makes a static table indexable."""
        taus = []
        for T01 in (300.0, 900.0, 1600.0):
            T02, p02, _ = exit_stagnation_from_map(T01, p01, pr, 0.85, nasa9, kind)
            taus.append(reference_tau(T01, T02, p02 / p01, nasa9, T_REF))
        assert taus[1] == pytest.approx(taus[0], rel=1e-10)
        assert taus[2] == pytest.approx(taus[0], rel=1e-10)

    @staticmethod
    def _design_key(d, gas):
        """The key the disk forms, evaluated at the design state exactly.

        The duct was *sized* from that state, so this has to return the design
        ECMF. It is the whole closure in one line and needs no marching, which
        makes it the cheapest place to catch anything that puts the disk and the
        table in different units.
        """
        theta, delta = d.T01 / T_REF, d.p01 / P_REF
        pr = d.p02 / d.p01
        tau = _reference_tau(d.T01, d.T02, pr, d.T02 / d.T01, gas)
        return (d.W * math.sqrt(theta) / delta) * math.sqrt(tau) / pr

    @pytest.mark.parametrize("gasname", ["nasa9", "perfect"])
    def test_a_turbine_design_state_returns_its_own_design_key(
        self, nasa9, tmp_path, gasname
    ):
        gas = nasa9 if gasname == "nasa9" else PERFECT
        m = load_beta_map(turbine_workbook(tmp_path / "t.xlsx"), gas).densify(9)
        line = m._speed_line(0.9)[0]
        lo, hi = float(line.min()), float(line.max())
        d = design_from_map(m, 0.9, 0.5 * (lo + hi), inlet_mach=0.25,
                            T01=1600.0, p01=1.2e6)
        assert self._design_key(d, gas) == pytest.approx(d.point.ecmf, rel=5e-06)

    def test_a_compressor_design_state_returns_its_own_design_key(self, nasa9):
        m = load_beta_map(COMPRESSOR, nasa9).densify(9)
        line = m._speed_line(1.0)[0]
        lo, hi = float(line.min()), float(line.max())
        d = design_from_map(m, 1.0, lo + 0.35 * (hi - lo), inlet_mach=0.45)
        assert self._design_key(d, nasa9) == pytest.approx(d.point.ecmf, rel=5e-06)

    def test_the_uncorrected_reading_really_is_wrong_at_turbine_entry(self, nasa9,
                                                                     tmp_path):
        """The premise, stated as a number: without this correction the key is
        off by more than a percent, which is four orders past the 1e-6 gate."""
        m = load_beta_map(turbine_workbook(tmp_path / "t.xlsx"), nasa9)
        point = self._at_a_grid_node(m, i=5, j=2)
        T02, p02, _ = exit_stagnation_from_map(
            1600.0, 1.2e6, point.PR, point.efficiency, nasa9, "turbine"
        )
        raw = T02 / 1600.0
        assert abs(raw / self._map_tau(point, nasa9) - 1.0) > 0.01

    def test_a_perfect_gas_needs_no_correction(self, tmp_path):
        """``τ`` is a similarity invariant there, so the correction is identity
        and the perfect-gas key is what it always was."""
        m = load_beta_map(turbine_workbook(tmp_path / "t.xlsx"), PERFECT)
        point = self._at_a_grid_node(m, i=5, j=2)
        T02, p02, _ = exit_stagnation_from_map(
            1600.0, 1.2e6, point.PR, point.efficiency, PERFECT, "turbine"
        )
        assert T02 / 1600.0 == pytest.approx(self._map_tau(point, PERFECT), rel=1e-12)


# ---------------------------------------------------------------------------
# Map persistence
# ---------------------------------------------------------------------------


class TestASavedMapRemembersItsGas:
    """`ECMFMap.save` exists so a solver run does not rebuild a static table.
    A real-gas table is derived with that gas's ``h`` and ``s°`` throughout, so
    what comes back has to be the same gas — storing ``gamma``/``cp`` only gave
    a ``PerfectGas(nan, nan)``, which is the silent substitution the load-time
    check is supposed to prevent."""

    def test_a_real_gas_table_round_trips(self, nasa9, tmp_path):
        built = ECMFMap.from_beta_map(load_beta_map(COMPRESSOR, nasa9).densify(9))
        back = ECMFMap.load(built.save(tmp_path / "m.npz"))
        assert type(back.gas).__name__ == "Nasa9Gas"
        assert back.gas.R == pytest.approx(nasa9.R, rel=1e-15)
        assert back.gas.cp_at(1600.0) == pytest.approx(nasa9.cp_at(1600.0), rel=1e-15)
        key, nc = float(built.key[5, 1]), float(built.corrected_speed[1])
        assert back.evaluate(key, nc) == built.evaluate(key, nc)

    def test_loading_a_real_gas_table_needs_no_mechanism_file(self, nasa9, tmp_path):
        """Rebuilt from the stored blocks, not from the name — so the table does
        not depend on Cantera or on ``airNASA9.yaml`` still being there."""
        built = ECMFMap.from_beta_map(load_beta_map(COMPRESSOR, nasa9).densify(9))
        back = ECMFMap.load(built.save(tmp_path / "m.npz"))
        assert back.gas.regions == nasa9.regions
        assert back.gas.weights == nasa9.weights

    def test_an_equal_gas_is_accepted_and_a_different_one_refused(self, nasa9, tmp_path):
        built = ECMFMap.from_beta_map(load_beta_map(COMPRESSOR, nasa9).densify(9))
        path = built.save(tmp_path / "m.npz")
        ECMFMap.load(path, Nasa9Gas.from_cantera())  # equal coefficients, new object
        with pytest.raises(ValueError, match="already carry the first"):
            ECMFMap.load(path, PERFECT)

    def test_a_perfect_gas_table_still_refuses_a_real_one(self, nasa9, tmp_path):
        built = ECMFMap.from_beta_map(load_beta_map(COMPRESSOR, PERFECT).densify(9))
        path = built.save(tmp_path / "m.npz")
        assert type(ECMFMap.load(path).gas).__name__ == "PerfectGas"
        with pytest.raises(ValueError, match="already carry the first"):
            ECMFMap.load(path, nasa9)


# ---------------------------------------------------------------------------
# The inlet boundary condition
# ---------------------------------------------------------------------------


def _uniform_duct(gas, T0, p0, M1, ncell=101, area=0.3):
    """A plain duct with no disk, seeded at the exact uniform state for ``M1``."""
    T = T0
    for _ in range(200):
        u = M1 * gas.speed_of_sound(T)
        nxt = gas.temperature_from_enthalpy(gas.enthalpy(T0) - 0.5 * u * u)
        if abs(nxt - T) < 1e-14 * T0:
            T = nxt
            break
        T = nxt
    u = M1 * gas.speed_of_sound(T)
    p = p0 * gas.pressure_ratio_isentropic(T0, T)
    rho = p / (gas.R * T)
    grid = Grid.uniform(0.0, 1.0, ncell, area)
    s = Solver(
        grid, gas,
        StagnationInletStaticOutlet(p0_in=p0, T0_in=T0, p_back=p),
        ReferenceState(rho=rho, u=u, p=p),
        config=SolverConfig(cfl=2.0),
    )
    s.set_state(rho, u, p)
    return s, rho * u * area


class TestTheInletDeliversWhatItWasGiven:
    """A constant-area duct at a uniform state whose stagnation condition *is*
    the imposed one must stay there. That is a fixed-point property of the
    boundary condition, not a convergence property of the solver, so it holds to
    round-off or it does not hold at all."""

    @pytest.mark.parametrize("T0,p0", [(288.15, 101325.0), (1600.0, 1.2e6)])
    def test_a_uniform_duct_holds_its_stagnation_state(self, nasa9, T0, p0):
        s, _ = _uniform_duct(nasa9, T0, p0, 0.45)
        s.run(max_steps=3000, tol=1e-12)
        for cell in (2, 50, 98):
            st = s.station_state_at(cell)
            T0m, p0m = stagnation_from_static(
                st.p / (st.rho * nasa9.R), st.p, st.u, nasa9
            )
            assert T0m == pytest.approx(T0, rel=1e-9), f"T0 drifted at cell {cell}"
            assert p0m == pytest.approx(p0, rel=1e-8), f"p0 drifted at cell {cell}"

    def test_the_perfect_gas_still_does_too(self):
        s, _ = _uniform_duct(PERFECT, 288.15, 101325.0, 0.45)
        s.run(max_steps=3000, tol=1e-12)
        st = s.station_state_at(50)
        T0m, p0m = stagnation_from_static(
            st.p / (st.rho * PERFECT.R), st.p, st.u, PERFECT
        )
        assert T0m == pytest.approx(288.15, rel=1e-11)
        assert p0m == pytest.approx(101325.0, rel=1e-10)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


#: Densification of the β axis before the ECMF key is built. 36, not the 9 the
#: β-path tests use: at 9 the *interpolation* floor is ~2e-06 on both gases
#: (measured: perfect +1.82e-06, nasa9 +1.65e-06 on the same case), which sits
#: on top of the 1e-6 gate and would grade the real gas on the map's resolution
#: rather than on its thermodynamics. §3.42 sets 36.
DENSIFY = 36


def _march(path, gas, speed, frac, M1, T01, p01, *, ncell=201, n_smear=7,
           steps=30000, tol=1e-11):
    """Size a duct from a map point, seed its steady profile, hold it there."""
    bm = load_beta_map(path, gas).densify(DENSIFY)
    em = ECMFMap.from_beta_map(bm)
    line = bm._speed_line(speed)[0]
    lo, hi = float(line.min()), float(line.max())
    d = design_from_map(bm, speed, lo + frac * (hi - lo),
                        inlet_mach=M1, T01=T01, p01=p01)
    A = d.area
    grid = Grid.uniform(0.0, 1.0, ncell, A)
    cell = (ncell - n_smear) // 2
    disk = EcmfCompressor(cell=cell, ecmf_map=em, corrected_speed=speed,
                          sample_offset=2, exit_offset=3, n_smear=n_smear,
                          inlet_lag=1e-2)
    s = Solver(
        grid, gas,
        StagnationInletStaticOutlet(p0_in=d.p01, T0_in=d.T01, p_back=d.p_back),
        ReferenceState(rho=d.station1.rho, u=d.station1.u, p=d.station1.p),
        config=SolverConfig(cfl=2.0), source=disk,
    )
    rho, u, p = np.empty(ncell), np.empty(ncell), np.empty(ncell)
    rho[:cell], u[:cell], p[:cell] = d.station1.rho, d.station1.u, d.station1.p
    rho[cell + n_smear:] = d.station2.rho
    u[cell + n_smear:] = d.station2.u
    p[cell + n_smear:] = d.station2.p
    F = np.array([
        d.station1.rho * d.station1.u * A,
        (d.station1.rho * d.station1.u**2 + d.station1.p) * A,
        d.station1.rho * d.station1.u
        * (gas.enthalpy(d.station1.T) + 0.5 * d.station1.u**2) * A,
    ])
    q = np.array([0.0, d.Fx, d.SWx]) / n_smear
    for k in range(n_smear):
        x = state_from_flux(F + 0.5 * q, A, gas)
        rho[cell + k], u[cell + k], p[cell + k] = x.rho, x.u, x.p
        F = F + q
    s.set_state(rho, u, p)

    for k in range(steps):
        try:
            s.advance(s.timestep())
        except (NonPhysicalState, ValueError) as ex:  # pragma: no cover
            pytest.fail(f"died at step {k}: {ex}")
        if k % 250 == 0 and k > 0 and \
                float(s.residual_norm()[0]) / d.W * s.grid.dx.mean() < tol:
            break
    return float(s.face_fluxes()[0].mean()) / d.W - 1.0, d


@pytest.mark.slow
class TestTheMachinesRunOnIt:
    """The whole chain on the real gas: workbook -> map -> design -> march. The
    gate is the same 1e-6 hold the perfect-gas library sweep is measured against,
    so a real-gas run is not being graded on an easier curve."""

    def test_a_compressor_at_standard_day(self, nasa9):
        err, d = _march(COMPRESSOR, nasa9, 1.0, 0.35, 0.45, 288.15, 101325.0)
        assert abs(err) < HOLD, f"PR {d.point.PR:.4g} held to {err:+.3e}"

    def test_a_turbine_at_turbine_entry_conditions(self, nasa9, tmp_path):
        """1600 K and 1200 kPa — where ``cp`` is 21.7% off its standard-day
        value, and where ``θ = 5.55`` makes the corrected-work scaling wrong."""
        path = turbine_workbook(tmp_path / "t.xlsx")
        err, d = _march(path, nasa9, 0.9, 0.5, 0.25, 1600.0, 1.2e6)
        assert d.T02 < d.T01, "a turbine must cool"
        assert abs(err) < HOLD, f"PR {d.point.PR:.4g} held to {err:+.3e}"

    def test_the_perfect_gas_holds_the_same_compressor(self):
        """The control. Same machinery, same gate, the gas that was already
        validated — so a failure above is about the real gas and not the rig."""
        err, _ = _march(COMPRESSOR, PERFECT, 1.0, 0.35, 0.45, 288.15, 101325.0)
        assert abs(err) < HOLD, f"held to {err:+.3e}"
