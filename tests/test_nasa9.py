"""Gate for the NASA9 real gas.

The reason it exists: on air ``cp`` rises **21.7%** between 288 K and 1600 K and
``γ`` falls from 1.3988 to 1.3061. A compressor at standard-day inlet barely
notices, which is why the perfect-gas results elsewhere in this project stand. A
turbine runs at turbine-entry conditions, and §3.39 puts them at 1600 K — right
in the range where a calorically perfect gas is simply the wrong model.

These check the polynomials against Cantera rather than against themselves,
because the whole point of using NASA9 is that the coefficients are somebody
else's authority and not ours.
"""

import math

import pytest

from q1d.gas import Nasa9Gas, PerfectGas

ct = pytest.importorskip("cantera")

GAS = PerfectGas(1.4, 1004.7)


@pytest.fixture(scope="module")
def nasa9():
    return Nasa9Gas.from_cantera()


@pytest.fixture(scope="module")
def cantera_air():
    s = ct.Solution("airNASA9.yaml")
    return s


T_SAMPLES = [200.0, 288.15, 500.0, 800.0, 1600.0, 2500.0, 3000.0, 5000.0]


class TestAgainstCantera:
    """The coefficients are Cantera's; the evaluation is ours. Check the join."""

    @pytest.mark.parametrize("T", T_SAMPLES)
    def test_cp_matches(self, nasa9, cantera_air, T):
        cantera_air.TPX = T, ct.one_atm, {"N2": 0.79, "O2": 0.21}
        assert nasa9.cp_at(T) == pytest.approx(cantera_air.cp_mass, rel=1e-8)

    @pytest.mark.parametrize("T", T_SAMPLES)
    def test_enthalpy_matches(self, nasa9, cantera_air, T):
        cantera_air.TPX = T, ct.one_atm, {"N2": 0.79, "O2": 0.21}
        assert nasa9.enthalpy(T) == pytest.approx(cantera_air.enthalpy_mass, rel=1e-8)

    def test_the_gas_constant_matches(self, nasa9, cantera_air):
        cantera_air.TPX = 300.0, ct.one_atm, {"N2": 0.79, "O2": 0.21}
        assert nasa9.R == pytest.approx(
            ct.gas_constant / cantera_air.mean_molecular_weight, rel=1e-12
        )

    @pytest.mark.parametrize("T", T_SAMPLES)
    def test_entropy_difference_matches(self, nasa9, cantera_air, T):
        """``s°`` carries an arbitrary constant, so compare *differences*.

        Only differences are ever used — an isentrope is
        ``s°(T₂) − s°(T₁) = R·ln(p₂/p₁)`` — so pinning the absolute value would
        pin something the model does not depend on.
        """
        ref = 300.0
        cantera_air.TPX = T, ct.one_atm, {"N2": 0.79, "O2": 0.21}
        s_hi = cantera_air.entropy_mass
        cantera_air.TPX = ref, ct.one_atm, {"N2": 0.79, "O2": 0.21}
        s_lo = cantera_air.entropy_mass
        mine = nasa9.entropy_ref(T) - nasa9.entropy_ref(ref)
        assert mine == pytest.approx(s_hi - s_lo, rel=1e-8, abs=1e-9)


class TestItIsNotAPerfectGas:
    def test_cp_rises_with_temperature(self, nasa9):
        assert nasa9.cp_at(1600.0) / nasa9.cp_at(288.15) == pytest.approx(1.217, abs=0.005)

    def test_gamma_falls_with_temperature(self, nasa9):
        assert nasa9.gamma_at(288.15) == pytest.approx(1.3988, abs=1e-3)
        assert nasa9.gamma_at(1600.0) == pytest.approx(1.3061, abs=1e-3)

    def test_the_power_law_is_wrong_by_a_useful_amount(self, nasa9):
        """How much the constant-γ isentrope misses by, at turbine conditions.

        This is the error the perfect gas carries, and the reason for the class.
        """
        T1, T2 = 1600.0, 1100.0
        exact = nasa9.pressure_ratio_isentropic(T1, T2)
        g = nasa9.gamma_at(T1)
        naive = (T2 / T1) ** (g / (g - 1.0))
        assert abs(naive / exact - 1.0) > 0.02, "premise gone: the two now agree"


class TestRoundTrips:
    @pytest.mark.parametrize("T", [250.0, 288.15, 1234.5, 1600.0, 2800.0])
    def test_enthalpy_inverts(self, nasa9, T):
        assert nasa9.temperature_from_enthalpy(nasa9.enthalpy(T)) == pytest.approx(T, rel=1e-10)

    @pytest.mark.parametrize("T1,T2", [(288.15, 700.0), (1600.0, 900.0), (400.0, 401.0)])
    def test_the_isentrope_inverts(self, nasa9, T1, T2):
        pr = nasa9.pressure_ratio_isentropic(T1, T2)
        assert nasa9.temperature_isentropic(T1, pr) == pytest.approx(T2, rel=1e-10)

    def test_an_isentrope_of_unit_ratio_does_nothing(self, nasa9):
        assert nasa9.pressure_ratio_isentropic(700.0, 700.0) == pytest.approx(1.0, abs=1e-14)
        assert nasa9.temperature_isentropic(700.0, 1.0) == pytest.approx(700.0, rel=1e-12)


class TestTheInterfaceIsShared:
    """A call site written against the temperature-aware accessors takes either
    gas unchanged. That is what makes the migration possible at all."""

    @pytest.mark.parametrize("T", [288.15, 900.0])
    def test_perfect_gas_answers_the_same_questions(self, T):
        assert GAS.cp_at(T) == GAS.cp
        assert GAS.gamma_at(T) == GAS.gamma

    def test_perfect_gas_isentrope_matches_its_own_power_law(self):
        pr = GAS.pressure_ratio_isentropic(288.15, 400.0)
        assert pr == pytest.approx((400.0 / 288.15) ** GAS.g_over_gm1, rel=1e-15)
        assert GAS.temperature_isentropic(288.15, pr) == pytest.approx(400.0, rel=1e-12)

    def test_perfect_gas_entropy_reproduces_its_own_isentrope(self):
        """``s° = cp·ln T`` must give back the power law, or the two paths differ."""
        T1, T2 = 288.15, 640.0
        by_entropy = math.exp((GAS.entropy_ref(T2) - GAS.entropy_ref(T1)) / GAS.R)
        assert by_entropy == pytest.approx(GAS.pressure_ratio_isentropic(T1, T2), rel=1e-12)


class TestItRefusesToPretendToBeConstant:
    @pytest.mark.parametrize("attr", ["gamma", "cp", "g_over_gm1", "gm1", "cv"])
    def test_scalar_thermo_attributes_are_absent(self, nasa9, attr):
        """Silently returning a reference value is how a 20% error gets shipped."""
        with pytest.raises(AttributeError, match="varies with temperature"):
            getattr(nasa9, attr)


class TestConstruction:
    def test_a_nasa7_mechanism_is_refused(self):
        """``air.yaml`` is NASA7. The formats differ in coefficient count and in
        the low-temperature term, so accepting one as the other is wrong rather
        than approximate."""
        with pytest.raises(ValueError, match="not NASA9"):
            Nasa9Gas.from_cantera("air.yaml", {"N2": 0.79, "O2": 0.21})

    def test_composition_is_renormalised(self):
        a = Nasa9Gas.from_cantera(composition={"N2": 79.0, "O2": 21.0})
        b = Nasa9Gas.from_cantera(composition={"N2": 0.79, "O2": 0.21})
        assert a.R == pytest.approx(b.R, rel=1e-14)
        assert a.cp_at(1000.0) == pytest.approx(b.cp_at(1000.0), rel=1e-14)

    def test_an_empty_composition_is_refused(self):
        with pytest.raises(ValueError, match="positive total"):
            Nasa9Gas.from_cantera(composition={"N2": 0.0})
