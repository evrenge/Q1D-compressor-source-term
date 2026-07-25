import pytest

from q1d.gas import AIR_LEGACY, PerfectGas


def test_R_is_derived_from_gamma_and_cp():
    # PLAN.md §3.1: cp=1005, gamma=1.4 gives R=287.1429, NOT standard-air 287.05.
    assert AIR_LEGACY.R == pytest.approx(287.14285714285705, rel=1e-15)
    assert AIR_LEGACY.cv == pytest.approx(1005.0 - 287.14285714285705, rel=1e-15)


def test_mayer_relation_holds():
    for gamma, cp in [(1.4, 1005.0), (1.33, 1150.0), (1.667, 5193.0)]:
        gas = PerfectGas(gamma=gamma, cp=cp)
        assert gas.cp - gas.cv == pytest.approx(gas.R, rel=1e-14)
        assert gas.cp / gas.cv == pytest.approx(gamma, rel=1e-14)


def test_exponent_helpers():
    gas = AIR_LEGACY
    assert gas.g_over_gm1 == pytest.approx(1.4 / 0.4)
    assert gas.gm1_over_g == pytest.approx(0.4 / 1.4)
    assert gas.g_over_gm1 * gas.gm1_over_g == pytest.approx(1.0, rel=1e-15)


def test_enthalpy_roundtrip():
    gas = AIR_LEGACY
    for T in [200.0, 288.15, 1600.0]:
        assert gas.temperature_from_enthalpy(gas.enthalpy(T)) == pytest.approx(T, rel=1e-15)


def test_speed_of_sound():
    # 288.15 K with R=287.1429 -> not quite the textbook 340.3 for standard air
    assert AIR_LEGACY.speed_of_sound(288.15) == pytest.approx(340.35, abs=0.05)
