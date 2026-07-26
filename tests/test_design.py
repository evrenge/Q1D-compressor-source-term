"""Designing a duct from a map, and the feasibility report that precedes it."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from q1d.analytic import InfeasibleOperatingPoint, max_flow_function
from q1d.design import CHOKE_MARGIN, design_from_map, reachable_ecmf_range
from q1d.gas import PerfectGas
from q1d.maps import P_REF, T_REF, load_beta_map

DATA = Path(__file__).resolve().parents[1] / "data" / "maps"
GAS = PerfectGas(1.4, 1004.7)


def _maps():
    return sorted(DATA.glob("*.xlsx"))


@pytest.fixture(params=_maps(), ids=lambda p: p.stem)
def real_map(request):
    if not _maps():  # pragma: no cover - data-dependent
        pytest.skip("no map workbooks in data/maps")
    return load_beta_map(request.param, GAS).densify(9)


def _mid_ecmf(m, n):
    e = m._speed_line(n)[0]
    return float(0.5 * (e.min() + e.max()))


def test_design_reproduces_the_map_point_it_was_built_from(real_map):
    """The design is only useful if it *is* the map point, to round-off."""
    for n in real_map.corrected_speed[::9]:
        n = float(n)
        d = design_from_map(real_map, n, _mid_ecmf(real_map, n))
        assert d.point.Wc == pytest.approx(d.W, rel=1e-12)  # standard day: Wc == W
        assert d.p02 / d.p01 == pytest.approx(d.point.PR, rel=1e-12)
        assert (d.T02 - d.T01) * GAS.cp == pytest.approx(d.point.corrected_work, rel=1e-12)


def test_design_is_a_consistent_steady_state(real_map):
    """Mass, momentum and energy must balance across the two stations."""
    n = float(real_map.corrected_speed[len(real_map.corrected_speed) // 2])
    d = design_from_map(real_map, n, _mid_ecmf(real_map, n))
    A = d.area
    st1, st2 = d.station1, d.station2
    assert st1.rho * st1.u * A == pytest.approx(d.W, rel=1e-12)
    assert st2.rho * st2.u * A == pytest.approx(d.W, rel=1e-12)
    # momentum flux jump == Fx, energy flux jump == SWx: the identity the
    # source term exists to impose.
    dmom = (st2.rho * st2.u**2 + st2.p) * A - (st1.rho * st1.u**2 + st1.p) * A
    dene = d.W * (GAS.cp * st2.T + 0.5 * st2.u**2) - d.W * (GAS.cp * st1.T + 0.5 * st1.u**2)
    assert dmom == pytest.approx(d.Fx, rel=1e-10)
    assert dene == pytest.approx(d.SWx, rel=1e-10)


def test_back_pressure_is_the_exit_static_pressure(real_map):
    n = float(real_map.corrected_speed[0])
    d = design_from_map(real_map, n, _mid_ecmf(real_map, n))
    assert d.p_back == pytest.approx(d.station2.p, rel=0.0, abs=0.0)
    assert d.p_back < d.p02  # static below stagnation, by definition


def test_inlet_mach_is_honoured_exactly(real_map):
    n = float(real_map.corrected_speed[-1])
    for M1 in (0.2, 0.35, 0.45, 0.6):
        d = design_from_map(real_map, n, _mid_ecmf(real_map, n), inlet_mach=M1)
        assert d.station1.M == pytest.approx(M1, rel=1e-10)


def test_exit_is_subsonic_on_every_speed_line(real_map):
    """A compressor exit above Mach 1 is not a valid reading of these maps."""
    for n in real_map.corrected_speed[::9]:
        d = design_from_map(real_map, float(n), _mid_ecmf(real_map, float(n)))
        assert d.station2.M < 1.0
        assert d.station1.M < 1.0


def test_exit_temperature_stays_in_the_physical_range(real_map):
    """Compressors heat the flow to ~900 K at most; anything beyond is a bug."""
    for n in real_map.corrected_speed[::9]:
        d = design_from_map(real_map, float(n), _mid_ecmf(real_map, float(n)))
        assert T_REF < d.T02 < 900.0


def test_area_scales_with_mass_flow_at_fixed_inlet_mach(real_map):
    """A = W sqrt(R T01) / (p01 Phi(M1)) — linear in W, nothing else."""
    from q1d.analytic import flow_function

    n = float(real_map.corrected_speed[len(real_map.corrected_speed) // 2])
    d = design_from_map(real_map, n, _mid_ecmf(real_map, n), inlet_mach=0.45)
    expect = d.W * math.sqrt(GAS.R * d.T01) / (d.p01 * flow_function(0.45, GAS))
    assert d.area == pytest.approx(expect, rel=1e-12)


def test_off_standard_day_scales_the_mass_flow_but_not_the_map_point(real_map):
    """Wc is the invariant; W follows delta/sqrt(theta)."""
    n = float(real_map.corrected_speed[len(real_map.corrected_speed) // 2])
    e = _mid_ecmf(real_map, n)
    ref = design_from_map(real_map, n, e)
    hot = design_from_map(real_map, n, e, T01=330.0, p01=80_000.0)
    theta, delta = 330.0 / T_REF, 80_000.0 / P_REF
    assert hot.point.Wc == pytest.approx(ref.point.Wc, rel=1e-12)
    assert hot.W == pytest.approx(ref.W * delta / math.sqrt(theta), rel=1e-12)
    assert hot.station1.M == pytest.approx(ref.station1.M, rel=1e-10)


def test_a_high_inlet_mach_is_reported_not_silently_accepted(real_map):
    """The *inlet* is the limiting station at equal areas (PLAN.md Phase 1)."""
    n = float(real_map.corrected_speed[0])
    with pytest.raises(InfeasibleOperatingPoint, match="inlet station would sit"):
        design_from_map(real_map, n, _mid_ecmf(real_map, n), inlet_mach=0.99)


def test_the_exit_never_chokes_first_at_equal_areas(real_map):
    """PR/sqrt(tau) > 1 everywhere on these maps, so Phi2 < Phi1 always."""
    for n in real_map.corrected_speed[::9]:
        n = float(n)
        d = design_from_map(real_map, n, _mid_ecmf(real_map, n), inlet_mach=0.6)
        assert d.station2.M < d.station1.M
        assert d.point.PR / math.sqrt(d.T02 / d.T01) > 1.0


def test_design_rejects_nonsense_inputs(real_map):
    n = float(real_map.corrected_speed[0])
    e = _mid_ecmf(real_map, n)
    for M1 in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="inlet_mach"):
            design_from_map(real_map, n, e, inlet_mach=M1)
    with pytest.raises(ValueError, match="positive"):
        design_from_map(real_map, n, e, T01=-1.0)


def test_reachable_range_contains_the_point_the_area_was_sized_for(real_map):
    n = float(real_map.corrected_speed[len(real_map.corrected_speed) // 2])
    e = _mid_ecmf(real_map, n)
    d = design_from_map(real_map, n, e, inlet_mach=0.45)
    lo, hi = reachable_ecmf_range(real_map, n, d.area)
    assert lo <= d.point.ecmf <= hi


def test_reachable_range_shrinks_as_the_duct_does(real_map):
    n = float(real_map.corrected_speed[len(real_map.corrected_speed) // 2])
    d = design_from_map(real_map, n, _mid_ecmf(real_map, n), inlet_mach=0.45)
    wide = reachable_ecmf_range(real_map, n, d.area)
    tight = reachable_ecmf_range(real_map, n, d.area * 0.75)
    assert (tight[1] - tight[0]) <= (wide[1] - wide[0])


def test_reachable_range_reports_an_impossible_area(real_map):
    n = float(real_map.corrected_speed[-1])
    d = design_from_map(real_map, n, _mid_ecmf(real_map, n))
    with pytest.raises(InfeasibleOperatingPoint, match="too small for this speed line"):
        reachable_ecmf_range(real_map, n, d.area * 0.05)


def test_reachable_points_really_are_below_the_choke_limit(real_map):
    """Spot-check the range's own claim at both ends."""
    n = float(real_map.corrected_speed[len(real_map.corrected_speed) // 2])
    d = design_from_map(real_map, n, _mid_ecmf(real_map, n))
    lim = max_flow_function(GAS) * CHOKE_MARGIN
    for target in reachable_ecmf_range(real_map, n, d.area):
        pt = real_map.evaluate_at_ecmf(target, n)
        W = pt.Wc  # standard day: W == Wc
        T02 = T_REF + pt.corrected_work / GAS.cp
        phi1 = W * math.sqrt(GAS.R * T_REF) / (d.area * P_REF)
        phi2 = W * math.sqrt(GAS.R * T02) / (d.area * pt.PR * P_REF)
        assert phi1 < lim and phi2 < lim


def test_describe_names_the_operating_point(real_map):
    n = float(real_map.corrected_speed[0])
    text = design_from_map(real_map, n, _mid_ecmf(real_map, n)).describe()
    for token in ("Nc=", "ECMF=", "PR=", "M1=", "M2=", "Fx=", "SWx="):
        assert token in text


# -- the inlet closure ------------------------------------------------------


def test_inlet_Wc_lookup_is_the_inverse_of_the_beta_lookup(real_map):
    """Wc -> beta -> Wc must round-trip wherever the closure is usable."""
    for n in real_map.corrected_speed[::9]:
        n = float(n)
        if not real_map.inlet_closure_is_invertible(n):
            continue
        wc = real_map._speed_line(n)[1]
        for target in np.linspace(wc.min(), wc.max(), 25):
            got = real_map.evaluate_at_Wc(float(target), n)
            assert got.Wc == pytest.approx(float(target), rel=1e-9)


def test_inlet_closure_refuses_a_vertical_speed_line(real_map):
    """Where Wc is not monotonic no inlet-only closure can pick a point."""
    for n in real_map.corrected_speed:
        n = float(n)
        if real_map.inlet_closure_is_invertible(n):
            continue
        with pytest.raises(ValueError, match="not monotonic"):
            real_map.evaluate_at_Wc(1.0, n)
        break
    else:
        pytest.skip(f"{real_map.name}: every speed line is invertible")


def test_inlet_closure_availability_matches_the_measured_split(real_map):
    """SubsonicCompressor is invertible throughout; Transsonic is not above 88%."""
    bad = [
        float(n) for n in real_map.corrected_speed if not real_map.inlet_closure_is_invertible(n)
    ]
    if real_map.name.startswith("Subsonic"):
        assert bad == []
    elif real_map.name.startswith("Transsonic"):
        # Raw data loses Wc monotonicity at 0.88; the densified lines between
        # 0.791 (monotonic) and 0.88 (not) inherit it, so the boundary moves
        # down to 0.80. That is interpolation between a good line and a bad
        # one, not a defect of the refinement.
        assert bad and 0.79 < min(bad) <= 0.88
