"""Map loading, ECMF derivation and (β, Nc) densification.

The workbooks live in ``data/maps``; tests that need them skip when absent so
the suite still runs on a checkout without the data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from q1d.gas import PerfectGas
from q1d.maps import P_REF, T_REF, BetaMap, load_beta_map

DATA = Path(__file__).resolve().parents[1] / "data" / "maps"
GAS = PerfectGas(1.4, 1004.7)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _maps():
    return sorted(DATA.glob("*.xlsx"))


def _load(path):
    return load_beta_map(path, GAS)


@pytest.fixture(params=_maps(), ids=lambda p: p.stem)
def real_map(request):
    if not _maps():  # pragma: no cover - data-dependent
        pytest.skip("no map workbooks in data/maps")
    return _load(request.param)


def _synthetic(nb=5, ns=4) -> BetaMap:
    """A smooth, strictly monotone stand-in that needs no workbook."""
    beta = np.linspace(0.0, 1.0, nb)
    speed = np.linspace(0.6, 1.05, ns)
    b, s = np.meshgrid(beta, speed, indexing="ij")
    Wc = 20.0 * s * (1.0 + 0.15 * b)
    PR = 1.0 + 4.0 * s**2 * (1.0 - 0.25 * b)
    eff = 0.80 + 0.06 * b * (1.0 - b) - 0.05 * (s - 0.9) ** 2
    dh0s = GAS.cp * T_REF * (PR**GAS.gm1_over_g - 1.0)
    cw = dh0s / eff
    tau = 1.0 + cw / (GAS.cp * T_REF)
    return BetaMap(
        name="synthetic",
        beta=beta,
        corrected_speed=speed,
        Wc=Wc,
        PR=PR,
        efficiency=eff,
        corrected_work=cw,
        ecmf=Wc * np.sqrt(tau) / PR,
        gas=GAS,
    )


# -- loading ----------------------------------------------------------------


def test_loaded_map_shapes_and_axes(real_map):
    n = (len(real_map.beta), len(real_map.corrected_speed))
    for f in ("Wc", "PR", "efficiency", "corrected_work", "ecmf"):
        assert getattr(real_map, f).shape == n
    assert np.all(np.diff(real_map.beta) > 0)
    assert np.all(np.diff(real_map.corrected_speed) > 0)


def test_corrected_work_round_trips_through_efficiency(real_map):
    """``CW = Δh₀ₛ/η`` must be recoverable — the identity densify() relies on."""
    dh0s = GAS.cp * T_REF * (real_map.PR**GAS.gm1_over_g - 1.0)
    assert np.allclose(dh0s / real_map.corrected_work, real_map.efficiency, rtol=1e-12)


def test_ecmf_is_monotonic_in_beta_on_every_speed_line(real_map):
    """The property that makes ECMF invertible without β — PLAN §3.5."""
    assert all(real_map.monotonic_in_beta("ecmf"))


def test_ecmf_spans_far_more_than_inlet_flow(real_map):
    """ECMF conditioning against Wc: the reason for the coordinate change."""
    assert real_map.span_in_beta("ecmf").max() > real_map.span_in_beta("Wc").max()


def test_load_rejects_inconsistent_sheets(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    p = tmp_path / "bad.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"i": [0, 1, 2], "beta": [0.0, 0.5, 1.0]}).to_excel(
            w, sheet_name="beta_lines", index=False
        )
        for s in ("mass_flow", "pressure_ratio", "efficiency"):
            pd.DataFrame({"i": [0, 1], "0.9": [1.0, 2.0]}).to_excel(w, sheet_name=s, index=False)
    with pytest.raises(ValueError, match="inconsistent sheet shapes"):
        load_beta_map(p, GAS)


# -- evaluation -------------------------------------------------------------


def test_evaluate_at_ecmf_recovers_its_own_grid_points():
    m = _synthetic()
    for j, n in enumerate(m.corrected_speed):
        for i, b in enumerate(m.beta):
            got = m.evaluate_at_ecmf(m.ecmf[i, j], n)
            assert got.PR == pytest.approx(m.PR[i, j], rel=1e-10)
            assert got.corrected_work == pytest.approx(m.corrected_work[i, j], rel=1e-10)
            assert got.beta == pytest.approx(b, abs=1e-10)


def test_speed_is_clamped_not_extrapolated():
    """Cubic extrapolation in speed measures worse than linear; never do either."""
    m = _synthetic()
    lo, hi = m.corrected_speed[0], m.corrected_speed[-1]
    for off in (0.2, 5.0):
        assert m.evaluate_at_beta(0.5, lo - off).PR == pytest.approx(m.evaluate_at_beta(0.5, lo).PR)
        assert m.evaluate_at_beta(0.5, hi + off).PR == pytest.approx(m.evaluate_at_beta(0.5, hi).PR)


def test_ecmf_outside_the_tabulated_range_returns_the_end_point():
    m = _synthetic()
    e = m.ecmf[:, 0]
    assert m.evaluate_at_ecmf(e.min() * 0.5, m.corrected_speed[0]).beta in (m.beta[0], m.beta[-1])
    assert m.evaluate_at_ecmf(e.max() * 2.0, m.corrected_speed[0]).beta in (m.beta[0], m.beta[-1])


def test_minimum_inlet_area_passes_the_highest_corrected_flow():
    from q1d.analytic import max_flow_function

    m = _synthetic()
    a = m.minimum_inlet_area()
    phi = m.Wc.max() * math.sqrt(GAS.R * T_REF) / (a * P_REF)
    assert phi == pytest.approx(max_flow_function(GAS), rel=1e-12)


# -- densification ----------------------------------------------------------


def test_densify_preserves_the_original_nodes(real_map):
    """Refinement must not move the data it was given."""
    d = real_map.densify(9)
    assert len(d.beta) == 9 * (len(real_map.beta) - 1) + 1
    assert len(d.corrected_speed) == 9 * (len(real_map.corrected_speed) - 1) + 1
    assert np.allclose(d.beta[::9], real_map.beta)
    assert np.allclose(d.corrected_speed[::9], real_map.corrected_speed)
    for f in ("Wc", "PR", "corrected_work", "ecmf", "efficiency"):
        assert np.allclose(getattr(d, f)[::9, ::9], getattr(real_map, f), rtol=1e-9), f


def test_densify_keeps_ecmf_monotonic_in_beta(real_map):
    """A refinement that broke invertibility would be worse than no refinement."""
    assert all(real_map.densify(9).monotonic_in_beta("ecmf"))


def test_densify_derives_ecmf_rather_than_interpolating_it(real_map):
    """ECMF at a refined node must equal Wc·√τ/PR of that node, not a spline."""
    d = real_map.densify(3)
    tau = 1.0 + d.corrected_work / (GAS.cp * T_REF)
    assert np.allclose(d.ecmf, d.Wc * np.sqrt(tau) / d.PR, rtol=1e-12)


def test_densify_is_identity_at_factor_one_and_rejects_zero():
    m = _synthetic()
    assert m.densify(1) is m
    assert m.densify(1, nc_factor=1) is m
    with pytest.raises(ValueError, match="factors must be >= 1"):
        m.densify(0)
    with pytest.raises(ValueError, match="factors must be >= 1"):
        m.densify(9, nc_factor=0)


def test_densify_refines_the_two_axes_independently():
    """β and Nc answer different questions, so they get separate factors.

    β resolution sets the within-line interpolation floor; Nc resolution sets
    the cross-speed blending error. Refining one does nothing for the other, so
    a single factor for both is a choice rather than a necessity — and one worth
    being able to measure.
    """
    m = _synthetic(nb=5, ns=4)
    nb, ns = len(m.beta), len(m.corrected_speed)
    d = m.densify(6, nc_factor=2)
    assert len(d.beta) == (nb - 1) * 6 + 1
    assert len(d.corrected_speed) == (ns - 1) * 2 + 1
    assert d.PR.shape == (len(d.beta), len(d.corrected_speed))

    # One axis only, on each side.
    b_only = m.densify(6, nc_factor=1)
    assert len(b_only.beta) == (nb - 1) * 6 + 1
    assert np.array_equal(b_only.corrected_speed, m.corrected_speed)
    n_only = m.densify(1, nc_factor=6)
    assert np.array_equal(n_only.beta, m.beta)
    assert len(n_only.corrected_speed) == (ns - 1) * 6 + 1

    # The default is still square, so every existing call is unchanged.
    sq = m.densify(6)
    assert len(sq.beta) == (nb - 1) * 6 + 1
    assert len(sq.corrected_speed) == (ns - 1) * 6 + 1


def test_densify_reproduces_pchip_across_speed():
    """The point of the whole exercise: dense table + linear lookup == PCHIP."""
    pchip = pytest.importorskip("scipy.interpolate").PchipInterpolator
    m = _synthetic(nb=6, ns=6)
    d = m.densify(9)
    target = 0.5 * (m.corrected_speed[2] + m.corrected_speed[3])
    ref = pchip(m.corrected_speed, m.corrected_work, axis=1)(target)
    got = np.array([d.evaluate_at_beta(b, target).corrected_work for b in m.beta])
    assert np.abs(got - ref).max() / np.abs(ref).max() < 2e-3


def test_densify_improves_a_held_out_speed_line(real_map):
    """Leave-one-out: refining the map must beat linear between the same lines."""
    if len(real_map.corrected_speed) < 5:
        pytest.skip("needs at least 5 speed lines")
    j = len(real_map.corrected_speed) // 2
    keep = [k for k in range(len(real_map.corrected_speed)) if k != j]

    def reduced(mm):
        return BetaMap(
            name="held-out",
            beta=mm.beta,
            corrected_speed=mm.corrected_speed[keep],
            Wc=mm.Wc[:, keep],
            PR=mm.PR[:, keep],
            efficiency=mm.efficiency[:, keep],
            corrected_work=mm.corrected_work[:, keep],
            ecmf=mm.ecmf[:, keep],
            gas=mm.gas,
        )

    n = real_map.corrected_speed[j]
    truth = real_map.corrected_work[:, j]
    scale = np.abs(truth).max()
    sparse = reduced(real_map)
    dense = sparse.densify(9)

    def err(mm):
        got = np.array([mm.evaluate_at_ecmf(e, n).corrected_work for e in real_map.ecmf[:, j]])
        return np.abs(got - truth).max() / scale

    assert err(dense) <= err(sparse) * 1.05
