"""Gate for turbine maps — the loader was modelling every one as a compressor.

Three separate errors sat in one expression, and all three are load-bearing:

* ``Δh₀ˢ = cp·T_ref·(PR^k − 1)`` is the **compression** isentropic relation. A
  turbine expands, and the ideal change is a *drop* of ``cp·T_ref·(1 − PR^−k)``
  with ``PR`` the expansion ratio. Not a sign flip — a different formula.
* ``work = ideal/η`` is the **compression** convention. A compressor needs more
  than ideal; a turbine delivers less, so η multiplies.
* ``ECMF = Wc·√τ/PR`` assumes ``PR = p₀₂/p₀₁``. A turbine tabulates
  ``p₀₁/p₀₂``, so the factor is ``PR``, not its reciprocal.

Together they gave every cell of every supplied turbine map a temperature
*rise*: a PR 3, η 0.9 stage came out at **+41%** where it must drop **24%**.
Three of the six workbooks raised on ``τ ≤ 0`` and the other three loaded and
were quietly wrong, which is the worse half.

These build maps in memory rather than reading the supplied workbooks, which
live outside the repository — the numbers below are closed-form, so they pin the
physics rather than a particular vendor's file.
"""

import numpy as np
import pandas as pd
import pytest

from q1d.gas import PerfectGas
from q1d.maps import T_REF, _infer_kind, load_beta_map

GAS = PerfectGas(1.4, 1004.7)
K = GAS.gm1_over_g


def write_map(path, PR, Wc, eff, beta=None, speed=(0.8, 0.9, 1.0)):
    """Write a workbook in the supplied maps' layout."""
    PR, Wc, eff = np.asarray(PR, float), np.asarray(Wc, float), np.asarray(eff, float)
    beta = np.linspace(0.0, 1.0, PR.shape[0]) if beta is None else np.asarray(beta, float)
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame({"i": np.arange(len(beta)), "beta": beta}).to_excel(
            xl, sheet_name="beta_lines", index=False
        )
        for sheet, a in (("mass_flow", Wc), ("pressure_ratio", PR), ("efficiency", eff)):
            df = pd.DataFrame(a, columns=[str(s) for s in speed])
            df.insert(0, "beta", beta)
            df.to_excel(xl, sheet_name=sheet, index=False)
    return path


def turbine_arrays(n_beta=6, n_speed=3, eff_value=0.9):
    """Expansion ratio rising with β, and flow rising with it — a turbine."""
    PR = np.linspace(1.5, 4.0, n_beta)[:, None] * np.ones((1, n_speed))
    Wc = np.linspace(10.0, 14.0, n_beta)[:, None] * np.linspace(1.0, 1.1, n_speed)
    eff = np.full((n_beta, n_speed), eff_value)
    return PR, Wc, eff


def compressor_arrays(n_beta=6, n_speed=3):
    """PR rising with β while flow falls — a compressor throttled toward surge."""
    PR = np.linspace(1.5, 4.0, n_beta)[:, None] * np.ones((1, n_speed))
    Wc = np.linspace(14.0, 10.0, n_beta)[:, None] * np.linspace(1.0, 1.1, n_speed)
    eff = np.full((n_beta, n_speed), 0.85)
    return PR, Wc, eff


class TestKindIsInferredFromTheMachine:
    """``dWc/dPR`` separates them, and on the supplied maps it is unanimous.

    A compressor throttled toward surge passes less flow at more pressure ratio;
    a turbine passes more as the expansion ratio opens. Measured over all
    seventeen workbooks: positive on 64 of 64 turbine speed lines, negative on
    109 of 109 compressor lines, no map mixed.
    """

    def test_flow_rising_with_pressure_ratio_is_a_turbine(self):
        PR, Wc, _ = turbine_arrays()
        assert _infer_kind(Wc, PR) == "turbine"

    def test_flow_falling_with_pressure_ratio_is_a_compressor(self):
        PR, Wc, _ = compressor_arrays()
        assert _infer_kind(Wc, PR) == "compressor"

    def test_the_workbook_is_classified_on_load(self, tmp_path):
        write_map(tmp_path / "t.xlsx", *turbine_arrays())
        write_map(tmp_path / "c.xlsx", *compressor_arrays())
        assert load_beta_map(tmp_path / "t.xlsx", GAS).kind == "turbine"
        assert load_beta_map(tmp_path / "c.xlsx", GAS).kind == "compressor"

    def test_an_explicit_kind_overrides_the_inference(self, tmp_path):
        write_map(tmp_path / "t.xlsx", *turbine_arrays())
        m = load_beta_map(tmp_path / "t.xlsx", GAS, kind="compressor")
        assert m.kind == "compressor"
        assert np.all(m.corrected_work > 0.0), "forced compressor must add work"

    def test_a_nonsense_kind_is_refused(self, tmp_path):
        write_map(tmp_path / "t.xlsx", *turbine_arrays())
        with pytest.raises(ValueError, match="kind must be"):
            load_beta_map(tmp_path / "t.xlsx", GAS, kind="pump")


class TestATurbineExtractsWorkAndCools:
    def test_corrected_work_is_negative(self, tmp_path):
        write_map(tmp_path / "t.xlsx", *turbine_arrays())
        m = load_beta_map(tmp_path / "t.xlsx", GAS)
        assert np.all(m.corrected_work < 0.0), "a turbine takes work out of the flow"

    def test_the_temperature_ratio_is_below_one(self, tmp_path):
        write_map(tmp_path / "t.xlsx", *turbine_arrays())
        m = load_beta_map(tmp_path / "t.xlsx", GAS)
        tau = 1.0 + m.corrected_work / (GAS.cp * T_REF)
        assert np.all(tau < 1.0), f"turbine must cool, got tau up to {tau.max():.4f}"

    def test_it_matches_the_closed_form(self, tmp_path):
        """PR 3, η 0.9 drops 24%. The compressor form gave +41% here."""
        PR, Wc, eff = turbine_arrays()
        write_map(tmp_path / "t.xlsx", PR, Wc, eff)
        m = load_beta_map(tmp_path / "t.xlsx", GAS)
        want = -eff * GAS.cp * T_REF * (1.0 - PR**-K)
        assert m.corrected_work == pytest.approx(want, rel=1e-13)

        tau = 1.0 + m.corrected_work / (GAS.cp * T_REF)
        wrong = 1.0 + (PR**K - 1.0) / eff  # what the loader used to produce
        assert np.all(tau < 1.0) and np.all(wrong > 1.0), "premise of the fix is gone"

    def test_efficiency_multiplies_rather_than_divides(self, tmp_path):
        """Halving η must halve the work extracted, not double it."""
        PR, Wc, _ = turbine_arrays()
        w = {}
        for e in (0.9, 0.45):
            write_map(tmp_path / f"t{e}.xlsx", PR, Wc, np.full(PR.shape, e))
            w[e] = load_beta_map(tmp_path / f"t{e}.xlsx", GAS).corrected_work
        assert w[0.45] == pytest.approx(0.5 * w[0.9], rel=1e-13)

    def test_ecmf_uses_the_expansion_ratio_the_right_way_up(self, tmp_path):
        """``ECMF = Wc·√τ·(p₀₁/p₀₂)``, and a turbine tabulates that ratio directly."""
        PR, Wc, eff = turbine_arrays()
        write_map(tmp_path / "t.xlsx", PR, Wc, eff)
        m = load_beta_map(tmp_path / "t.xlsx", GAS)
        tau = 1.0 + m.corrected_work / (GAS.cp * T_REF)
        assert m.ecmf == pytest.approx(Wc * np.sqrt(tau) * PR, rel=1e-13)


class TestNonPositiveEfficiency:
    """Repaired on turbines, deliberately left alone on compressors.

    The asymmetry is the whole point. On a compressor those cells all have
    PR < 1 — 12 of 12 across the supplied maps — so dividing by a negative η
    cancels against a negative numerator and gives τ > 1: work in, temperature
    up, pressure down, a stalled corner correctly modelled. On a turbine η
    multiplies and 35 of 36 such cells have PR > 1, so nothing cancels and the
    cell is simply wrong.
    """

    def test_a_turbine_cell_is_repaired_and_recorded(self, tmp_path):
        PR, Wc, eff = turbine_arrays()
        eff[0, 1] = -0.03
        write_map(tmp_path / "t.xlsx", PR, Wc, eff)
        m = load_beta_map(tmp_path / "t.xlsx", GAS)
        assert (0, 1) in m.repaired_efficiency
        assert m.efficiency[0, 1] > 0.0
        assert m.corrected_work[0, 1] < 0.0, "repaired cell must still extract work"

    def test_the_repair_does_not_touch_the_other_cells(self, tmp_path):
        PR, Wc, eff = turbine_arrays()
        clean = load_beta_map(write_map(tmp_path / "a.xlsx", PR, Wc, eff), GAS)
        eff[0, 1] = -0.03
        holed = load_beta_map(write_map(tmp_path / "b.xlsx", PR, Wc, eff), GAS)
        m = np.ones(eff.shape, bool)
        m[0, 1] = False
        assert holed.corrected_work[m] == pytest.approx(clean.corrected_work[m], rel=1e-13)

    def test_a_compressor_cell_is_left_alone(self, tmp_path):
        """Its sign cancellation is correct physics; repairing it would break it."""
        PR, Wc, eff = compressor_arrays()
        PR[0, :] = 0.95  # not compressing at this corner
        eff[0, 1] = -0.2
        write_map(tmp_path / "c.xlsx", PR, Wc, eff)
        m = load_beta_map(tmp_path / "c.xlsx", GAS)
        assert m.repaired_efficiency == ()
        assert m.efficiency[0, 1] == -0.2, "compressor efficiency must be untouched"
        tau = 1.0 + m.corrected_work[0, 1] / (GAS.cp * T_REF)
        assert tau > 1.0, "work in with pressure falling still heats the flow"

    def test_a_speed_line_with_no_usable_cell_is_refused(self, tmp_path):
        """There is nothing to interpolate from, and inventing one would be a lie."""
        PR, Wc, eff = turbine_arrays()
        eff[:, 1] = -0.1
        write_map(tmp_path / "t.xlsx", PR, Wc, eff)
        with pytest.raises(ValueError, match="no cell with eta > 0"):
            load_beta_map(tmp_path / "t.xlsx", GAS)


class TestTurbinesAreKeyedOnPR:
    """ECMF keys a compressor; it does not key a turbine.

    Monotonic in β on 135 of 135 compressor speed lines, but only **17 of 64**
    turbine lines — while ``PR`` is monotonic on **64 of 64**. Neither choice
    generalises, so the key follows the machine.
    """

    def build(self, tmp_path, arrays, name="m.xlsx"):
        from q1d.maps import ECMFMap

        write_map(tmp_path / name, *arrays)
        return ECMFMap.from_beta_map(load_beta_map(tmp_path / name, GAS).densify(3))

    def test_a_turbine_keys_on_pressure_ratio(self, tmp_path):
        e = self.build(tmp_path, turbine_arrays())
        assert e.key_field == "PR"

    def test_a_compressor_still_keys_on_ecmf(self, tmp_path):
        e = self.build(tmp_path, compressor_arrays())
        assert e.key_field == "ecmf"
        assert e.Wc is None, "Wc follows from an ECMF key exactly; do not tabulate it"

    def test_the_key_ascends_so_interp_can_use_it(self, tmp_path):
        e = self.build(tmp_path, turbine_arrays())
        assert np.all(np.diff(e.key, axis=0) > 0.0)

    def test_the_point_reproduces_the_pressure_ratio_asked_for(self, tmp_path):
        """The PR-keyed analogue of the compressor's key-reproduction property."""
        e = self.build(tmp_path, turbine_arrays())
        worst = 0.0
        for j in range(len(e.corrected_speed)):
            nc = float(e.corrected_speed[j])
            lo, hi = float(e.key[0, j]), float(e.key[-1, j])
            for f in (0.05, 0.5, 0.95):
                q = lo + f * (hi - lo)
                worst = max(worst, abs(e.evaluate(q, nc).PR / q - 1.0))
        assert worst == 0.0, f"PR not reproduced: {worst:.3e}"

    def test_ecmf_comes_back_with_the_turbine_factor(self, tmp_path):
        e = self.build(tmp_path, turbine_arrays())
        nc = float(e.corrected_speed[0])
        q = float(e.key[len(e.key) // 2, 0])
        pt = e.evaluate(q, nc)
        tau = 1.0 + pt.corrected_work / (GAS.cp * T_REF)
        assert pt.ecmf == pytest.approx(pt.Wc * np.sqrt(tau) * pt.PR, rel=1e-12)

    def test_the_ecmf_attribute_refuses_to_alias_a_pr_key(self, tmp_path):
        """A wrong label on a table that still interpolates cleanly is the
        failure mode §3.37 spent an afternoon on. Loud beats convenient."""
        e = self.build(tmp_path, turbine_arrays())
        with pytest.raises(AttributeError, match="keyed on PR"):
            _ = e.ecmf

    def test_densify_keeps_the_machine_it_was_given(self, tmp_path):
        """A densified turbine that reverted to compressor forms would undo the
        loader's decision one call later, and silently."""
        write_map(tmp_path / "t.xlsx", *turbine_arrays())
        m = load_beta_map(tmp_path / "t.xlsx", GAS)
        d = m.densify(4)
        assert d.kind == "turbine"
        assert np.all(d.corrected_work < 0.0)
        tau = 1.0 + d.corrected_work / (GAS.cp * T_REF)
        assert d.ecmf == pytest.approx(d.Wc * np.sqrt(tau) * d.PR, rel=1e-12)
        # η is *re-derived* from the refined PR and work, never interpolated
        # (it has a pole on `TranssonicCompressor`, §3.6). So it is exact on the
        # supplied β nodes and only close between them — assert both, rather
        # than pick a tolerance loose enough to hide the first.
        assert d.efficiency[::4] == pytest.approx(0.9, rel=1e-12), "exact on the nodes"
        assert d.efficiency == pytest.approx(0.9, abs=1e-2), "close between them"
