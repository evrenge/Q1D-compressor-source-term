"""Several blade rows on one duct — the step from a component to a machine.

Everything before this ran one disk. A multistage compressor is not one disk
repeated: each stage sees its own inlet corrected flow, and across a stage of
PR ≈ 2 the corrected flow roughly halves, so a single map runs out of table after
two stages (:class:`q1d.maps.ScaledMap`). The annulus must taper with it, or the
axial Mach number climbs stage by stage until the duct chokes.

What this file pins is the property that makes an engine possible at all: **the
error does not grow with stage count**. A scheme whose per-stage error compounds
would be useless for a machine with ten rows however good it looked on one.

The taper between rows is C¹ (a smoothstep), not piecewise-linear. That is worth
a test of its own: the kink in a linear profile puts a delta in ``dA/dx`` into
the geometric source, and the measured cost was **170×** in converged mass flow
(``PLAN.md`` §3.21).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from q1d.analytic import state_from_flux, static_from_stagnation
from q1d.boundary import StagnationInletStaticOutlet
from q1d.compressor import CompositeSource, FlowMatchedCompressor
from q1d.gas import PerfectGas
from q1d.grid import Grid
from q1d.maps import P_REF, T_REF, ScaledMap, load_beta_map
from q1d.solver import NonPhysicalState, ReferenceState, Solver, SolverConfig

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

GAS = PerfectGas(1.4, 1004.7)
MAP = Path(__file__).resolve().parents[1] / "data" / "maps" / "SubsonicCompressor.xlsx"
SPEED = 1.0
FRAC = 0.35  # a little to the surge side of mid-line, PR 2.22 per stage


def area_for_mach(T0, p0, W, mach):
    T = T0 / (1.0 + 0.5 * GAS.gm1 * mach * mach)
    p = p0 * (T / T0) ** GAS.g_over_gm1
    return W / ((p / (GAS.R * T)) * mach * math.sqrt(GAS.gamma * GAS.R * T))


def design(n_stages, speed=SPEED, frac=FRAC, inlet_mach=0.45, p01=101325.0, T01=288.15):
    """A repeating-stage machine: same relative point on every stage's own map."""
    m = load_beta_map(MAP, GAS).densify(9)
    ecmf = m._speed_line(speed)[0]
    lo, hi = float(ecmf.min()), float(ecmf.max())
    pt = m.evaluate_at_ecmf(lo + frac * (hi - lo), speed)

    W = pt.Wc * (p01 / P_REF) / math.sqrt(T01 / T_REF)
    stations = [(T01, p01)]
    scales = []
    for _ in range(n_stages):
        T0, p0 = stations[-1]
        th, de = T0 / T_REF, p0 / P_REF
        scales.append(W * math.sqrt(th) / de / pt.Wc)  # so the lookup lands on pt
        stations.append((T0 + pt.corrected_work * th / GAS.cp, pt.PR * p0))
    areas = [area_for_mach(T0, p0, W, inlet_mach) for T0, p0 in stations]
    return m, W, stations, scales, areas, pt


def taper(xs, As, c1=True):
    """Area against x through the constant-area pads and the tapers between them.

    ``c1=False`` is the piecewise-linear profile it replaced — kept so the
    regression test can assert the C¹ one is better rather than merely asserting
    that it works.
    """

    def area_fn(x):
        x = np.asarray(x, float)
        out = np.interp(x, xs, As)
        if not c1:
            return out
        for j in range(len(xs) - 1):
            a, b = xs[j], xs[j + 1]
            if b <= a or As[j] == As[j + 1]:
                continue
            msk = (x > a) & (x < b)
            if not msk.any():
                continue
            t = (x[msk] - a) / (b - a)
            out[msk] = As[j] + (As[j + 1] - As[j]) * (t * t * (3.0 - 2.0 * t))
        return out

    return area_fn


def build(n_stages, ncell=601, pad=2, offset=2, lag=1e-2, cfl=2.0, c1=True):
    m, W, stations, scales, areas, pt = design(n_stages)
    gap = (ncell - 40) // max(n_stages, 1)
    cells = [offset + 8 + k * gap for k in range(n_stages)]
    assert cells[-1] + pad + 3 < ncell, "machine does not fit in the duct"

    xs, As = [0.0], [areas[0]]
    for k, c in enumerate(cells):
        xs += [(c - pad) / ncell, (c + 1 + pad) / ncell]
        As += [areas[k], areas[k]]
    xs.append(1.0)
    As.append(areas[n_stages])
    grid = Grid.uniform(0.0, 1.0, ncell, taper(np.asarray(xs), np.asarray(As), c1))

    disks = [
        FlowMatchedCompressor(
            cell=c, beta_map=ScaledMap(m, s), corrected_speed=SPEED,
            sample_offset=offset, n_smear=1, inlet_lag=lag, beta0=pt.beta,
        )
        for c, s in zip(cells, scales, strict=True)
    ]
    st_in = static_from_stagnation(*stations[0], W, areas[0], GAS)
    st_ex = static_from_stagnation(*stations[-1], W, areas[-1], GAS)
    s = Solver(
        grid, GAS,
        StagnationInletStaticOutlet(p0_in=stations[0][1], T0_in=stations[0][0],
                                    p_back=st_ex.p),
        ReferenceState(rho=st_in.rho, u=st_in.u, p=stations[0][1]),
        config=SolverConfig(cfl=cfl), source=CompositeSource(disks),
    )

    ac = grid.a_cell
    rho, u, p = np.empty(ncell), np.empty(ncell), np.empty(ncell)
    for i in range(ncell):
        k = sum(1 for c in cells if i >= c + 1)
        st = static_from_stagnation(*stations[k], W, ac[i + 1], GAS)
        rho[i], u[i], p[i] = st.rho, st.u, st.p
    for k, c in enumerate(cells):
        A = areas[k]
        a = static_from_stagnation(*stations[k], W, A, GAS)
        b = static_from_stagnation(*stations[k + 1], W, A, GAS)
        Fx = (b.p - a.p) * A + W * (b.u - a.u)
        SWx = W * GAS.cp * (stations[k + 1][0] - stations[k][0])
        F = np.array([
            a.rho * a.u * A,
            (a.rho * a.u**2 + a.p) * A,
            a.rho * a.u * (GAS.cp * a.T + 0.5 * a.u**2) * A,
        ])
        x = state_from_flux(F + 0.5 * np.array([0.0, Fx, SWx]), A, GAS)
        rho[c], u[c], p[c] = x.rho, x.u, x.p
    s.set_state(rho, u, p)
    return s, disks, W, stations, pt


def hold(n_stages, steps=200_000, tol=1e-11, **kw):
    s, disks, W, stations, pt = build(n_stages, **kw)
    for k in range(steps):
        try:
            s.advance(s.timestep())
        except (NonPhysicalState, ValueError) as ex:  # pragma: no cover
            pytest.fail(f"{n_stages} stages failed at step {k}: {ex}")
        if k % 500 == 0 and k > 0:
            if float(s.residual_norm()[0]) / W * s.grid.dx.mean() < tol:
                break
    mf = s.face_fluxes()[0]
    return {
        "err": float(mf.mean()) / W - 1.0,
        "nonuniformity": float(mf.max() - mf.min()) / W,
        "opr": stations[-1][1] / stations[0][1],
        "stage_PR": pt.PR,
        "reversals": sum(d.reversals for d in disks),
    }


def test_the_stages_are_different_machines():
    """The premise for `ScaledMap`: one map cannot serve them all."""
    m, _, _, scales, _, _ = design(6)
    assert scales[0] == pytest.approx(1.0, rel=1e-12), "stage 1 is the reference"
    assert scales[-1] < 0.05, f"corrected flow only fell to {scales[-1]:.3g} of stage 1"
    wc = m._speed_line(SPEED)[1]
    assert scales[-1] * float(wc.mean()) < float(wc.min()), (
        "stage 6 would still be inside the unscaled table, so the test proves nothing"
    )


def test_the_annulus_has_to_taper():
    """Holding area constant would drive the axial Mach number up every stage."""
    _, _, _, _, areas, _ = design(6)
    assert areas[-1] < 0.2 * areas[0]
    assert all(b < a for a, b in zip(areas, areas[1:], strict=False)), "taper is not monotone"


@pytest.mark.slow
def test_the_error_does_not_grow_with_stage_count():
    """The property that makes a whole engine possible.

    Deliberately **relative**, not an absolute threshold. The absolute error is a
    property of the mesh — 601 cells over the whole machine is 100 cells a row at
    six stages and 600 at one — so a fixed gate would mostly be measuring how
    many cells each row happened to get. What has to be true for an engine is
    that stacking rows does not *compound* the error, and that is what this
    asserts: six stages, OPR ~120, no worse than three times one stage.

    Each stage is PR 2.22, so six of them are an engine-scale overall pressure
    ratio built from a real map, on one mesh, with no internal boundary
    conditions anywhere.
    """
    runs = {n: hold(n) for n in (1, 2, 4, 6)}
    assert all(r["reversals"] == 0 for r in runs.values())
    for n, r in runs.items():
        assert r["nonuniformity"] < 1e-5, (
            f"{n} stages: mass flux varies by {r['nonuniformity']:.3e} along the duct"
        )
    one = abs(runs[1]["err"])
    for n in (2, 4, 6):
        assert abs(runs[n]["err"]) < max(3.0 * one, 1e-5), (
            f"{n} stages, OPR {runs[n]['opr']:.1f}: W off by {runs[n]['err']:.3e} "
            f"against {runs[1]['err']:.3e} for one stage — the error compounds"
        )


@pytest.mark.slow
def test_six_stages_reach_engine_scale_pressure_ratio():
    """Guards the premise of the test above — that OPR is actually large."""
    _, _, stations, _, _, pt = design(6)
    assert pt.PR > 2.0
    assert stations[-1][1] / stations[0][1] > 100.0


@pytest.mark.slow
def test_the_c1_taper_beats_the_piecewise_linear_one():
    """A kink in ``A(x)`` puts a delta into ``p·dA/dx``, and it is expensive.

    Measured at 170× on the four-stage machine (``PLAN.md`` §3.21). The gate is
    an order of magnitude, so a modest change in the taper's shape does not
    make this brittle.
    """
    smooth = hold(4, c1=True)
    kinked = hold(4, c1=False)
    assert abs(smooth["err"]) < 0.1 * abs(kinked["err"]), (
        f"C1 {smooth['err']:+.3e} against piecewise-linear {kinked['err']:+.3e}"
    )
