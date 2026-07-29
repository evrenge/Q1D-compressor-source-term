# Quasi-1D Euler with turbomachinery source terms

A compressor or turbine is modelled as an **actuator disk**: a band of cells in a
1-D duct that receives a momentum source and an energy source, sized so that the
flow leaving the band is the state the machine's map says it should be. There is
no blade geometry and no mass source — the machine is a jump condition that the
PDE has to be consistent with.

`PLAN.md` is the engineering log: every number in this repository was measured,
and it says where and how. It is 240 kB and it is *not* an entry point. **This
file is the entry point.** It defines the vocabulary and says which piece of code
does what.

---

## 1. The vocabulary

Turbomachinery maps use a compact and not-very-guessable notation. These are the
terms that appear throughout the code and `PLAN.md`.

### Corrected (referred) quantities

A machine's behaviour depends on its inlet state. **Corrected** quantities remove
that dependence so one table describes the machine at any inlet condition. They
are referred to a standard day: `T_ref = 288.15 K`, `p_ref = 101325 Pa`.

| symbol | name | definition | in code |
| --- | --- | --- | --- |
| `θ` (theta) | temperature ratio | `T₀₁ / T_ref` | `theta` |
| `δ` (delta) | pressure ratio | `p₀₁ / p_ref` | `delta` |
| `Wc` | corrected mass flow | `W·√θ / δ` | `MapPoint.Wc` |
| `Nc` | corrected speed | `N / √θ` | `corrected_speed` |

**Corrected parameters are a perfect-gas construction.** They are exact only when
`cp` is constant. §3.45 (5) of `PLAN.md` is what happens when that is forgotten
on a real gas — see §5 below.

### Map quantities

| symbol | name | meaning | in code |
| --- | --- | --- | --- |
| `PR` | pressure ratio | `p₀₂/p₀₁` for a compressor; **`p₀₁/p₀₂`** for a turbine, so both read above 1 | `MapPoint.PR` |
| `η` (eta) | isentropic efficiency | actual work vs ideal, for the same `PR` | `MapPoint.efficiency` |
| `η_p` | polytropic efficiency | the same idea per infinitesimal step; **composes exactly**, which is why stage splitting uses it | `design.split_equal_work` |
| `τ` (tau) | temperature ratio | `T₀₂/T₀₁` across the machine | derived |
| `CW` | corrected work | `Δh₀/θ` — specific work, referred to standard day | `MapPoint.corrected_work` |
| `β` (beta) | map coordinate | a **label**, not a physical quantity: an index along a speed line, 0 to 1 | `MapPoint.beta` |
| speed line | | one curve of constant `Nc` across the whole `β` range | `BetaMap._speed_line` |

`β` is worth dwelling on because it confuses everyone once. It carries no
physics. It exists because a speed line is not a function of any single physical
variable — near choke it is vertical in flow, near surge it is flat in pressure
ratio — so the map is tabulated against an arbitrary monotone parameter and you
interpolate in `β`. **On every map supplied here, `β = 1` is surge and `β = 0` is
choke.**

### Keys — how the solver finds its place on the map

The solver measures the flow and has to ask the map "which operating point is
this?" The quantity it asks with is the **key**, and the choice matters:

| key | meaning | why |
| --- | --- | --- |
| ICMF | inlet corrected flow, `Wc` | the obvious choice, and **rank-deficient**: on 13 of 45 supplied speed lines the whole `β` range spans almost no `Wc`, so the inverse does not exist (§3.26) |
| **ECMF** | exit corrected flow, `Wc·√τ/PR` | monotone in `β` on every supplied speed line, both machine types. **This is what the solver uses.** |
| `Φ` (Phi) | flow function, `W√(RT₀)/(A p₀)` | dimensionless, carries no reference state; used for station algebra, not for map lookup |

`ECMF = Wc·√τ/PR` for a compressor and `Wc·√τ·PR` for a turbine — the same
expression, since the two store that pressure ratio the other way up.

### Regimes

| term | meaning |
| --- | --- |
| **surge** | `dPR/dECMF ≥ 0` — no restoring force. The operating point does not come back. A run that refuses here is *correct*, not broken |
| **choke** | sonic at a station; mass flow cannot rise further whatever the back pressure |

---

## 2. How the pieces fit

```
  workbook (.xlsx)
        │  load_beta_map(path, gas)
        ▼
    BetaMap ──densify(n)──▶ BetaMap ──ECMFMap.from_beta_map──▶ ECMFMap
        │                                                          │
        │  design_from_map(map, Nc, ecmf, T01, p01, inlet_mach)    │
        ▼                                                          │
    DuctDesign          ← the 0D model: area, back pressure,       │
        │                 both station states, Fx and SWx,         │
        │                 algebraically, with no PDE               │
        ▼                                                          ▼
     Solver  ◀────────────────  EcmfCompressor (the disk)  ◀───────┘
        │                       reads the field, looks up the map,
        │                       returns the source array
        ▼
   converged field  ──compare──▶  DuctDesign     (§3.46: agree to 2.8e-06)
```

Read that bottom line as the whole verification strategy: the **0D model** and
the **Q1D solver** are two independent routes to the same steady state, and the
gate is that they agree.

---

## 3. The modules

| module | what it is |
| --- | --- |
| `gas.py` | thermodynamics — `PerfectGas`, `Nasa9Gas` |
| `maps.py` | workbook loading, `β`→ECMF conversion, densification, map persistence |
| `analytic.py` | the closed-form 0D layer: station algebra, flow function, flux inversion |
| `design.py` | `design_from_map` — size a duct so that a chosen map point *is* its steady state |
| `grid.py` | the 1-D mesh and areas |
| `solver.py` | the PDE: Roe flux, MUSCL reconstruction, 5-stage Runge-Kutta |
| `boundary.py` | characteristic inflow/outflow — **this is where Riemann invariants live** |
| `compressor.py` | the actuator-disk source terms (six classes, see below) |
| `riemann.py` | exact Riemann solver, used **only** to verify the Sod shock tube |
| `implicit.py` | implicit time integration (optional) |

---

## 4. The disk classes — which is which

This is the single most confusing part of the codebase, so it is worth stating
flatly. There are six, and **five of them are for real work while one is test
scaffolding.**

| class | driven by | keys on | status |
| --- | --- | --- | --- |
| **`EcmfCompressor`** | `ECMFMap` (a workbook) | exit ECMF, read one step behind | **the current one — use this** |
| `FlowMatchedCompressor` | `BetaMap` | `β` relaxed on the flow residual | works; superseded |
| `UnsteadyMappedCompressor` | `BetaMap` | `β` as a state | works; superseded |
| `InletFlowCompressor` | `BetaMap` | inlet `Wc` | works, but the inverse does not exist on 13 of 45 speed lines |
| `MappedCompressor` | `BetaMap` | exit `Wc` | early version |
| `ActuatorDisk` | a `CompressorMap` **protocol** — usually `ConstantCompressorMap`, i.e. a fixed `PR`/`η` you type in | inlet `Φ₁` | **test scaffolding, not a map path** |

`ActuatorDisk` predates map support entirely. You hand it a constant `PR` and
`η` rather than a workbook, which is the only case with a closed-form answer
(`analytic.zero_d_compressor`) to check the source-term plumbing against. It is
used by one test file and nothing else. **It is not on the path from a workbook
to a converged operating point**, and nothing in that path depends on it.

*Turbines use the same classes.* A turbine is not a separate implementation — it
is a map whose `PR` is stored as an expansion ratio and whose work is negative.
`load_beta_map` detects which from `dWc/dPR` and carries a `kind` field through.

---

## 5. Gas models — what is real and what is not

```python
from q1d.gas import PerfectGas, Nasa9Gas

gas = PerfectGas(gamma=1.4, cp=1004.7)     # constant cp
gas = Nasa9Gas.from_cantera()              # NASA9 polynomials, dry air
```

`Nasa9Gas` reads 9-coefficient NASA polynomials from Cantera's `airNASA9.yaml`
once at construction and evaluates them itself. On air, `cp` rises **21.7%**
between 288 K and 1600 K and `γ` falls 1.3988 → 1.3061.

It deliberately has **no `gamma` or `cp` attribute** — reading either as a scalar
is the mistake the class exists to prevent, so both raise `AttributeError`. Use
`cp_at(T)`, `gamma_at(T)`, `pressure_ratio_isentropic(T1, T2)`,
`temperature_isentropic(T1, PR)`.

### Real-gas coverage

**The whole workbook→map→design→solver path is real-gas**, for compressors and
turbines alike:

`load_beta_map` · `densify` · `ECMFMap` (incl. `save`/`load`) · `ScaledMap` ·
`design_from_map` · `DuctDesign` · `split_equal_work` · `steady_profile` ·
`build_case` · the solver · **all five map-driven disk classes**

**Perfect-gas only**, and why:

| what | why |
| --- | --- |
| `ActuatorDisk` | test scaffolding (§4); uses `max_flow_function` and `compressor_source_terms` |
| `zero_d_compressor`, `compressor_exit_stagnation`, `compressor_source_terms`, `minimum_back_pressure` | the Phase-1 closed-form 0D compressor, built on `τ = 1 + (PR^κ−1)/η` |
| `flow_function`, `max_flow_function`, `mach_from_flow_function`, `choked_mass_flow` | the constant-γ flow-function family. `flow_function_at(M, gas, T0)` and `max_flow_function_at(gas, T0)` are the real-gas equivalents |
| `riemann.py` | exact Riemann solver, perfect-gas by construction. Sod verification only |

### What the gas model is worth

Measured at the same map point, so `PR`, `Wc` and `η` are identical and every
delta is the thermodynamics (`PLAN.md` §3.46):

| | compressor PR 2.2 | compressor PR 22.8 | turbine PR 2.6 | turbine PR 3.9 |
| --- | --- | --- | --- | --- |
| `τ` | −0.1% | **−2.8%** | **+3.6%** | **+5.6%** |
| `T₀₂` | −0.4 K | **−22.7 K** | **+46.5 K** | **+63.7 K** |
| duct area | +0.2% | +0.2% | **+3.6%** | **+3.7%** |

Mass flow does not move at all — it is `Wc`, which is workbook data. The gas
moves everything *thermal*, and the geometry that follows.

**Caveat worth knowing:** the composition is dry air (N₂/O₂, no argon). A real
turbine runs on combustion products with a different `R` and `cp`. The
thermodynamics is correct for the gas it is given; the gas is not yet the right
one for a turbine.

**Cost:** a real-gas step is ~11.5× a perfect-gas one.

---

## 6. Running one

```python
from q1d.boundary import StagnationInletStaticOutlet
from q1d.compressor import EcmfCompressor
from q1d.design import design_from_map, steady_profile
from q1d.gas import Nasa9Gas
from q1d.grid import Grid
from q1d.maps import ECMFMap, load_beta_map
from q1d.solver import ReferenceState, Solver, SolverConfig

gas = Nasa9Gas.from_cantera()

# 1. Map. `densify` refines the (beta, Nc) grid before the ECMF conversion.
beta_map = load_beta_map("data/maps/SubsonicCompressor.xlsx", gas).densify(36)
ecmf_map = ECMFMap.from_beta_map(beta_map)

# 2. Pick an operating point and size a duct around it (this is the 0D model).
line = beta_map._speed_line(1.0)[0]
target = float(line.min()) + 0.35 * (float(line.max()) - float(line.min()))
design = design_from_map(beta_map, 1.0, target, inlet_mach=0.45,
                         T01=288.15, p01=101325.0)

# 3. Build the Q1D case.
n_cells, n_smear = 201, 7
disk_cell = (n_cells - n_smear) // 2
grid = Grid.uniform(0.0, 1.0, n_cells, design.area)
disk = EcmfCompressor(cell=disk_cell, ecmf_map=ecmf_map, corrected_speed=1.0,
                      sample_offset=2, exit_offset=3, n_smear=n_smear,
                      inlet_lag=1e-2)
solver = Solver(
    grid, gas,
    StagnationInletStaticOutlet(p0_in=design.p01, T0_in=design.T01,
                                p_back=design.p_back),
    ReferenceState(rho=design.station1.rho, u=design.station1.u,
                   p=design.station1.p),
    config=SolverConfig(cfl=2.0), source=disk,
)

# 4. Seed with the discrete steady profile, not a two-state jump.
solver.set_state(*steady_profile(design, n_cells, disk_cell, n_smear))
solver.run(max_steps=30000, tol=1e-11)

# 5. The gate: converged mass flow against the 0D design.
error = float(solver.face_fluxes()[0].mean()) / design.W - 1.0
```

A turbine is the same code — a turbine workbook, `T01=1600.0`, `p01=1.2e6`, and
a lower `inlet_mach` because the exit is the station that chokes.

**Map persistence.** Building a map parses a workbook, refines the grid and
re-derives ECMF: 0.06–0.70 s. A solver run should not pay that per case.

```python
ecmf_map.save("subsonic_nc.npz")
ecmf_map = ECMFMap.load("subsonic_nc.npz")     # carries its own gas
```

---

## 7. Where the numbers are

Everything asserted here was measured. `PLAN.md` §3 is the log, newest first.
The sections most worth reading:

| § | topic |
| --- | --- |
| 3.5, 3.26 | why ECMF and not inlet corrected flow |
| 3.6 | why the map stores corrected work rather than η |
| 3.23 | reading stations from the fluxes — worth ~11 cells per blade row |
| 3.44 | the failure census: 1395/1423 cells, and why the 28 refusals are correct |
| 3.45 | the real gas: five defects a perfect gas cannot show you |
| 3.46 | the 0D ↔ Q1D gate, and what the gas model itself is worth |

Tests: `pytest` for the fast suite, `pytest -m "not slow"` to skip the marching
cases. `tests/test_nasa9_endtoend.py` is the real-gas gate.
