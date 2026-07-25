# Quasi-1D Euler Solver with Turbomachinery Source Terms — Development Plan

Status: **Phase 1 complete, Phase 2 next**
Last updated: 2026-07-25

---

## 1. Goal

A quasi-1D Euler solver in which turbomachinery components (compressor, turbine)
are represented as **momentum and energy source terms** injected into the
interior of the flowpath — not as internal boundary conditions. The Python
implementation here is a reference and prototyping platform; the production
solver will be ported to C++ separately.

Scope grows in layers: constant-PR/η compressor on ideal gas → real gas
(NASA9) → real multi-speed compressor maps → turbine → multi-component.

---

## 2. Architectural principles

These four rules exist because violating the first one produced the systematic
error found in the prototype. They are binding on all phases.

### P1. Precompute the *fluid*, not the *operating point*

| Precomputed / tabulated | Computed at runtime |
| --- | --- |
| Gas properties: `h(T)`, `s°(T)`, `cp(T)` and the inverses `T(h)`, `T(s°)` | `Fx`, `SWx` |
| Compressor/turbine map: `PR(Φ₁, N)`, `η(Φ₁, N)` | Station static states |

Gas properties are functions of the fluid — universal, with no operating point
baked in. The map is given data describing the machine. **Everything that
depends on the local flow state is computed where the local flow state lives.**

The prototype tabulated `Fx` [N] and `SWx` [W] — dimensional quantities that
scale with `p₀₁·A₁` and `W·cp·T₀₁` respectively — against a single scalar
abscissa. That table is valid only at the inlet condition it was generated at.
Every downstream symptom (no inlet-state dependence, the exit-referenced
corrected-flow coordinate, the downstream sampling feedback loop) followed from
that one decision.

There is **one code path** for ideal and real gas. `SWx = W·(h₀₂ − h₀₁)` and
`Fx = pₛ₂A₂ − pₛ₁A₁ + W(u₂ − u₁)` are exact in both. `IdealGas` is the same
interface with linear `h(T)`.

### P2. The source is a source, always

Single-cell conservative injection into the interior. No internal boundary
conditions, no node splitting. This is the native capability of a quasi-1D
finite-volume formulation and it makes the integral balance exact by
construction.

A normalized smear distribution over `n_smear` cells is designed into the
interface from the start, defaulting to a single cell. Widening it later is a
parameter change, not a refactor, and it opens up finite-length blade rows and
distributed axial force modelling.

### P3. The solver never sees `gamma` directly

All thermodynamics goes through a `Gas` interface. Three places in the
prototype assume a calorically perfect gas *structurally*, not just
numerically, and each must be reformulated for real gas:

1. `cons_to_prim`: `p = (γ−1)(ρE − ρu²/2)` becomes an inversion of
   `e(T) = h(T) − RT` per cell per stage. **This, not the source term, is the
   expensive part of real gas.**
2. The Roe average assumes constant γ.
3. The characteristic BCs use `2c/(γ−1)`, which is the constant-γ Riemann
   invariant.

**Decision: consistent real-gas treatment, not frozen-γ** (see §6).

### P4. Every phase has a quantitative gate

No phase is complete on inspection. Each ends with a numerical acceptance
criterion, and the tolerance is chosen to be tight enough that a real defect
cannot hide inside it. Where a closed-form answer exists, it is the gate.

---

## 3. Verified reference values

Computed and cross-checked during review. These are the acceptance targets.

### 3.1 The compressor test case has an exact closed-form steady state

Configuration: constant-area duct, `A = 0.1 m²`, inlet total `p₀₁ = 101325 Pa`,
`T₀₁ = 288.15 K`, exit static back pressure `p_b = 101325 Pa`, `PR = 1.2`,
`η_is = 0.9`, `γ = 1.4`, `cp = 1005 J/kg/K` (hence `R = 287.1429`, *not*
287.05 — self-consistent but not standard air).

| quantity | value |
| --- | --- |
| **W** | **21.490742688146575 kg/s** |
| M₁ / M₂ | 0.6647817795154938 / 0.5170711949922851 |
| p₀₂ / T₀₂ | 121590 Pa / 305.27011981156431 K |
| pₛ₂ | 101325 Pa (= back pressure, exactly) |
| Fx | 1731.2433536618760 N |
| SWx | 369763.7101088717 W |
| τ = T₀₂/T₀₁ | 1.05941391570906 |
| Φ₁ / Φ₁ᵐᵃˣ | 0.891 |

> **Corrected 2026-07-25 (Phase 1).** M₁, Fx and SWx were previously listed as
> 0.66516, 1730.963 N and 369.9 kW. Those came from a review script that
> evaluated the row at `W = 21.4972` — a mistyped `21.490743`. The mass flow
> itself was never wrong (it was verified two independent ways), only the
> quantities derived from it. The values above are produced by
> `q1d.analytic.zero_d_compressor` and pinned in `tests/test_analytic.py`.

`W_ff` in the prototype's map generator — described there as a "choked flow"
reference — is in fact **exactly this number**. It is the mass flow through A₂
when the exit static pressure equals p₀₁, which is precisely the back-pressure
condition the solver imposes. M₂ = 0.517; nothing is choked.

Choke limits for reference: station 1 at 24.1201 kg/s, station 2 at
28.1208 kg/s. The operating point sits at 89% of the inlet choke limit.

### 3.2 Ideal-gas source terms are exactly inlet-invariant

Non-dimensionalizing by `p₀₁·A₁` and `W·cp·T₀₁`:

```
F̂ ≡ Fx/(p₀₁·A₁) = (pₛ₂/p₀₁)(A₂/A₁)(1 + γM₂²) − (pₛ₁/p₀₁)(1 + γM₁²)
Ŝ ≡ SWx/(W·cp·T₀₁) = τ − 1
```

with M₁ from `Φ₁ = W√(R·T₀₁)/(A₁·p₀₁)` and M₂ from
`Φ₂ = Φ₁·(A₁/A₂)·√τ/PR`. Both are pure functions of Φ₁ given γ, A₂/A₁ and the
map. Verified across three inlet conditions (101325/288.15, 60000/250,
150000/320) whose raw `Fx` differ by a factor of 2.9:

```
max relative spread of F̂ :  ~1e-15  (machine precision)
Ŝ  identical by construction
```

**This invariance is a test of the ideal-gas path, not an architectural
assumption.** It does not survive real gas (§3.3), which is why the source
terms are computed at runtime rather than tabulated.

### 3.3 The invariance does not survive NASA9

With variable cp the one-parameter collapse degrades to a two-parameter
surface:

- `Ŝ` becomes `[h(T₀₂ₛ) − h₀₁]/(η·h₀₁)` where `s°(T₀₂ₛ) = s°(T₀₁) + R·ln(PR)`
  — a function of **T₀₁ and PR**, not PR alone.
- `F̂` becomes `F̂(Φ₁, T₀₁)`, since the flow-function and static/total relations
  all carry γ(T).

Correct real-gas forms, exact in both regimes:

```
SWx = W·(h₀₂ − h₀₁)
h₀₂ = h₀₁ + (h(T₀₂ₛ) − h₀₁)/η ,   s°(T₀₂ₛ) = s°(T₀₁) + R·ln(PR)
```

### 3.4 Measured cost of the alternatives

Over 33,750 source evaluations (a full 0.5 s run at the prototype's settings):

| approach | cost |
| --- | --- |
| Full runtime inversion (2× `setStatic`, 24 + 17 iterations) | 0.53 s |
| Dimensionless table + `np.interp` | 0.11 s |
| **Prototype's `interp1d` linear scan** | **3.45 s** |
| `entropy_corr` Python loop (3×/stage) | 2.61 s |
| `entropy_corr` vectorized with `np.where` | 0.56 s |

The precomputed table **cost 6.5× more than computing the physics inline**.
The linear-scan lookup is 30× more expensive than the fixed-point inversion it
was introduced to avoid. Performance therefore does not motivate precomputing
the source terms; correctness and structure decide it.

---

## 4. Prototype review findings

### 4.1 Confirmed correct — do not churn

- **`Fx` and `SWx` are correctly independent.** A common failure is writing
  `q_energy = u·F_blade`, which silently builds an isentropic actuator disk
  with nowhere for η to enter. The prototype does not do this; the entropy rise
  emerges correctly from the (Fx, SWx) pair.
- Momentum-theorem sign convention, consistent with `rhs[1] -= q1`.
- Characteristic BC algebra at both ends, verified independently against the
  standard derivation.
- Dimensional bookkeeping through the update (N → N/m via `dt/dx`), and
  `p·dA` as the geometric momentum source.
- Index bookkeeping: 99 interior cells, 100 faces, one ghost each end,
  `im1 = 99` is the last interior cell.
- Low-storage RK structure (each stage restarting from `cvold`).

### 4.2 Defects, by severity

| # | Finding | Severity | Fixed in |
| --- | --- | --- | --- |
| 1 | Source terms tabulated as dimensional quantities against a single abscissa — valid only at the generating inlet condition | Critical (architectural) | P3 |
| 2 | Lookup keyed on the **downstream** state the source itself produces — explicit feedback loop, one cell from the discontinuity, inside an RK stage | High | P3 |
| 3 | `interp1d` linear scan: 3.45 s, 30× the cost of the physics it replaces | High (perf) | P2 |
| 4 | `entropy_corr` scalar Python loop: 2.61 s | High (perf) | P2 |
| 5 | `setStatic` fixed-point iteration converges at rate M²; silently returns non-converged state above M ≈ 0.90. Confirmed: hits the 100-iteration cap at M₁ = 0.9243 | High | P1 |
| 6 | Geometric source overwritten rather than accumulated at the disk cell (`q1[50] = ...` should be `+=`). Latent — harmless only because `da = 0` | High (latent) | P2 |
| 7 | `uref` taken from the IC where u = 1.28e-4 m/s — a factor 1.4e6 below the operating velocity, so the velocity limiter never disengages | Medium | P2 |
| 8 | `volref = 1.0` against a 1.01e-3 m³ cell scales all limiter thresholds by 3.2e-5 | Medium | P2 |
| 9 | IC derived from a 1e-8 Pa pressure difference: M = 3.8e-7 (450× machine epsilon); `e = cv·T₀` uses stagnation temperature; `ρEA` omits u²/2 | Medium | P2 |
| 10 | No residual norm, no convergence test — fixed `tend = 0.5` regardless of whether the solution converged at 0.15 s | Medium | P2 |
| 11 | No supersonic inlet handling; `dis` clamped to 1e-20 silently | Medium | P2 |
| 12 | Derivative tables `dFxdW`/`dSWxdW` computed, plotted, and never saved | Low | superseded by P3 |
| 13 | `diss`, `rs`, `ls` allocated and unused; `dt` not clipped at `tend`; `print(t)` every step | Low | P2 |
| 14 | `print('choked')` when the state is actually *infeasible*, not choked | Low | P1 |

Limiter detail for #7/#8 — measured threshold vs. typical smooth-region
gradient:

```
rho: eps2n=1.62e-04  vs  du²=1e-06   -> limiter OFF in smooth regions (intended)
u  : eps2n=1.77e-12  vs  du²=1e-02   -> limiter ON everywhere         (defect)
p  : eps2n=1.11e+06  vs  du²=1e+04   -> limiter OFF in smooth regions (intended)
```

Disengaging the limiter in smooth regions is the *purpose* of Blazek's ε². Only
the velocity threshold is broken, and only because of `uref`.

### 4.3 Findings rejected on review

Raised during earlier review; examined and found incorrect. Recorded so they do
not get re-raised.

- **"`Fx` has no zero crossing, therefore no equilibrium exists."** `Fx` is the
  blade force on the fluid. A compressor always pushes the fluid; `Fx` should be
  positive everywhere and has no reason to cross zero. Equilibrium is a
  vanishing *residual*, not a vanishing source. The blade force is balanced by
  the static pressure rise it creates, which lives in the flux term
  `(ρu² + p)A`. At steady state
  `f₁|ᵣ − f₁|ₗ = [W·u₂ + pₛ₂A₂] − [W·u₁ + pₛ₁A₁] ≡ Fx` identically. The
  equilibrium exists, is unique and subsonic, and is tabulated in §3.1.
- **"Corrected-flow coordinate mismatch causes a ~17% steady-state error."**
  At convergence the sampled cell sits downstream of the disk at
  T₀ = 305.27, p₀ = 121590, so the runtime coordinate evaluates to 0.8577·W —
  identical to the table. The discrepancy exists only at t = 0. The real defect
  at that location is the feedback loop (§4.2 #2), not a scale factor.
- **"Ghost-cell speed of sound is stale."** `dv[:,0]` and `dv[:,-1]` are never
  read. Reconstruction uses `cv` and `p` directly, `BC()` reads interior cells
  1 and `im1`, and the timestep slices `lambdac[1:-1]`. Harmless.
- **"Limiter thresholds are miscalibrated by ~18 orders of magnitude,
  critical."** Overstated — see §4.2. Two of the three behave as designed.

### 4.4 Design note: use Φ₁, not corrected flow, as the map key

```
Φ₁  = W√(R·T₀₁)/(A₁·p₀₁)            — no reference state
Wc₁ = W√(T₀₁/288.15)/(p₀₁/101325)   — carries an arbitrary reference
```

They differ by a constant. The prototype's t = 0 abscissa mismatch was possible
*only* because corrected flow carries a reference state that can be chosen
inconsistently between generation and runtime. Φ₁ has no reference state, so
that bug class becomes unrepresentable. **Φ₁ is the internal key; Wc₁ is
carried as a display column** for plotting against real map data.

Caveat: Φ₁(M₁) is monotonic only for subsonic flow, and dΦ/dM → 0 as M → 1, so
the inversion is ill-conditioned near choke. Not an issue at M₁ = 0.665; it is
what will bite when pushing toward the choke line.

---

## 5. Phases and gates

### Phase 0 — Skeleton and baseline ✅ **complete**

- Package layout, `pyproject.toml`, `pytest`, `ruff`.
- Both original scripts land in `legacy/`, **frozen and never edited**, so
  every later result can be diffed against original behaviour. Provenance and
  transcription verification in `legacy/README.md`.
- Profile the legacy run; record baseline wall time and hot spots.

**Gate met.** See `BASELINE.md`. Headline results:

- **14.77 s / 10,915 steps / 1.35 ms per step**, flat per-step cost.
- **The legacy solver works**: W = 21.492878 against the analytic 21.490743,
  an error of +0.0099%. Several findings in the original review implied no
  steady operating point was reachable; one is reachable and it is reached.
- The far field is uniform to ~1e-8 on both sides of the disk, confirming that
  the Phase 3 gate of 1e-8 is achievable rather than aspirational.
- The 1e-4 error is traced quantitatively to the cell-52 lookup station lying
  inside the smeared region (`pt` low by 3.4e-4 → ~0.3 N of force error out of
  1731 N). This confirms §4.2 finding #2 and is removed by sampling upstream.
- **Convergence-based stopping is worth 3.2x**, larger than every
  micro-optimisation combined. `tend = 0.5` is ~3x longer than needed.

Two corrections to earlier figures in this document, recorded rather than
silently amended:

- An initial baseline of 386 s was a harness artefact (capturing the script's
  10,915 per-step `print` calls), not solver cost. True cost is 14.77 s.
- The §3.4 microbenchmarks are correct as per-call costs but overstated the
  share of runtime they represent. At the true step count they account for
  ~0.7 s of 14.77 s, not the majority. Phase 2 priorities in `BASELINE.md` are
  reordered accordingly.

### Phase 1 — 0D analytic reference ✅ **complete**

- `q1d.gas.PerfectGas` — thermodynamics behind an object rather than module
  globals, shaped as the seed of the Phase 4 `Gas` interface (P3).
- `q1d.analytic` — flow function with closed-form derivative, bracketed Newton
  for Φ → M on both branches, station states, choke limits, compressor
  stagnation states, source terms, and `zero_d_compressor`.
- Explicit feasibility reporting: `InfeasibleOperatingPoint` distinguishes a
  non-existent state from a *choked* one, which is a valid physical state.

**Gate met.** 64 tests, all passing, `ruff` clean. Covered: the `W_ff`
identity, the reference operating point to 1e-12, both choke limits, internal
self-consistency (`pₛ₂ == p_b`), the §3.2 invariance collapse, and regression
against five `legacy/` map points plus two `setStatic` calls.

Two findings from the phase:

- **§3.1 carried wrong values for M₁, Fx and SWx** — corrected above.
- **With `A₁ = A₂` the exit can never choke before the inlet.** Exit choke
  demands `Φ₁/Φ₁ᵐᵃˣ = (A₂/A₁)·PR/√τ`, which is 1.166 at equal areas, so
  station 1 always saturates first. The exit only chokes first when
  `A₂/A₁ < √τ/PR = 0.8577`. Relevant to D8: the equal-area assumption makes
  the inlet the sole choke-limited station, which simplifies Phase 5's choke
  handling.

The key structural test is
`test_source_terms_reconstruct_station_2_from_station_1`: it solves the
mass/momentum/energy balance independently of `analytic.py` (as a quadratic in
exit velocity) and recovers station 2 to 1e-10. That identity is what makes the
Phase 3 gate reachable — the Q1D solver's discrete conservation is exact, so
the downstream uniform state is whatever this reconstruction gives.

### Phase 2 — Solver core, optimized, source terms OFF

Performance (measured wins from §3.4):
- `np.where` for `entropy_corr`, `np.interp` for lookups.
- Preallocation and `out=` on hot arrays; fuse operations to cut numpy
  per-call overhead, which at 99 cells exceeds the actual arithmetic.
- Throttled diagnostics; remove dead allocations.
- Residual-norm history with convergence-based stopping.
- Establish the true stability limit empirically: the `ark` coefficients are
  Blazek's hybrid set, valid for dissipation evaluated at stages 1/3/5 only,
  whereas the prototype evaluates it at every stage. The CFL limit those
  coefficients were tuned for therefore does not apply.

Correctness:
- `uref`/`volref` from a reference state, not from the IC.
- Physically sane initialization.
- `+=` on the geometric source.
- `dt` clipped at `tend`.
- Supersonic branches at both boundaries; no silent clamping.

Expected outcome: ~5–10× over baseline (numpy only, no JIT).

**Gate — with no source term anywhere in the code:**
1. Uniform flow preserved to machine zero in a **varying-area** duct
   (well-balancedness of the `p·dA` term).
2. Sod shock tube against the exact Riemann solution.
3. Steady converging–diverging nozzle against the analytic area–Mach relation.

This establishes that the discretization is sound *before* any compressor
physics can be blamed for anything.

### Phase 3 — Actuator disk, ideal gas, runtime evaluation

- `Gas` interface; `IdealGas` implementation.
- `CompressorMap` returning `PR, η` from `Φ₁` (constant values initially).
- Source computed at runtime from the **upstream**-sampled state.
- Injection through the smear distribution, `n_smear = 1` by default.

**Gate:** converged Q1D mass flow matches the 0D algebra to **≤ 1e-8 relative**
across a matrix of ambient conditions (p₀₁, T₀₁, p_b, PR, η), plus mesh
convergence.

The tolerance is deliberately tight and is justified: at steady state the flow
upstream and downstream of the disk is uniform, so the Roe dissipation vanishes
identically and p₀ is preserved to machine precision; global discrete
conservation makes the jump across the smeared region exactly (Fx, SWx); and
Fx, SWx are constructed so that applying them to the station-1 state *produces*
the station-2 state. Anything looser than 1e-8 indicates a real defect.

Because the ideal-gas source terms are exactly inlet-invariant (§3.2), sweeping
ambient conditions tests the **interface** — whether local state is measured and
applied correctly — rather than re-testing the physics. That is precisely where
the original defect lived.

### Phase 4 — Real gas (NASA9)

- NASA 9-coefficient polynomials for `cp°/R`, `H°/RT`, `S°/R` with their
  temperature intervals.
- Tabulated `h(T)`, `s°(T)`, `cp(T)` and the inverses `T(h)`, `T(s°)` on a
  temperature grid, so every runtime inversion becomes an interpolation rather
  than a Newton solve.
- Consistent real-gas Roe average (Vinokur–Montagné or Glaister).
- Reformulated characteristic BCs (the `2c/(γ−1)` invariant does not survive).
- `cons_to_prim` via `e(T) = h(T) − RT` inversion — the dominant new cost.

**Gate:** real-gas 0D ↔ Q1D match at the Phase 3 tolerance, **and** exact
recovery of the Phase 3 ideal-gas results when NASA9 is replaced by constant
cp — a regression on the entire thermodynamic layer.

### Phase 5 — Real compressor maps

- **Phase 5a — ICMF with β-lines.** The first maps supplied are inlet corrected
  mass flow with β as a dimension: `PR(β, N)`, `Wc₁(β, N)`, `η(β, N)`. Lookup
  means solving for β such that `Wc₁(β, N)` matches the measured value.
  Ingestion, interpolation, and an explicit extrapolation policy.
- **Phase 5b — ECMF as the choke-line remedy.** See §7 Q3. Adopted only if
  clamp-and-warn at the choke line proves insufficient.
- This is where `∂PR/∂Φ₁ < 0` finally supplies aerodynamic stiffness and surge
  dynamics become structurally representable — impossible with a constant-PR
  map, which has zero stiffness by construction.

**Gate:** speedline reproduction, throttle sweep, and stable operation across
the map.

### Beyond

Turbine (same source formulation, opposite signs), multi-component flowpaths,
implicit time integration.

---

## 6. Decisions log

| # | Decision | Rationale | Date |
| --- | --- | --- | --- |
| D1 | Source injection, not internal boundary conditions | Native to the quasi-1D FV formulation; exact integral balance; no node splitting. Smearing over several cells available if needed, and opens finite-length blade-row modelling | 2026-07-25 |
| D2 | Python for the reference implementation; C++ ported separately by the author | Prototyping speed; the Python version is the oracle, not the product | 2026-07-25 |
| D3 | **No Numba.** numpy vectorization only | The production solver is C++ regardless; a JIT dependency buys speed the prototype does not need. Accepts ~5–10× instead of ~50× | 2026-07-25 |
| D4 | Source terms computed at runtime; only gas properties and maps are tabulated | The dimensionless collapse does not survive NASA9 (§3.3), and runtime evaluation costs 0.42 s (§3.4). One code path for both gas models | 2026-07-25 |
| D5 | Φ₁ as the internal map key; Wc₁ as a display column | Φ₁ carries no reference state, making the reference-mismatch bug class unrepresentable | 2026-07-25 |
| D6 | **Consistent real gas, not frozen-γ** | A turbine at 1600 K downstream of a compressor at 300 K spans a γ range where the inconsistency stops being second-order. (Note: frozen-γ evaluates γ(T) locally from real cp — it is not "γ = 1.4 always" — but it still holds γ constant within the Roe average and the Riemann invariants) | 2026-07-25 |
| D7 | Legacy scripts frozen in `legacy/`, never edited | Preserves a diffable reference for original behaviour | 2026-07-25 |
| D8 | `A₁ = A₂` through Phase 4; variable area is its own later step | Matches the prototype and the available compressor data (inlet and outlet areas only). Avoids reconciling the `p·dA` source with the map's station definitions inside the smeared region before it is needed | 2026-07-25 |
| D9 | ICMF + β-lines first; ECMF held in reserve for the choke line | The supplied maps are in ICMF+β form. ECMF is the better parameterization (§7 Q3) but requires closing a downstream-state feedback loop properly, which is deferred until choke behaviour demands it | 2026-07-25 |
| D10 | Steady-state acceptance only, extended with a hold test | No transient reference data exists yet. A solver that reaches the analytically known point and holds it is accepted for now; revisited when transient data becomes available | 2026-07-25 |

Dependencies: `numpy`, `matplotlib`, `pytest`. `scipy` optional — root-finding
will be hand-written and guarded to avoid the dependency unless it earns its
place.

---

## 7. Resolved questions

### Q1 — Sampling distance *(resolved: measurement decides)*

How many cells upstream of the disk should Φ₁ be sampled? Far enough to clear
the smeared region, near enough to limit transient lag. Implemented as a
parameter, default 3 cells upstream, with a sensitivity sweep in Phase 3. If
the converged answer moves with sampling distance, that is itself diagnostic —
the Phase 3 gate says it should not.

### Q2 — Area change across the disk *(resolved: A₁ = A₂ for now)*

Only inlet and outlet areas of the compressor are available, and the prototype
used equal areas. Equal areas through Phase 4; variable area becomes its own
step, at which point the `p·dA` source inside the smeared region must be
reconciled with the map's station definitions. See D8.

### Q3 — Choke-line behaviour *(resolved: ICMF+β first, ECMF in reserve)*

Φ₁ → M₁ inversion is ill-conditioned as M₁ → 1 because dΦ/dM → 0 (§4.4). More
fundamentally: **when the inlet is choked, inlet corrected flow does not
determine the operating point** — the speedline is vertical, W is pinned, and
PR is set by downstream conditions. No reparameterization of inlet quantities
fixes this; it is physics. β-lines are a well-posedness device for
interpolation, not a resolution of the underlying degeneracy.

Exit corrected mass flow does resolve it:

```
Wc₂/Wc₁ = √τ / PR
```

On a choked segment `Wc₁` is constant while PR sweeps, so `Wc₂ ∝ 1/PR`
decreases strictly monotonically — the vertical segment unwraps. Traversing a
full speedline from choke to surge, `Wc₁` falls and PR rises, so `Wc₂` falls
monotonically throughout and `PR(Wc₂)` is single-valued over the entire line.

The cost is that looking up on `Wc₂` requires `T₀₂` and `p₀₂`, which are the
map's outputs. In a 0D deck that is circular. In this solver it is not — the
downstream state is available from the flow field, so it is a measurement
rather than a prediction. But it is structurally the same downstream feedback
that produced defect §4.2 #2, and it must be closed properly (sub-iterated
within the RK stage, or relaxed) rather than left explicit across stages.

**Plan:** ICMF+β through Phase 5a with clamp-and-warn at the choke line;
ECMF adopted in Phase 5b only if clamping proves insufficient. See D9.

### Q4 — Transient validation *(resolved: steady-only for now)*

No transient reference data exists. Acceptance is steady-state, extended with
an explicit **hold test** so that "holds the point steadily" is a measurement
rather than a judgement:

> After the residual reaches its floor, continue for 10× the number of steps
> taken to converge. Require that the residual stay at its floor (no secular
> growth) and that mass flow drift by less than the Phase 3 tolerance over that
> interval.

This catches slow instabilities that a converged-and-stopped run hides —
including exactly the kind a downstream-keyed source term would produce. See
D10.

## 8. Known gaps

- **No transient acceptance criterion.** The solver's purpose is transient
  response and every gate is steady-state. The hold test (Q4) demonstrates
  stability, not transient *accuracy*. To be revisited when transient data
  becomes available. Candidate interim check: acoustic reflection timing
  against characteristic theory, which validates wave propagation but says
  nothing about the disk's own transient behaviour.
- **`legacy/` provenance.** The frozen reference must be the author's original
  `.py` files. Code reconstructed from a PDF rendering is not a reference.
