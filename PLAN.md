# Quasi-1D Euler Solver with Turbomachinery Source Terms — Development Plan

Status: **Phase 3 complete, Phase 4 next**
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
150000/320) whose raw `Fx` differ by a factor of exactly 2.5 (at fixed Φ₁,
`Fx ∝ p₀₁`, so the ratio is 150000/60000):

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
| 6 | ~~Geometric source overwritten rather than accumulated at the disk cell (`q1[50] = ...` should be `+=`)~~ **This finding was wrong — see §4.5.** The legacy overwrite is correct given how `Fx` is defined | withdrawn | — |
| 6b | `Fx` is the *total* momentum-flux jump, which already contains the wall reaction `∫p·dA`; it is the blade force only when `A₁ = A₂`, and goes negative for a contraction | High (latent) | P1 ✅ |
| 7 | `uref` taken from the IC where u = 1.28e-4 m/s — a factor 1.4e6 below the operating velocity, so the velocity limiter never disengages | Medium | P2 |
| 8 | ~~`volref = 1.0` against a 1.01e-3 m³ cell scales all limiter thresholds by 3.2e-5~~ **This finding was wrong — see §4.6.** `volref = 1.0` is a unit normalisation and the scaling is the intended mesh dependence | withdrawn | — |
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

### 4.6 Correction: `volref = 1.0` is correct

Found in Phase 2, and it withdraws finding #8. The threshold is

```
eps² = limfac³ · ref² · (vol_cell / volref)^1.5
```

The `vol_cell^1.5` factor is the Venkatakrishnan/van Albada **smoothness
parameter** `ε² ~ (K·Δx)³`. It is *supposed* to shrink with mesh size, so that
the limiter disengages only where the solution varies less than the local mesh
scale and engages at discontinuities. `volref` is a unit normalisation, not a
representative cell volume.

Setting `volref` to a typical cell volume — the "fix" this document originally
called for — scales all three thresholds up by ~3e4 on a 100-cell grid,
switching the limiter **off across shocks**. Doing so produces negative
pressures on a Sod shock tube within a few hundred steps.

The genuine defect is #7 alone: `uref` taken from a near-stagnant initial
condition. `ReferenceState` now documents `vol` as a normalising volume and
defaults it to 1.0.

### 4.5 Correction: `Fx` already contains the wall reaction

Found in the Phase 1 audit, and it withdraws finding #6 above.

Integrating the quasi-1D momentum equation across the machine:

```
d/dx[(ρu² + p)A] = p·dA/dx + f_blade
  ⟹  Δ[(ρu² + p)A] = ∫p·dA + F_blade
```

`Fx = pₛ₂A₂ − pₛ₁A₁ + W(u₂ − u₁)` is the **left-hand side** — the total axial
momentum source, blade force *plus* wall pressure reaction. Consequences:

- **Finding #6 is withdrawn.** Changing `q1[50] = Fx` to `+=` would add
  `p·dA` on top of a quantity that already contains it, double-counting the
  wall term. The legacy overwrite was right. The correct rule is: *inject the
  total momentum source and do not add the geometric term in that cell*, or
  *subtract `∫p·dA` first and then accumulate*. Either is consistent; mixing
  them is not.
- **`Fx` is not positive everywhere.** At `A₁ = 0.1, A₂ = 0.05, W = 10` it is
  **−3709.91 N**, because the wall term dominates. The §4.3 rebuttal below is
  a *constant-area* statement.
- A zero-thickness actuator disk has one area by definition, so
  `compressor_source_terms` now rejects `A₁ ≠ A₂` unless the caller passes
  `wall_pressure_integral` explicitly. This makes the accounting a decision
  rather than an accident, and it is what Phase 3 will rely on.

### 4.3 Findings rejected on review

Raised during earlier review; examined and found incorrect. Recorded so they do
not get re-raised.

- **"`Fx` has no zero crossing, therefore no equilibrium exists."** The
  conclusion does not follow, though the premise needs the qualifier added in
  §4.5: at **constant area** `Fx` is the blade force on the fluid, a compressor
  always pushes, and `Fx` stays positive. Equilibrium is a
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
mass/momentum/energy balance (as a quadratic in exit velocity) and recovers
station 2 to 1e-10. That identity is what makes the Phase 3 gate reachable —
the Q1D solver's discrete conservation is exact, so the downstream uniform
state is whatever this reconstruction gives.

**Narrowing an earlier overstatement:** this test is independent of
`analytic.py`'s *code path*, not of its *definitions*. `Fx` and `SWx` are
constructed from station 2, so recovering station 2 is a bookkeeping identity —
it is blind to errors in the compressor thermodynamics themselves (τ built with
`·η` instead of `/η` passes it). It remains a strong test of the
source-term ↔ station bookkeeping; the compressor relation is pinned separately
by `test_exit_stagnation_matches_legacy`.

#### Phase 1 audit

An independent adversarial audit re-derived the physics from first principles,
recomputed every §3.1 value without importing `q1d` (agreement ≤ 4.4e-15,
including a 60-digit `Decimal` cross-check), and ran 36 mutations against the
suite. It confirmed the §3.1 corrections and the equal-area choke result above.
Defects it found, all now fixed:

| finding | fix |
| --- | --- |
| **NaN passed silently through the Mach inversion**, returning M = 0.99999999999999 — every comparison against NaN is False, so it slipped both range checks and bisected onto the sonic point | explicit `isnan` guard; tested at all three entry points |
| **`Fx` includes the wall reaction** (§4.5) | documented; `A₁ ≠ A₂` now rejected unless `wall_pressure_integral` is given |
| **Newton had degraded to bisection.** The bracket was shrunk onto the current iterate *before* stepping from it, so near the root the step landed on the bracket boundary and was rejected: 40 iterations at M = 0.5, 38 of them fallbacks | reordered to the `rtsafe` form (NR §9.4). Now **5–13** evaluations across the range; guarded by a `maxiter` cap below what bisection would need |
| Inlet-choked back pressures reported as generic infeasibility | dedicated `InletChokeLimited` carrying `W_choke` and `pb_min` |
| `PR < 1` silently applied the compressor efficiency convention to an expansion | rejected with a message naming the turbine convention |
| Two `# pragma: no cover` comments asserted unreachable branches that are reachable | comments corrected, limits named as constants |
| Feasibility slack inconsistent between call sites | single `FEASIBILITY_SLACK` constant |
| §3.2's "factor of 2.9" | it is exactly 2.5 |

Tautological tests it identified, all replaced or strengthened:

- `test_choked_mass_flow_is_exactly_sonic` was **fully circular** —
  `choked_mass_flow` multiplies by `max_flow_function` and
  `static_from_stagnation` divides it back out, so replacing
  `max_flow_function` with any value left the test green. Now checked against
  the independent closed form and pinned absolutely.
- The invariance collapse test **could not see a wrong `Fx` at all**: a
  sign-flipped momentum term still collapses to 2e-15, because the collapse
  only tests dimensional homogeneity. F̂ and Ŝ are now pinned absolutely.
- The exit-choke regime boundary was untested (the only test drove M₂ ≈ 1.56,
  so a threshold moved to 0.95 or 1.05 survived). Now bracketed at ±0.1%.
- No test asserted `Fx` for `A₁ ≠ A₂`, so swapping the two areas survived.

Suite: **88 tests**, `ruff` clean.

### Phase 2 — Solver core, optimized, source terms OFF ✅ **complete**

Modules: `q1d.grid` (explicit ghost-cell layout), `q1d.boundary` (conditions as
objects, with the supersonic branches the legacy code lacked), `q1d.solver`
(Roe + MUSCL + 5-stage RK, residual norms, convergence-based stopping),
`q1d.riemann` (exact Riemann solver, verification only).

**Gate met**, with no source term anywhere in the tests:

1. **Well-balancedness — bitwise zero.** The momentum residual of a stagnant
   varying-area duct is exactly `0.0`, both reconstruction orders. Achieved by
   writing the geometric source as `p·A_r − p·A_l` rather than `p·(A_r − A_l)`:
   floating-point multiplication does not distribute, so the two differ by an
   ulp and only the first cancels the flux term exactly.
2. **Sod against the exact Riemann solution.** L1 errors 2.0e-3 (ρ) at n=400,
   converging at ~1st order as expected for captured discontinuities, with
   2nd-order reconstruction beating 1st.
3. **Steady nozzle against the area–Mach relation.** Face mass flux uniform to
   1e-12; every cell on the analytic Φ(M) curve to 2e-4.

Performance: **14.77 s → 5.76 s (2.6x)**. Full analysis in `BASELINE.md`.

#### Findings

- **CFL is not a single number.** The limit *falls with mesh refinement* —
  2.475 (n=40), 2.452 (80), 2.440 (160), 2.435 (320) at order 2 — and order 1
  is far more forgiving at ~3.13. The legacy default of 2.5 is past it on every
  mesh. Default now 2.0, verified stable at n=80/160/320 over 5000 steps.
- **Two review findings withdrawn, both mine** — §4.5 (`Fx` contains the wall
  reaction, so `+=` on the geometric source would double-count) and §4.6
  (`volref = 1.0` is a unit normalisation; "fixing" it switches the limiter off
  across shocks). The genuine limiter defect was `uref` alone.
- **A sign error in the new boundary code**, not the legacy one: subsonic
  outflow used `J ∓ 2c_b/(γ−1)` with the sign inverted at both ends.
- **The scheme is not strictly monotone for systems.** Density, pressure *and*
  velocity overshoot/undershoot at the shock foot and rarefaction head (0.5%,
  0.7%, 0.8%). All five shrink monotonically with `limiter_factor`, which
  attributes them to the van Albada smoothness parameter rather than the
  Riemann solver. Bounds are measured, not chosen.
- **Nozzle stagnation-pressure loss converges at ~3rd order** (1.10e-3 → 1.99e-6
  at n = 100…800) — numerical entropy generation through the throat, not a
  spurious source. A plateau would have corrupted the Phase 3 map lookup.

#### Phase 2 audit

An adversarial audit ran 63 mutations and re-derived the scheme against the
frozen legacy. **It confirmed the numerics are the legacy numerics** — `rhs` is
bitwise identical term by term for {constant, bump} × {order 1, 2}, the only
intended difference being the geometric source form. It also found that
**three of the gate's headline claims were wrong**, all of them mine:

| claim | reality |
| --- | --- |
| "CFL 2.44 stable, 2.46 not" as a scheme property | Measured at (n=80, 400 steps), where the instability is invisible. **2.44 diverges at n=160** — it reads 4.6e-5 at 400 steps and 6.6e-2 by 5000. |
| "well-balanced to bitwise zero" | True only for a *bitwise-uniform pressure field*. `set_state` stored `p` instead of deriving it from `cv`, so the gate measured a state the time-stepper never visits. Realised residual is 1.8e-12 (order 1) / 3.6e-12 (order 2). |
| "1.19x faster per step" | Timed `advance()` without the `residual_norm()` that `run()` called every step. Counting it, the shipped code ran at **1.472 ms/step — slower than legacy's 1.350**. |

Two test assertions were also vacuous, asserting bounds the measured quantity
beat by six orders of magnitude, and all three equations were normalised by
`p·A` when the energy flux scale is ~1e6 W against ~1e4 N — making a one-ulp
energy residual look 100× worse than a one-ulp momentum one.

Fixed since: `set_state` derives `p` from `cv`; the well-balancedness gate is
split into the scheme property (bitwise, hypothesis stated) and the realised
state (round-off, per-equation scales); CFL constants are split by order and
documented as upper bounds needing margin; `record_every` defaults to 10;
`run()` rejects `local_time_stepping` with `t_end` and returns converged for an
already-stagnant field instead of silently turning a relative tolerance into an
absolute SI one; `Grid` validates what it previously assumed; supersonic inflow
rejects a non-supersonic imposed triple.

**Coverage the audit found missing, now closed:** `rhs -= self.source(self)`
had never executed in any test — the entire interface Phase 3 is built on — and
no supersonic boundary branch had ever run despite being advertised above as an
improvement over the legacy code. Both now have dedicated tests, including the
source integral balance (`dt·Σq` exactly) that the Phase 3 tolerance rests on.
An asymmetric area, a stretched mesh and both reconstruction orders were added
after mutation testing showed six geometry bugs surviving because every grid
was uniform and every area symmetric.

#### Caveat carried into Phase 3

The 1e-8 Phase 3 tolerance is a **constant-area** statement. `BASELINE.md`
shows the constant-area far field uniform to 1e-8 because the Roe dissipation
vanishes identically there. With variable area, the ~3rd-order stagnation
pressure loss above enters, so the mesh must be fine enough to keep that term
below the tolerance. Relevant when D8 (equal areas) is eventually relaxed.

### Phase 3 — Actuator disk, ideal gas, runtime evaluation ✅ **complete**

`q1d.compressor`: a `CompressorMap` protocol returning `PR` and `η` from the
dimensionless inlet flow function `Φ₁`, and an `ActuatorDisk` that computes
`Fx` and `SWx` **at runtime** from the locally measured upstream stagnation
state. Nothing dimensional is tabulated — P1 realised in code.

**Gate met.** The Q1D solver reproduces the closed-form 21.490742688 kg/s to
≤1e-8, across five ambient conditions, three mesh densities (59/99/199), three
smear widths, and a 10× hold with no drift. `T₀₂` is pinned independently of
`p₀₂`, so a disk that ignored efficiency cannot pass: only `T₀₂` carries η, and
the isentropic value differs by 0.564% — five orders above the gate.

#### The sampling standoff, measured

`PLAN.md` §7 Q1 deferred this to measurement, and the measurement mattered. The
disk creates a **numerical boundary layer extending upstream as well as
downstream** — an artefact of the reconstruction stencil, not physics — whose
influence decays by ~10× every two to three cells:

| offset | W error | | offset | W error |
| --- | --- | --- | --- | --- |
| 1 | −3.3e-04 | | 8 | +2.4e-08 |
| 3 | −1.8e-05 | | 12 | +7.5e-11 |
| 5 | −1.0e-06 | | 16 | +2.3e-13 |

**The initial default of 3 sat inside that layer**, so the disk read
`p₀₁ = 101318` instead of `101325` and the converged mass flow was wrong by
1.8e-5 — the prototype's cell-52 defect (§4.2 #2) mirrored onto the upstream
side. Default is now 12, three orders inside the gate. The unit is *cells*, not
length, because the contamination comes from the stencil and does not shrink
under refinement.

#### Other results

- **Smearing does not move the operating point.** `n_smear` of 1, 3 and 7 all
  land on the same mass flow to 1e-8: global conservation is exact regardless
  of how the source is distributed, so only the local profile changes. This is
  the property that makes finite-length blade rows a parameter change (P2).
- **The disk refuses placement where the area varies across its cells.** A
  zero-thickness disk has one area; otherwise the momentum source carries the
  wall reaction as well as the blade force (§4.5).
- **A caveat inherited from Phase 2:** the 1e-8 tolerance is a constant-area
  result. With variable area the ~3rd-order stagnation-pressure loss enters and
  the mesh must be fine enough to keep it below tolerance.

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
  *Mitigated:* the transcription reproduces six values computed independently
  beforehand, and the solver reaches the analytic answer to +0.0099%.
- **The inlet-choked branch is not modelled.** `zero_d_compressor` solves the
  subsonic-matched problem only. Below `minimum_back_pressure` (93839 Pa at the
  reference conditions) the inlet chokes, mass flow pins at 24.1201 kg/s, and
  the back pressure stops setting the operating point — the model raises
  `InletChokeLimited` rather than guessing. Whether a constant-area duct with a
  sonic inlet and subsonic exit is even well posed is a genuine physics
  question, not just a missing feature; deferred until Phase 5 needs it.
- **Accuracy limits of the Mach inversion**, measured: worst subsonic absolute
  error 1.4e-12 at M = 0.9998 (consistent with the documented `O(√δ)`
  conditioning near choke); relative error degrades to 5e-10 as γ → 1
  (`PerfectGas(1.0001, 4000)` at M = 0.999), well outside any gas of interest
  but worth knowing before NASA9 arrives.
- **Supersonic bracket is capped** at `MAX_SUPERSONIC_MACH = 1e4`; flow
  functions below ~1.5e-18 are rejected rather than solved.
- **Mutations that still survive the suite** (from the Phase 2 audit, not yet
  addressed). *These describe gaps in the TEST SUITE, found by deliberately
  breaking a throwaway copy of the code to see whether the tests notice. The
  solver itself is unmodified — in particular the Harten entropy fix is present
  and unchanged, as it has been throughout.* Three Roe
  average mutations (`dd = rav/rr`, swapped `uav`, `q2a = uav²`) shift Sod L1 by
  3–4% against a 6e-3 bound that permits 200%. Two boundary mutations survive:
  writing the ghost from interior cell 2 instead of 1, and dropping the kinetic
  term from ghost energy. Reverse inflow is never exercised, so the inflow
  velocity sign is unconstrained. These want a dedicated Roe-average jump
  condition test, a sonic-point case, and a ghost-state consistency test.
- **No positivity safeguard.** Toro test 2 (the near-vacuum "123 problem")
  raises `NonPhysicalState` with the entropy fix on or off. Expected for a bare
  Roe solver, but untested and unguarded.
