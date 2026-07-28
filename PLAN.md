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

### 3.5 The supplied maps make ECMF mandatory, not optional

Measured on `HPC02.xlsx` and `SingleStgTurbine.xlsx` (GasTurb-style β maps: 20
β lines × N speed lines, sheets for `mass_flow`, `pressure_ratio`,
`efficiency`, plus `surge_line` for compressors; efficiency is isentropic).

Relative span along β at fixed corrected speed, and whether it is monotonic:

| | ICMF span | mono | ECMF span | mono |
| --- | --- | --- | --- | --- |
| Compressor Nc = 0.45 | 119.8% | ✓ | 160.1% | ✓ |
| Compressor Nc = 0.72 | 48.9% | ✗ | 93.6% | ✓ |
| Compressor Nc = 1.00 | **7.9%** | ✗ | **72.1%** | ✓ |
| Compressor Nc = 1.06 | **3.1%** | ✗ | **72.7%** | ✓ |
| Turbine Nc = 0.80 | 9.6% | ✗ | 143.5% | ✓ |
| Turbine Nc = 1.10 | **7.6%** | ✗ | **163.1%** | ✓ |

with `Wc2/Wc1 = √τ/PR`.

**Inverting ICMF → β is ill-posed over most of the operating range**, not merely
near choke: non-monotonic above 72% speed on the compressor and at every speed
on the turbine, with the whole β range compressed into 3–10% of mass flow. ECMF
is monotonic everywhere and 20× better conditioned. This reverses D9.

The cost is that ECMF needs the *downstream* stagnation state, which is the
feedback path that produced defect §4.2 #2. The difference is that in a Q1D
solver the downstream state is **measured from the field**, not predicted by
the map — and a compressor in a duct genuinely does have its operating point
set by both ends. The care required is that the feedback be closed properly
(sub-iterated within the RK stage or relaxed), not left explicit across stages.

### 3.6 The map stores corrected work — not η, not τ, not loss

Two candidates were considered for what the map should carry alongside `Wc`
and `PR`, and both were rejected by measurement.

**η is singular.** On `TranssonicCompressor.xlsx`, 4 of 180 points (all at
Nc = 0.359, low β — the stalled/windmilling corner) have `PR < 1` with the
compressor still doing work. Isentropic efficiency is undefined there, not
merely negative: along that speed line η runs −2.2016 → −0.1430 → +0.3741,
changing sign through a pole.

**τ = T₀₂/T₀₁ is not ambient-invariant.** It is invariant for a *perfect* gas
by construction, which is what makes the trap easy to fall into. For a real gas
the isentropic temperature ratio at fixed PR depends on where you sit on the cp
curve. At η = 0.85, PR = 2, referenced to 288 K:

| T₀₁ | TR perfect gas | TR real gas | drift |
| --- | --- | --- | --- |
| 230 K | 1.257663 | 1.259216 | +0.151% |
| 313 K | 1.257663 | 1.256420 | −0.072% |
| 800 K | 1.257663 | 1.229068 | −2.247% |
| 1200 K | 1.257663 | 1.214215 | −3.428% |

Storing τ would bake in the T₀₁ used for the conversion — 0.22% across the
compressor range, 3.4% into turbine territory. That is the same class of error
as the prototype's dimensional `Fx` table (P1), reintroduced through a
different door.

**Two candidates survive: loss `(h₀₂ − h₀₂ₛ)/N²` and corrected work
`Δh₀/T₀₁`.** Holding the map's PR and η fixed and varying T₀₁, both drift
identically and both are 12–24× better than τ:

| T₀₁ | τ−1 | Δh₀/T₀₁ | loss/T₀₁ |
| --- | --- | --- | --- |
| 230 K | +0.737% | **+0.063%** | **+0.063%** |
| 313 K | −0.350% | −0.030% | −0.030% |
| 900 K | −12.902% | −1.158% | −1.158% |
| 1200 K | −16.751% | −1.530% | −1.530% |

They drift identically because at fixed η they are proportional:
`loss = Δh₀(1−η)`. This test therefore cannot rank them; they diverge only
where η varies, which is across the map rather than across ambient. Both are
finite and monotone through the singular corner (Δh₀ 2932.8 → 8001.3 J/kg
while loss runs the other way, 9389.6 → 2796.5).

**Corrected torque drifts identically too**, for the same reason:
`Qc = Q/δ ∝ Wc·Δh₀/(Nc·θ)`. All three candidates are the same quantity up to
constants at a fixed map point, so ambient invariance is structurally incapable
of ranking them. The ranking has to come from behaviour **across** the map,
where η and Wc vary and the three separate.

Measured over the full grid of both nominated maps at 288 K:

| | range ratio | sign change | non-monotonic in β | worst curvature |
| --- | --- | --- | --- | --- |
| Subsonic η | 1.41 | no | 12/12 | 1.46 |
| Subsonic **corrected work** | 15.07 | no | **0/12** | **0.31** |
| Subsonic loss | 47.70 | no | 12/12 | 1.42 |
| Subsonic corrected torque | 36.49 | no | 3/12 | 0.50 |
| Transonic η | n/a | **YES** | 8/9 | 1.98 |
| Transonic **corrected work** | 45.48 | no | **0/9** | **0.19** |
| Transonic loss | 20.34 | no | 9/9 | 0.44 |
| Transonic corrected torque | 35.59 | no | 2/9 | 0.50 |

(curvature = max |second difference| / mean |first difference| along β, a proxy
for interpolation difficulty.)

**Corrected work is monotonic in β on every speed line of both maps — uniquely
so — with curvature 2–7× lower than any alternative.** Loss is non-monotonic on
*every* line of both maps, making it the worst interpolant of the three despite
being perfectly well-behaved pointwise. Corrected torque is second but
non-monotonic on a few lines.

**Directness confirms it: the map stores corrected work.** The runtime path
becomes

```
h₀₂ = h₀₁ + Δh₀        ->  SWx = W·Δh₀   and   T₀₂ = T(h₀₂)
p₀₂ = PR · p₀₁
```

with **no entropy inversion at all**. Storing loss would require solving
`s°(T₀₂ₛ) = s°(T₀₁) + R·ln(PR)` on every evaluation purely to add the loss back
on. η and the isentropic reference remain available as derived outputs for
reporting, off the hot path.

*Corrected torque* was also considered: `Q_c ∝ Wc·(Δh₀/T₀₁)/Nc` is a composite
of three quantities the map already stores separately, so it adds coupling
without adding information. It is trivially recoverable from `Δh₀` and `W` when
shaft dynamics need it.

**Residual drift is irreducible at two parameters.** The 1.53% at 1200 K is the
§3.3 effect: no two-parameter grouping is exactly invariant for a real gas. It
is measured *relative to the map's own reference temperature*, so what matters
is the range of use relative to that reference, not the absolute level — a
turbine map generated near its operating temperature carries far less than
1.5%. If it ever exceeds map accuracy, the third dimension from §3.3 is the
fix.

### 3.7 How much corrected work diverges from efficiency — all 21 maps

The decision in §3.6 rested on invariance, monotonicity and directness. This
measures the remaining question: **if the map stores corrected work instead of
efficiency, how much does the reconstructed `Δh₀` differ?** Both are calibrated
at a reference temperature and then used to reconstruct `Δh₀` at another, with
real-gas thermodynamics (Cantera, dry air). Worst case over every point of
every supplied map.

**Calibrated at standard day, used everywhere** — the naive choice:

| | 200 K | 600 K | 1000 K | 1600 K |
| --- | --- | --- | --- | --- |
| Compressors (13 maps) | 0.88% | 4.23% | 8.08% | **10.69%** |
| Turbines (8 maps) | 0.18% | 0.90% | 2.67% | **4.37%** |

**Calibrated near each component's own operating range** — the fix:

| Compressors, calibrated 288 K | 200 K | 300 K | 400 K | 500 K | 600 K | 700 K |
| --- | --- | --- | --- | --- | --- | --- |
| worst of 13 maps | 0.88% | 0.14% | 1.43% | 2.85% | 4.23% | 5.46% |

| Turbines, calibrated 1300 K | 900 K | 1100 K | 1300 K | 1500 K | 1800 K |
| --- | --- | --- | --- | --- | --- | --- |
| worst of 8 maps | 1.53% | 0.65% | **0.00%** | 0.49% | 1.05% |

Conclusions:

- **Turbines are comfortable**: ≤1.53% across 900–1800 K.
- **Compressors are fine to ~400 K and degrade after** — 5.46% at 700 K, a rear
  HPC stage, where calibrating at standard day is a real extrapolation.
- **The drift is entirely relative to the calibration temperature** (exactly
  0.00% at the calibration point). So the calibration temperature must be a
  **property of each map**, not a hard-coded 288.15.

**What this number is not.** It is the divergence between two defensible
modelling choices, not an error against truth. Under exact similarity η and
`Δh₀/θ` are both preserved and would agree exactly; they diverge only because
γ(T) breaks similarity. Without rig data at several inlet temperatures neither
can be declared correct. Where a map is calibrated near its operating point
they agree to ≤1.5%, so the choice barely matters; where they diverge by 5–10%
the extrapolation is far enough that *both* are questionable.

Corrected work is still preferred, on the grounds this test does not measure:
it is monotonic in β on all 21 speed lines of the two nominated maps where η
manages 1 of 21, and it is finite where η has a pole (§3.6).

### 3.8 The interpolation error budget — where the error actually is

Every number below is scale-free: worst deviation in reconstructed `Δh₀`,
divided by that map's own `max|Δh₀|`. 22 supplied maps, 15 compressor and 7
turbine, real-gas (Cantera, dry air), compressors calibrated at 288.15 K and
exercised over 230–330 K, turbines at 1300 K over 900–1800 K.

**Correction to an earlier finding.** A previous sweep reported cross-speed
interpolation at 10–16% and concluded it dominated every other term. That sweep
held out the *end* speed lines as well, which asks the scheme to predict outside
the map's speed range — it measured **extrapolation, not interpolation**.
Restricted to interior lines the number is 2.8%/5.3%. The actionable half of the
finding survives: speed must be clamped at the map's ends (D13).

| error source | compressors | turbines |
| --- | --- | --- |
| ambient invariance of `L/θ` (§3.6) | 0.13% | 0.29% |
| interpolation along ECMF, half resolution | ~1.2% | ~0.55% |
| across speed lines, linear in Nc at fixed β | 2.84% | 5.32% |
| across speed lines, **PCHIP** in Nc at fixed β (D13) | **1.79%** | **2.71%** |
| across speed lines, inverting ECMF first then blending | 2.81% | 6.60% |
| *extrapolation* past the end speed lines, linear | 4.89% | 10.73% |
| *extrapolation* past the end speed lines, cubic | 7.74% | 8.39% |

Leave-one-interior-speed-line-out puts the query two gaps from its neighbours,
so these overstate a real query. Holding out at four gaps as well gives an
observed convergence order of only 1.2–1.6 — not the asymptotic 2 and 3, so the
tabulated maps carry near-kinks and the cubic gain is ~1.6–2×, not an order of
magnitude. Extrapolated back to native spacing the cross-speed term is
0.13–0.18% mean.

**Normalising the loss by speed.** `L/(θ·N²)` with *physical* `N` and
`L/(θ·N_c²)` with *corrected* `N_c` are not two candidates but one, plus an
error. Since `N = N_c√θ`, `θ·N² = θ²·N_c²`, so `L/(θ·N_c²)` **is** `L/N²`, the
physical loss coefficient — already ambient-invariant with no θ in it, because
`L ∝ θ` and `U² ∝ N² ∝ θ`. Writing `L/(θ·N²)` divides by θ a second time and
makes the reconstruction scale as `θ²`:

| stored | compressors 230–330 K | turbines 900–1800 K |
| --- | --- | --- |
| `L/θ` | 0.13% | 0.29% |
| `L/N²` ( ≡ `L/(θ·N_c²)` ) | 0.13% | 0.29% |
| `L/(θ·N²)` | **8.12%** | **33.06%** |

Across speed lines the `N_c` normalisation is a wash — `L/N_c^k` for
`k ∈ {−2,−1,0,1,2}` gives 2.02/1.84/**1.79**/1.88/2.43% on compressors and
3.18/2.90/2.71/**2.61**/2.73% on turbines, all within ±0.2% of `k = 0`. The `N²`
denominator is worth having for *definedness* — unlike η it never has a zero
denominator — but it buys nothing for interpolation. `k = 0` stands.

### 3.9 Driving the solver from a real map — what actually blocks it

Phase 3 reached its gate with a *constant* map at `PR = 1.2`, `Fx = 1731 N`. A
real map at design speed asks for `PR = 2.14` and `Fx = 34,964 N` through the
same kind of duct, and that is a different problem. Three separate obstacles
were found, and it matters that they are separate — each of the first two was
briefly mistaken for the others.

**(a) A sharp two-state seed is not a discrete steady state.** Cell `i` cannot
present station 1 to its left face and station 2 to its right face at once, so
a one-cell jump leaves the disk cell out of balance by the entire source.
Integrating the steady flux balance `F_{k+1} = F_k + q/n_smear` across the smear
region and inverting each face flux (`analytic.state_from_flux`,
`design.steady_profile`) gives the profile the solver is looking for. Measured
at `n_smear = 21`: residual 1e-5 to 1e-4 of the source scale through the smear
interior, against ~1 for the sharp seed. What remains is 1–3% at the two kinks
where the ramp meets the uniform regions — the limiter reacting to a slope
discontinuity.

**(b) The exit-corrected-flow closure is circular.** ECMF is the right
coordinate for a cycle code, where the exit state is known independently.
Inside the solver the exit state is produced by this very source, so keying on
it closes an algebraic loop through the source's own output, with gain
`−dlnPR/dlnECMF`:

| Nc | 0.53 | 0.73 | 0.93 | 1.00 | 1.10 | 1.20 |
| --- | --- | --- | --- | --- | --- | --- |
| SubsonicCompressor | 0.25 | 0.50 | 0.81 | 0.90 | 0.98 | **1.02** |
| TranssonicCompressor | 0.76 | 0.91 | 1.07 | **1.09** | — | — |

Above one the loop diverges for *every* under-relaxation factor: under-relaxing
a positive-gain loop gives `|1 + r(g − 1)| > 1` for all `r > 0`. Observed as an
ECMF oscillation 16 → 32 → 24 → 35 → 16 that is unchanged at CFL 0.2, 0.1 and
0.05, which is what rules out a timestep explanation.

**(c) The inlet closure is non-circular but ill-conditioned.** `maps.evaluate_at_Wc`
measures only upstream, where the field is clean to ~1e-8, so there is no
algebraic loop at all. The price is that the source's sensitivity to the
measured flow, `dlnFx/dlnW`, is set by the slope of the speed line:

| Nc | 0.53 | 0.73 | 0.93 | 1.00 | 1.10 | 1.20 |
| --- | --- | --- | --- | --- | --- | --- |
| `dlnPR/dlnWc` | −0.09 | −0.28 | −0.87 | −1.69 | −4.81 | −18.1 |
| `dlnFx/dlnW`, real map | −0.58 | −0.95 | −1.88 | **−3.29** | −8.38 | **−29.4** |
| `dlnFx/dlnW`, PR frozen | −0.18 | −0.15 | −0.11 | −0.10 | −0.09 | −0.08 |

The feedback is *restoring* (negative) — more flow, less thrust — but with the
sampling station 12 cells upstream it acts after a delay, and delayed negative
feedback of magnitude ≫ 1 oscillates. This is the dynamic form of the same
conditioning defect D9 recorded for ICMF as a lookup coordinate: `Wc` spans
133% of its range at 33% speed and 8% at 120%, so at the top of the map a
0.1% flow error is a 1.8% pressure-ratio error, and the loop amplifies it.

**Neither coordinate is usable at the top of these maps**, for opposite
reasons, and the two failure regions coincide. Below ~90% speed on
`SubsonicCompressor` both gains are below one and the inlet closure is the
better of the two.

**A control that is not a control.** Freezing `PR` and `Δh₀/θ` at their design
values does *not* isolate the source magnitude, because it removes the map's own
stabilising slope — the last row above shows the gain collapsing from −3.29 to
−0.10. Its failures (blow-up at `n_smear` 1 and 5, a limit cycle at 21 and 41)
are therefore evidence about a constant-`PR` compressor, which `PLAN.md` §Phase 3
already records as having zero aerodynamic stiffness by construction. It is
reported here because it was run and because it does bound the strength effect:
at a quarter strength (`PR = 1.285`) the same configuration converges to
`W` within 1.9e-7.

**What the solver does, speed by speed.** 401 cells, 81 smear cells, 12-cell
standoff, inlet closure, seeded from `steady_profile`, converged on the disk's
own mass flow going quiet, then held 10× longer. Error is against the map point
the duct was designed for.

| `SubsonicCompressor` Nc | 0.330 | 0.430 | 0.530 | 0.625 | 0.672 | 0.728 | 0.833 | 0.930 | 1.000 | 1.100 | 1.200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W error | 1.6e-11 | 3.8e-11 | 4.6e-11 | 1.0e-09 | 4.0e-08 | 2.1e-02 | 2.8e-01 | 6.1e-01 | 1.7e-01 | blew up | blew up |
| drift over the hold | 1.2e-12 | 1.7e-12 | 3.0e-11 | 7.4e-10 | 3.4e-08 | 4.6e-02 | 6.1e-01 | 4.3e-01 | 6.7e-01 | — | — |
| `|dlnFx/dlnW|` | 0.34 | 0.44 | 0.58 | 0.64 | **0.78** | **0.95** | 1.24 | 1.88 | 3.29 | 8.38 | 29.4 |

`TranssonicCompressor` behaves the same way: 1.3e-11 at Nc = 0.359, 5.5e-8 at
0.528, then 1.7e-01 at 0.661; from 0.880 up the inlet closure refuses to run at
all because `Wc` is not invertible there.

**The boundary is NOT the logarithmic loop gain.** An earlier version of this
section claimed it was, generalising from `SubsonicCompressor` alone.
`TranssonicCompressor` refutes it: it *holds* at `dlnFx/dlnW` = −4.41 (Nc 0.359)
and −3.38 (Nc 0.528) while `SubsonicCompressor` *fails* at −0.95 (Nc 0.728).
The logarithmic gain does not separate the data.

**The boundary is the acoustic impedance ratio.** A mass-flow perturbation `dW`
in a duct carries a force perturbation `c·dW`, so the duct's characteristic
impedance, as a force per unit mass flow, is the speed of sound. The disk
responds with `dFx/dW`, also a velocity, and negative. The group is

    Z = |dFx/dW| / c

| Nc | 0.330 | 0.530 | 0.672 | 0.728 | 0.833 | 1.000 | 1.100 | 1.200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Subsonic Z | 0.057 | 0.266 | **0.597** | **0.879** | 1.61 | 6.56 | 20.5 | 81.6 |
| verdict | held | held | held | no | no | no | blew up | blew up |

| Nc | 0.359 | 0.528 | 0.661 | 0.791 |
| --- | --- | --- | --- | --- |
| Transsonic Z | 0.601 | **1.31** | 2.98 | 13.3 |
| verdict | held | held | no | no |

Every held case has `Z ≤ 1.31`, every failure `Z ≥ 0.879`: one overlapping
pair, otherwise a clean split at `Z ≈ 1`. When `|dFx/dW| > c` the disk returns
more to an acoustic wave than the wave brought it — a **negative acoustic
resistance** — so reflected waves grow. That is scale-invariant and independent
of mesh, smear, standoff and timestep, which is exactly the pattern every
experiment showed.

**Remedies tried.** A first-order lag of time constant `τ` on the applied
source — the textbook fix for a delayed loop, and one with unit DC gain, so the
converged point is unchanged by construction:

| Nc = 0.7285, gain 0.95 | τ = 0 | 3e-4 | 1e-3 | 3e-3 | **1e-2** |
| --- | --- | --- | --- | --- | --- |
| W error | 1.3e-01 | 5.9e-02 | 2.0e-02 | 6.1e-03 | **5.7e-07** |

At Nc = 0.833 (gain 1.24) the same sweep does not converge at any `τ` tried
(best 1.25e-01). Shortening the standoff, which is the other way to cut the
delay, does not help at either speed — 5.4e-02 at offset 2 and 2.7e-02 at
offset 4 against 1.3e-01 at offset 12 — so the standoff is not the dominant
delay; the acoustic transit of the smear region and duct is.

**Mesh refinement separates two different failures.** Smear length held at 20%
of the duct, everything else fixed, `W` error against the map point:

| Nc | gain | 201 cells | 401 | 801 | 1601 |
| --- | --- | --- | --- | --- | --- |
| 0.672 | 0.78 | 1.7e-02 | **9.6e-09** | **3.4e-07** | **8.3e-08** |
| 0.728 | 0.95 | 1.1e-01 | 6.7e-03 | 1.1e-02 | **2.4e-06** |
| 0.833 | 1.24 | 3.0e-01 | 8.3e-02 | 4.4e-01 | 6.6e-02 |

The first two rows converge under refinement, so those failures are
**discretisation**, and either refining or damping removes them. The third does
not: 1.24 stays at O(1e-1) across an eightfold refinement, and the lag does not
touch it either. Two independent remedies failing the same way, mesh-
independently, is evidence that above `|dlnFx/dlnW| ≈ 1` the difficulty is in
the **problem**, not the discretisation of it.

**Two further tests, one refuted, one ill-posed — both recorded because they
were run.**

*The map-end clamp is not the mechanism.* `evaluate_at_Wc` returns the end point
outside the tabulated range, which is a hard nonlinearity that could in
principle sustain a limit cycle. It does not. At Z = 0.879 the lookup leaves the
map in 1.1% of evaluations and only after 72% of the run — the oscillation is
fully grown by then — and replacing the clamp with linear extrapolation off the
end slope moves the answer from 2.08e-02 to 1.75e-02. At Z = 6.56 extrapolation
is strictly *worse*: it drives the state past choke (Φ = 0.696 against Φmax
0.685) and the run dies. Clamping onset collapses with Z (72% → 11% → 2% of the
run), which is the signature of a growing instability that starts inside the
map, not a clamp-driven cycle.

*Shrinking the artificial delays helps but does not fix.* The 12-cell standoff
was measured in Phase 3 for a **one-cell** disk, and the 81-cell smear is 20% of
the duct where a blade row is ~1%; both were chosen for accuracy with no regard
for the loop. Cutting them to 2.6%/0.25% of the duct is worth 35× at Z = 1.61
(2.77e-01 → 7.92e-03) and 73× at Z = 0.879 (2.08e-02 → 2.85e-04). Neither
reaches the 1e-6 hold criterion, and the residual error is **non-monotone in
mesh** at fixed physical smear (7.9e-03 → 1.1e-02 → 4.3e-02 over 801 → 1601
cells), which is a limit cycle of reduced amplitude, not a converging solution.

*The duct-length sweep is a null test.* It returned bit-identical results at
L = 0.125, 0.25 and 0.5 m — same mass flow to the last digit, same clamp
fraction, same step count. That is not insensitivity: the problem is **invariant
under length scaling**, because the source is a total (N, W) spread over cells
whose volume scales with L, so source-per-volume and flux divergence both go as
1/L and changing L only rescales time. There is no intrinsic time scale to
compare against, which is why the *lag* test was meaningful — `τ` is a fixed
physical time — and why this one could not be. The test should not have been
run; it is recorded so the same mistake is not repeated.

**Superseded hypothesis: the exit boundary is a degenerate throttle.** A fixed static back pressure has a horizontal characteristic in the
(W, p) plane — it supplies no restoring pressure when the flow moves, which is
the least stabilising exit condition there is. A real throttle is an *area*:
pass more flow through it and the pressure it demands rises. On a steep speed
line the compressor's own characteristic is nearly vertical, so the
intersection of a near-vertical compressor line with a horizontal throttle line
is exactly the ill-conditioned case. If that is right, the fix is a nozzle
outlet rather than a pressure outlet, and it is a *modelling* fix, not a
numerical one.

A first test gave the outlet a linearised throttle stiffness
`p_back = p_back0 + K(W − W₀)`, with `k = K·W₀/p_back0` the dimensionless
stiffness. At Nc = 0.833 it improves the error but does not restore the hold —
2.8e-01 at `k = 0`, then 5.3e-02, 5.5e-02, 9.3e-02 at `k` = 0.5, 1, 2 — and
blows up within 25 steps at `k` = 5 and 10. **This is weak evidence either
way.** The implementation writes the boundary pressure from inside the source
callback, so it is updated within the Runge-Kutta stages rather than as a
proper characteristic outlet; the blow-up at large `k` is that coupling going
unstable, not the physics rejecting a stiff throttle. A real nozzle boundary
condition is needed before the hypothesis can be called confirmed or refuted.

**What this means for the model.** `Z ≈ 1` is not a numerical group. A real
blade row does not present its steady-state `dFx/dW` to a kilohertz acoustic
wave — its unsteady response rolls off with reduced frequency. Our disk presents
the full steady slope at every frequency, which is physically impossible, and is
what makes it a wave amplifier above `Z = 1`. **A quasi-steady map is not a
valid unsteady boundary condition**, and that is a modelling gap, not a bug to
be found. It also reframes the lag: a low-pass on the disk's response is the
right *family* of fix, because it is what the real physics does. `τ` was picked
by trial rather than derived, which is why it worked at Z = 0.879 and not at
Z = 1.61.

**Status.** Half resolved, and the half that works is measured rather than
asserted: a real map drives the solver to its own operating point to 1e-11 in
mass flow with 1e-12 drift, at five speeds on `SubsonicCompressor` and two on
`TranssonicCompressor` (`docs/phase5_ecmf_map.png`). Above the gain-one boundary
it does not, and the cause is **not settled**. See §8.

### 3.10 The map was never the problem

Everything in §3.9 — the ECMF circularity, the loop gain, `Z`, the β lag, the
map-end clamp — is about the *map*. A control that removes the map entirely
settles it.

**Constant map, closed-form reference, no lookup, no β, no coordinate choice.**
`ConstantCompressorMap` with the Phase 3 actuator disk, 401 cells, 21-cell
smear, checked against `zero_d_compressor`:

| PR | 1.20 | 1.40 | 1.60 | 1.80 | 2.00 | 2.20 | 2.50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Fhat = Fx/(p₁A)` | 0.213 | 0.430 | 0.651 | 0.873 | 1.096 | 1.321 | 1.659 |
| W error | **1.9e-10** | 9.5e-03 | 3.2e-02 | 3.8e-03 | 1.9e-02 | blew up | blew up |
| disturbance envelope | ×0.0014 | ×29.7 | — | — | — | — | — |

**The limit is between PR 1.2 and 1.4.** Phase 3 passed its 1e-8 gate at
PR = 1.2 and four phases were built on top of that without the disk ever being
tested above it. Real maps need 1.5–2.5, and radial machines reach 14.

Ruled out on this clean case: CFL (1.5, 0.8, 0.4, 0.2 all fail, so it is not
explicit-source stiffness and point-implicit is not the fix).

**The mechanism, and an error running through all of §3.9.** Every `Z` in this
document was computed by perturbing `W` with `T₀₁` and `p₀₁` held fixed. An
acoustic wave does not do that — it changes density, velocity and pressure
together. The disk sets `p₀₂ = PR·p₀₁` from a pressure it *measures*, so a wave
arriving at the sampling station raises `p₀₁`, raises `Fx`, and launches a
stronger wave. For a left-running wave (`δp = −ρcδu`, `δρ = δp/c²`) that path
contributes a term of size `Fx/(p₀₁A)` — the source strength itself — which a
W-only perturbation cannot see at all.

Recomputed along the characteristic:

| PR | 1.20 | 1.40 | 1.80 | 2.20 | 2.50 |
| --- | --- | --- | --- | --- | --- |
| `Z` (W only) | 0.064 | 0.107 | 0.166 | 0.204 | 0.226 |
| **`Z` acoustic** | **0.301** | **0.577** | 1.091 | 1.580 | 1.937 |
| verdict | held | not | not | blew up | blew up |

`Z` (W only) spans 0.06–0.23 across cases running from exact to divergent, and
against its own 0.82 threshold would call *all* of them stable. `Z` acoustic
orders them correctly and scores 97.5% on the 160-case map sweep against 96.4%.

**Not yet a calibrated criterion.** The constant-map set used a 21-cell smear
and the map sweep 81, and smear is worth 35–73× (§3.9), so the thresholds
(0.3–0.6 against 1.66) are not comparable. The physics is consistent; the
calibration is not, and no single number should be quoted until they are run at
matched smear.

**Consequence for the plan.** The failure is not in the map layer, so no map-side
remedy can fix it: not the closure, not the coordinate, not the lag, not
densification. Phase 5's stated blocker in §8 is wrong and is corrected there.
The next work is on the actuator-disk source itself — specifically that a single
total force computed from one remote station is applied uniformly across the
row, so the whole row responds in unison to one measurement. A distributed
source whose density responds to the *local* state in each cell is the
reformulation to test.

### 3.11 The prototype's systematic error was load-bearing

The prototype can be given valid high-PR operating points — the back pressure
and area are settings, not physics — and when it is, **it holds them**. Only its
initial condition needed patching (it expands isentropically from inlet total to
back pressure and goes complex for `pb > p01`); the numerics, limiter, Roe flux,
RK stages and source injection are untouched. `A = 0.1`, `η = 0.9`, `M₁ = 0.45`:

| PR | 1.20 | 1.40 | 1.60 | 2.00 |
| --- | --- | --- | --- | --- |
| W at the lookup station, error | 1.0e-03 | 4.8e-04 | 2.7e-04 | 3.1e-05 |
| far-field spread, cell 20 vs 80 | 2.7e-15 | 2.1e-12 | 6.7e-11 | 6.5e-09 |

The rewrite fails from PR 1.4, **including at legacy's own settings** (100
cells, single-cell disk, CFL 2.5). So this is a genuine regression, not a
difference of test configuration — moving cell count, smear and CFL one at a
time from legacy's values to ours changes nothing qualitative.

**What was removed.** Legacy tabulates `Fx` and `SWx` once, at its design inlet
state, and interpolates that table at runtime. That is exactly the systematic
error this project exists to fix: a table in newtons is only valid at the inlet
condition it was built at. But the frozen table also makes the source **blind to
the locally measured stagnation pressure**, which sets the acoustic feedback
gain to zero by construction. Phase 3 replaced it with runtime evaluation from
the local state — correct for varying inlet conditions, and the origin of the
loop in §3.10:

    a wave raises p01 at the sampling station
      -> Fx ~ (PR p01 - p1) A rises
      -> a stronger wave is launched,     loop gain ~ Fx/(p01 A)

Isolated by changing **only** where the source reads its inlet state, everything
else identical, on the clean constant-PR case at legacy geometry:

| PR | 1.20 | 1.40 | 1.60 | 2.00 | 2.50 |
| --- | --- | --- | --- | --- | --- |
| inlet state measured **locally** (Phase 3) | 2.4e-10 | 4.7e-02 | 1.2e-01 | blew up | blew up |
| inlet state from the **boundary condition** | 1.1e-10 | **4.3e-11** | **2.1e-11** | 7.4e-02 | 2.6e-02 |

One line of difference turns 4.7e-02 into 4.3e-11.

**Why the BC variant is not the fix.** It hard-codes a single upstream station,
breaks for multi-component flowpaths, and defeats the purpose of runtime
evaluation — responding to real inlet conditions is the entire point (P1). It
also stops working by PR 2.0 on its own, so it is not even a complete remedy.

**The specification this yields.** The source must respond to inlet conditions on
the **flow-through timescale** but not at **acoustic frequencies**. That is a lag
on the measured inlet stagnation state — not on β (§3.9, tried, the map slope
was never the issue) and not on the applied source (§3.9, tried, it delays the
instantaneous momentum-flux term too). The constant-PR case with its closed-form
reference and a clean pass/fail at PR ≥ 1.4 is the test bed for sizing it,
without any map machinery in the way.

**Reading of the whole investigation.** Phase 3's gate at PR = 1.2 sat just
inside the stable region, so the instability the fix introduced never showed,
and four phases were built on top of it. Everything in §3.9 — closures,
coordinates, loop gains, `Z`, densification, β lags — was chasing consequences
of that in the map layer, where the cause never was.

### 3.12 The fix: lag the inlet state the source reads

§3.11 gives the specification — track inlet conditions on a slow timescale, not
at acoustic frequencies. `compressor.InletFilter` is a first-order lag on the
measured `(T₀₁, p₀₁)`, applied by `ActuatorDisk` and `InletFlowCompressor` via
`inlet_lag` (seconds, default 0 = off). `W` stays instantaneous.

Constant PR, legacy geometry, closed-form reference. W error:

| PR | no lag | τ=1e-4 | τ=1e-3 | τ=3e-3 | τ=1e-2 | τ=3e-2 | τ=1e-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.4 | −6.1e-03 | 2.4e-02 | **2.04e-10** | **2.05e-10** | **2.05e-10** | **2.05e-10** | **2.05e-10** |
| 1.6 | −1.0e-01 | −8.4e-03 | −7.5e-08 | **2.00e-10** | **2.16e-10** | **2.04e-10** | **2.02e-10** |
| 2.0 | blew up | blew up | −1.2e-01 | +5.3e-02 | — | −2.4e-02 | −4.1e-02 |
| 2.5 | blew up | blew up | +3.6e-02 | +1.3e-01 | −1.6e-02 | −5.3e-03 | +1.7e-02 |

**What it fixes.** PR 1.4 and 1.6 now hold to 2e-10, the same accuracy the disk
reaches at 1.2. The usable range moves from PR < 1.3 to **PR ≤ 1.6**.

**What it does not.** PR 2.0 and 2.5 no longer *diverge* — a real improvement,
from blow-up to a bounded 1–5% oscillation — but they do not hold. A second
mechanism acts above ~1.7 and is not diagnosed. Radial machines reach PR 14, so
this is a step, not a solution.

**Why it is a stability device and not a fudge.** The lag has unit DC gain, so
the converged answer cannot depend on `τ`. Measured, not assumed: at PR 1.4 the
W error is 2.044e-10, 2.050e-10, 2.050e-10, 2.049e-10, 2.047e-10 across a
hundredfold range of `τ`, and `tests/test_compressor.py` pins that invariance
alongside the PR 1.2 point being undisturbed.

**Choosing τ.** The blade row's own through-flow time (~7e-5 s here) is far too
short — τ=1e-4 fails at every PR. One rotor revolution works: `rotor_period(rpm)
= 60/rpm` gives 6e-3 s at 10,000 rpm, comfortably inside the range that holds.
This is the author's `1/N` suggestion and needs no data the maps do not carry.
τ must not be tuned for stability: it is a transient-response parameter and
transient response is what the project exists to model.

**A start-up interaction, measured.** The Phase 3 default duct sits at
`M₁ = 0.665`, 11% from inlet choke. The lag slows the source's response enough
that starting from a *uniform* field overshoots into choke. Either margin
(`M₁ = 0.45`) or a steady seed avoids it; the regression tests use both.

**The gate that was missing.** `TestHighPressureRatio` now checks 1.4 and 1.6
with and without the lag, and asserts the unlagged case *fails* — so if this
regression ever silently repairs itself, the test says so rather than passing
quietly. Phase 3's single check at PR 1.2 is what let four phases build on a
defect.

### 3.13 Closing the second path: lag the mass flow too

§3.12 lagged `(T₀₁, p₀₁)` and reached PR 1.6. What remained instantaneous was
`W`: the sampling station's mass flow still responds to a passing wave within a
step, and `Fx` depends on it, so one feedback path stayed open. `InletFilter`
now lags all three on the same time constant.

Smallest `τ` that holds, constant PR, legacy geometry (duct `L/c` = 2.94e-3 s):

| PR | 1.40 | 1.60 | 2.00 | 2.50 |
| --- | --- | --- | --- | --- |
| τ needed | 3e-3 | 3e-3 | **1e-2** | none up to 3 s |
| τ/(L/c) | 1.0 | 1.0 | 3.4 | — |
| W error | 2.05e-10 | 2.01e-10 | 2.16e-10 | — |

At PR 2.0 the stagnation-only filter gives −2.38e-02 and the full filter
2.16e-10. **The usable range moves from PR ≤ 1.6 to PR ≤ 2.0.**

**The lag family has a ceiling, it does not merely need more.** At PR 2.5 no `τ`
on a ladder up to 3 s holds the point — three orders above the acoustic transit,
and far beyond anything defensible as a blade-row response time. So the
remaining obstacle is not the feedback the lag addresses.

**Two hypotheses tested and rejected.**

*Source/reconstruction imbalance.* The geometric source is well balanced and the
disk source is not, so MUSCL might be reading a source-imposed profile as a
solution gradient. If so, dropping to first order **near the disk** would fix
it. It does not: at PR 2.0 a local first-order region gives −9.8e-03 against
+4.1e-04 for plain second order, while *global* first order holds to 2.5e-14.
First order is damping the whole duct, not repairing anything at the source, so
this hypothesis is wrong and first order is a mask rather than a fix.

*Wider smearing.* At PR 2.0, second order gives 4.1e-04 at `n_smear` = 1 and
degrades to 1.2e-02 and 4.0e-02 at 11 and 21 cells. Spreading the source makes
it **worse**, which also rules out "the gradient is simply too steep for one
cell".

**Where each scheme stops** (inlet lag on throughout):

| | PR 2.0 | 2.5 | 3.0 | 4.0 |
| --- | --- | --- | --- | --- |
| second order | **2.16e-10** | fails | fails | fails |
| first order | 3.7e-14 | 1.2e-14 | 2.9e-12 | blows up |

First order buys another 1.5 in pressure ratio and then stops too, so above
PR 3 the difficulty is not the reconstruction either. At PR 4 the disk is asked
for `Fx/(p₁A)` = 3.6 — a 4.6× static pressure rise across the modelled element.

### 3.14 Staging helps; the binding limit is the *total* pressure ratio

No machine raises pressure 4:1 in one element. A PR-14 compressor is eight to
ten stages over a metre, and PR 14 split eight ways is 1.40 a stage — which the
disk holds to 2e-10. So the natural model for high pressure ratio is a train of
disks, not one disk with a huge jump.

**Staging works, and the stages do not fight each other.** Same total pressure
ratio, split different ways, 601 cells:

| total PR | stages | stage PR | gap | W error |
| --- | --- | --- | --- | --- |
| 2.0 | 1 | 2.00 | — | 3.72e-06 *(not held)* |
| 2.0 | 2 | 1.414 | 40 | **3.70e-07** |
| 2.0 | 2 | 1.414 | 80 | **5.11e-09** |
| 2.0 | 4 | 1.189 | 40 | **3.63e-10** |

Splitting improves the answer by four orders and wider spacing helps again, so
the disks are not coupling destructively.

**But the limit is on the total, not the stage.** At PR 4.0 every split tried
blows up — 2 stages at 10413 steps, 4 stages at 24833, 4 stages with `τ`=1e-1 at
46707 — even though a *single* PR-1.414 disk is stable indefinitely. More stages
buy time and never stability.

That points away from the disk. A wave launched by the last stage runs upstream
through every stage ahead of it, each amplifying it, and the duct's overall 4×
rise sits against a fixed stagnation-inlet boundary. The compounding is in the
**train plus its boundaries**, not in any one element.

**A mesh sensitivity worth recording.** The same single-stage PR 2.0 case holds
to 2.16e-10 on 100 cells and misses at 3.72e-06 on 601. Finer meshes are *less*
stable here, consistent with first order (more dissipative) outperforming second
throughout §3.13. Any claim about a pressure-ratio limit must state its mesh.

**Status.** Best current configuration — inlet lag on `(T₀₁, p₀₁, W)`, staged
disks, generous spacing — holds a total pressure ratio of **2.0** to 3.6e-10.
PR 4 and above is unresolved, and the next thing to examine is the inlet
boundary condition under a large adverse duct pressure rise, not the disk.

### 3.15 Above PR ≈ 3 — what it is not

A long elimination pass. Recorded because each result closes a door, and
several of these were things I would otherwise try again.

**It is not the source's state dependence.** A source with constant `Fx` and
`SWx`, reading nothing at all, blows up at PR 4 (step 5778, 201 cells). No
source-term formulation can fix that.

**It is not the per-cell gradient.** Spreading that constant source until the
static pressure rises only **0.7% per cell** (`n_smear` = 201 of 401) still
blows up. At PR 2 spreading *does* fix it — 1.20e-04 → 3.64e-07 across
`n_smear` 1 → 201, a smooth convergence — so the two pressure ratios fail for
different reasons and PR 2 was never a stability problem at all, only accuracy.

**It is not the reconstruction.** Local first order around the disk is no better
than plain second order; global first order reaches PR 3.0 and then blows up
too. First order damps the whole duct rather than repairing anything.

**It is not the map's shape.** A real map's stabilising `dPR/dW < 0` does not
help: `HPC01` holds PR 1.73 and fails from 2.26, the same place a constant map
fails.

**It is not the exit boundary.** A choked-nozzle outlet, sized to choke exactly
at the design point — the termination a real engine has, and one that pins
corrected flow — fails at step 873 against 683 for a fixed static pressure.

**It is not the equal-area assumption.** Contracting the duct downstream of the
disk so the exit axial Mach returns to 0.45, instead of collapsing to 0.045 at
PR 14, changes step 683 to step 1123.

**It is not damping.** On `HPC01` at PR 4.44 the required lag is *non-monotonic*:
τ = 1e-2 dies at step 683, 3e-2 at 4543, **1e-1 survives**, 3e-1 dies at 17333,
1.0 dies at 12549. A resonance, not a gain to be turned down.

**The inlet matters only up to PR ≈ 3.** Pinning the full inlet state
(Dirichlet — illegal for subsonic inflow, diagnostic only) rescues the frozen
source at PR 4 (5.9e-11 in 2000 steps) and a real map at PR 3.1 (9.1e-11). At
PR 4.44 and above it makes no difference whatever: 706 against 683, 95 against
95, 47 against 47. Partial absorption at the inlet does not help either — σ=0.2
delays failure from 10313 to 30666 steps, σ=0.5 survives but collapses the flow
(W off by 99.7%).

**What it looks like.** `HPC01` at Nc = 0.8, PR 4.44, traced through the first
683 steps: `Wc` oscillates 22.785 → 22.765 → 22.712 → 22.621 → 22.535 with the
amplitude doubling every ~100 steps, entirely **inside** the tabulated range
[17.39, 23.60], so nothing is being clamped. `PR` follows it 4.436 → 4.449 →
4.480 → 4.532 → 4.579. A clean exponential growth of the `W ↔ PR` loop that the
inlet is not part of and that damping cannot reach.

**Status.** Superseded by §3.16, which identifies the mechanism. The closing
suggestion here — solve the disk's operating point implicitly each step — was
**wrong**, and wrong for an instructive reason: the equilibrium it would march
to is itself a repeller, and no time integrator reaches a repeller. That is
measured in §3.16 and confirmed by BDF2 and SDIRK3 in §3.17.

### 3.16 The equilibrium is unstable, and the lag is what makes it so

Every entry in §3.15 asks "which numerical ingredient breaks the run?". The
question has no answer because the premise is wrong. Linearising about the
**designed steady state** — not about wherever a diverging run ended up — gives

| Nc | PR | max Re(λ) | Im | e-fold |
| --- | --- | --- | --- | --- |
| 0.500 | 1.734 | −2.99e+01 | 0 | — |
| 0.600 | 2.257 | **+7.00e+00** | 0 | 1.43e−01 s |
| 0.700 | 3.104 | +4.41e+01 | 0 | 2.27e−02 s |
| 0.800 | 4.436 | +7.73e+01 | 0 | 1.29e−02 s |
| 1.000 | 7.489 | +1.14e+02 | 0 | 8.74e−03 s |

`Im(λ) = 0`, so it is monotone divergence, not a resonance — §3.15's reading of
the `Wc` trace as an oscillation was the growth envelope, not a frequency. The
prediction is independent and it lands: at PR 4.44 an e-fold of 1.29e−2 s is
**650 steps**, against the observed blow-up at step **683**.

It is a property of the continuous model, not the grid. `max Re(λ)` at 101, 201
and 401 cells with the smear held at ~10% of the duct:

| Nc | PR | n=101 | n=201 | n=401 |
| --- | --- | --- | --- | --- |
| 0.500 | 1.734 | −2.9964e+01 | −2.9942e+01 | −2.9931e+01 |
| 0.700 | 3.104 | +4.4185e+01 | +4.4117e+01 | +4.4084e+01 |
| 0.800 | 4.436 | +7.7550e+01 | +7.7402e+01 | +7.7332e+01 |

Four significant figures, converging. Mesh refinement is not the fix and never
was.

**What that Jacobian actually describes.** `InletFilter.update` returns its
stored state whenever `solver.t` has not moved, and a finite-difference Jacobian
never moves `t`. So `∂q/∂U = 0` in every number above: the map contributes
nothing, and what is being linearised is a duct carrying a **fixed force [N] and
a fixed heat rate [W]**. That is a known-unstable device:

> `W` falls → `Δh₀ = Ẇ/W` rises → `T₀` rises → more thermal blockage → the duct
> passes less flow → `W` falls further.

The 0D static-stability slope confirms it, by pure algebra with no solver
involved, and flips sign at exactly the speed line the eigenvalue does:

| Nc | PR | `dp_exit/dW` frozen | normalised | with the map live | normalised |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 1.734 | −5.95e+02 | −0.0326 | −2.27e+04 | −1.24 |
| 0.6 | 2.257 | **+9.63e+01** | +0.0054 | −2.28e+04 | −1.27 |
| 0.7 | 3.104 | +4.19e+02 | +0.0228 | −3.30e+04 | −1.80 |
| 0.8 | 4.436 | +5.16e+02 | +0.0264 | −6.41e+04 | −3.28 |
| 1.0 | 7.489 | +4.86e+02 | +0.0216 | −1.50e+06 | −66.74 |

**The machine on its own map is stable at every speed** — slope −1.2 to −66.7,
firmly on the falling branch. The instability is manufactured by freezing the
source, and freezing the source is what the inlet lag of §3.12–3.13 does.

This also corrects §3.13. Lagging `W` was measured on the **constant-PR** map,
where `PR(Wc)` is flat and there is no restoring slope to delete, so lagging it
cost nothing. On a real map `W → Wc → PR` *is* the restoring force, and lagging
it removes the only thing holding the equilibrium up. The conclusion was right
for the case it was measured on and does not generalise.

**A first-order lag cannot be tuned out of this.** Putting the filter into the
eigenvalue problem as three extra states,

```
dx/dt = −R(x, y)/dx          x: 3n conserved variables
dy/dt = (m(x) − y)/τ         y: the sensed (T01, p01, W)
```

makes `τ` sweepable. Both limits reproduce numbers measured independently
(`τ → 0` the unfiltered source, `τ → ∞` the table above), which is what
validates the assembly:

| Nc | PR | live | 1e−4 | 1e−3 | 1e−2 | 3e−2 | 1e−1 | frozen | best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 1.734 | 1.72e+03 | 1.45e+03 | 2.03e+02 | 2.77e+01 | **−7.27e+00** | −2.64e+01 | −2.99e+01 | stable |
| 0.6 | 2.257 | 2.72e+03 | 2.29e+03 | 3.87e+02 | 3.45e+01 | 8.98e+00 | **−1.14e+00** | +7.00e+00 | stable |
| 0.7 | 3.104 | 4.38e+03 | 3.69e+03 | 1.47e+03 | 8.64e+01 | 5.02e+01 | **+3.25e+01** | +4.41e+01 | **no** |
| 0.8 | 4.436 | 6.69e+03 | 5.64e+03 | 3.00e+03 | 3.64e+02 | 1.01e+02 | **+6.75e+01** | +7.73e+01 | **no** |
| 1.0 | 7.489 | 1.57e+04 | 1.34e+04 | 9.47e+03 | 5.32e+03 | 3.49e+03 | 1.62e+03 | **+1.14e+02** | **no** |

There *is* an interior optimum — which is what §3.15's "non-monotonic τ" was
seeing — but the window **closes at PR ≈ 2.3**. Small `τ` opens the acoustic
loop of §3.11; large `τ` lands on the frozen rising branch; and from PR 3.1 the
minimum over all `τ` is positive. **No first-order lag on the sensed triple can
stabilise this closure.** That is an impossibility result, so the closure has to
change rather than be tuned.

**Two candidate structural fixes, measured.** Both follow from the mechanism,
and their results are as informative as a success would have been.

*Deliver the work against the live mass flow, keep the lookup lagged.* The row
does fixed work per kilogram; the rate should follow the mass actually passing.
Made **everything worse** — Nc 0.6 frozen +7.00 → +25.9, Nc 1.0 +114 → +289 —
even though its 0D slope is stabilising. The difference between the 0D argument
and the PDE is the **transport delay**: `W` is read twelve cells upstream, so
the source answers a wave that passed ~1.5e−4 s ago, and negative feedback
through a delay is an oscillator. The observed unfiltered growth rate at Nc 0.5,
1/782 s, is about nine transits of that offset.

*Weight the sources by the local cell mass flux.* Same idea with **zero** delay,
and the sign reverses: the energy source improves every speed (Nc 0.7 frozen
+44.1 → +18.2; Nc 0.6 +7.00 → −13.4, i.e. stable). The momentum source must
**not** be weighted this way — `d(ρu)/dt ∝ ρu` is exponential growth by
construction, at rate `(Fx/W)/L_smear ≈ 1600/0.1 = 1.6e+04 /s`, which is what
came out (1.6e+04 at Nc 0.8). Physically consistent: a blade row's axial force
is the static pressure difference it holds, near enough independent of the
instantaneous local momentum, so a fixed force is the right model for it.

*Shorten the sampling offset.* Monotone at every speed. Minimum over `τ`, offset
20 → 1: Nc 0.6 **+0.41 → −4.41** (unstable → stable), Nc 0.7 +36.0 → +25.5,
Nc 0.8 +77.3 → +59.5. About 25%, real, not enough alone.

**This one is not free**, and Phase 3 had already measured the price: the
reconstruction stencil creates a numerical boundary layer *upstream* of the
disk, so offset 1 reads `p₀₁` from inside it and biases the converged mass flow
by **−3.3e−04**, against 7.5e−11 at offset 12 (Phase 3, "The sampling standoff,
measured"). Buying 25% of a growth rate with seven orders of steady-state
accuracy is not a trade worth making on its own; it is recorded here because it
identifies the *delay* as a real contributor, which is what falsified the
live-mass-flow fix below.

*Stacked.* The two add but do not close the gap. Minimum over `τ`, both applied:
Nc 0.7 +32.5 → **+10.4**, Nc 0.8 +67.5 → +44.2, Nc 1.0 unchanged at +73.8.
Threefold, still a repeller.

**Which frozen quantity is the branch — four devices, one slope each.** Pure
algebra at the design point, `dp_exit/dW` normalised by `p_back/W`:

| Nc | PR | fixed `Fx`, fixed `SWx` | fixed `Fx`, fixed `Δh₀` | fixed `PR`, fixed `Δh₀` | live map |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 1.734 | −0.0326 | −0.0540 | −0.1083 | −1.2444 |
| 0.6 | 2.257 | **+0.0054** | −0.0117 | −0.0653 | −1.2693 |
| 0.7 | 3.104 | +0.0228 | **+0.0111** | −0.0362 | −1.7971 |
| 0.8 | 4.436 | +0.0264 | +0.0190 | −0.0190 | −3.2821 |
| 1.0 | 7.489 | +0.0216 | +0.0179 | −0.0077 | −66.7398 |

The first two columns predict the eigenvalues **exactly**: the shipped device
turns rising at PR 2.26 where the eigenvalue turns, and flux-weighted heat pushes
that to PR 3.10, which is precisely where flux-Q stopped rescuing speeds (Nc 0.6
stable, 0.7 not). The third column never rises — a device holding a pressure
*ratio* and a *specific* work is statically stable at every speed, which is also
what a map actually specifies. A force in newtons is only what that ratio is
worth at one operating point.

**And that device fails anyway, which is the most useful result here.**
Implemented with the delay removed (`sample_offset` 1), the ratio device is
*worse* than the shipped one above PR 2.3 — minimum over `τ`: +3.46 (Nc 0.6),
+42.9, +78.0, +134 (Nc 1.0), against −1.14, +32.5, +67.5, +114. So the 0D
criterion that predicted columns one and two correctly **does not predict column
three**. Something in the PDE that the lumped model cannot see is destabilising
it, and the common factor is that the source is driven by a **point measurement
of the flow**: any closure of that form carries a local feedback of rate
`|∂(source)/∂(flux)| / L_smear`, which is 1e3–1e4 /s here regardless of sign
conventions, while the lumped model assumes the whole system moves together.

**Status.** Mechanism identified for the frozen branch; the remaining obstacle
is now named and is structural. Neither the map, the mesh, the reconstruction,
the boundaries nor the time integrator was ever the problem. The next thing to
try is a closure that is **local**: a source in cell `k` computed from the state
in cell `k`, as throughflow body-force models do, rather than from a remote
sample. The one piece of evidence already pointing that way is that
flux-weighted heat — the only genuinely local variant tried — is also the only
one that improved every speed.

### 3.17 Implicit integration — what it buys, and what it cannot

Backward Euler, BDF2 and variable-step SDIRK3 (Alexander's L-stable, stiffly
accurate three-stage scheme), all Newton–Krylov and matrix-free because the
source is non-local and the Jacobian is therefore not block-tridiagonal.

Measured temporal order against a finely-stepped reference, halving `dt`:

| scheme | rates, coarse → fine | with reconstruction forced 1st order |
| --- | --- | --- |
| backward Euler | 0.39, 0.54, 0.69, 0.81 | 0.87 |
| BDF2 | 0.71, 0.97, 1.27, 1.63 | 2.06 |
| SDIRK3 | 1.62, 2.21, 2.68, 2.89 | 2.97 |

Each reaches its design order, and the approach is slow because the van Albada
limiter is only piecewise differentiable — until the step is small enough that
no face changes limiter branch during it, every scheme reads one order low. The
first-order column is what identifies the limiter as the cause rather than a
coefficient slip, and it is why `tests/test_implicit.py` measures order at
n = 16, 32, 64 rather than 4, 8, 16.

**What it buys.** Freedom from the CFL limit on the stable range: 50× the
explicit step from a perturbed state, and a residual driven to round-off in tens
of steps rather than tens of thousands. `ImplicitStepper` is also the natural
place to converge a design point before a stability measurement.

**What it cannot buy.** The high-PR blockage. §3.16 shows the target is a
repeller, and an A-stable scheme integrating toward a repeller still leaves it —
A-stability bounds the response to *stable* modes and says nothing about
unstable ones. Backward Euler getting 5× further in physical time than explicit
was never evidence of progress; it was a slower walk down the same unstable
manifold.

### 3.18 The fix: lag the operating point, never the level

§3.16 leaves one question: what closure is both statically stable *and* free of
the fast loop? The answer follows from the corrected-parameter framework the
whole project is built on, and it is a **correctness** repair, not a stabiliser.

`Fx` and `SWx` are computed from the *lagged* sample. Injecting them as a fixed
force in newtons and a fixed heat rate in watts freezes the dimensional **level**
along with the operating point — and a map says nothing about levels. It asserts
a pressure *ratio* and a corrected work, both invariant to the absolute pressure
and to the mass flow. So the shipped injection is not a compressor: raise the
local pressure 10% and the map demands 10% more pressure rise, while a fixed
force delivers the same absolute rise. That inconsistency *is* the rising branch.

The repair is to lag the dimensionless operating point and never the dimensional
scale factors:

```
q_ρu(k) = (Fx/n) · p_k / p̄_k          p̄ = lagged local static pressure
q_ρE(k) = (SWx/n) · m_k / m̄_k         m̄ = lagged local mass flux
```

Both evaluated from the **local** state of the cell being forced — local is
essential, since the same corrections driven from the upstream probe make every
eigenvalue worse (§3.16). Both ratios are one at *any* steady state, so the
operating point is untouched and every Phase 3 gate still holds.

**The reference is a lag of the field, not a prediction of it.** The first
version integrated the expected profile across the smear from the map point,
which is exact to the flux inversion but not to the discretisation, and the
leftover mismatch biased the converged mass flow by **−5.5e−04** — three orders
outside the Phase 3 gate. Referencing the field against its own past cannot
drift, because at convergence past and present are the same field.

**It wants a smear, and that is not a free parameter.** The scaling divides by
the pressure of the cell being forced, and forcing a cell raises its own pressure
by `Fx/A`, so there is a self-interaction of gain of order `(PR−1)/(2·n_smear)`.
Every measurement below used `n_smear = 21`, where that is ~0.1 even at PR 5. At
`n_smear = 1` it is not small. Measured on the staged train of §3.14:

| total PR | stages | stage PR | `n_smear` | scaling **off** | scaling **on** |
| --- | --- | --- | --- | --- | --- |
| 2.0 | 4 | 1.189 | 1 | **held, −1.7e−07** | −6.9e−04 |
| 2.0 | 4 | 1.189 | 11 | **held, +1.3e−07** | −2.2e−06 |
| 4.0 | 2 | 2.000 | 11 | **blows up, step 12715** | survives, −4.2e−04 |

Eleven times the smear buys three hundred times the accuracy in the scaled case,
which is the `1/n_smear` dependence showing directly. And the third row is the
crossover: at stage PR 2.0 the *unscaled* form is the one that dies, so the
scaling stops being a cost and becomes a requirement right where §3.16 put the
threshold.

Hence deliberately asymmetric defaults: `ActuatorDisk` **off** (it is used at low
stage pressure ratio and defaults to `n_smear = 1`), `InletFlowCompressor`
**on** (that is where the high-PR work was validated, always at `n_smear = 21`).
Turning it on with a narrow smear should be expected to make things worse.

**Measured.** Largest eigenvalue of the linearised design state, `HPC01`:

| Nc | PR | fixed force / rate | similarity-scaled |
| --- | --- | --- | --- |
| 0.6 | 2.257 | +7.00e+00 | **−6.98e+01** |
| 0.7 | 3.104 | +4.41e+01 | **−5.39e+01** |
| 0.8 | 4.436 | +7.73e+01 | **−3.90e+01** |
| 0.9 | 5.991 | +9.99e+01 | **−2.75e+01** |
| 1.0 | 7.489 | +1.14e+02 | **−1.91e+01** |

Stable at every speed to PR 7.49. Two implementations — the library's and an
independent scratchpad wrapper — agree to all printed digits, which is what
rules out a bug in the rig.

Nonlinearly, on the committed `data/maps/SubsonicCompressor.xlsx` at Nc 1.0,
PR 2.141: unscaled reaches W = −1.03e−02 and is still drifting at −2.2e−02;
scaled **holds at +3.69e−10 with 5.1e−11 drift**. That case is now the gate in
`tests/test_similarity.py`, with the negative half asserted explicitly.

On `HPC01` the runs that used to die at steps **683, 95, 47 and 36** (Nc 0.8,
0.9, 1.0, 1.05) now survive; on `SubsonicCompressor` Nc 1.1 and 1.2, failures at
steps 3763 and 362 are gone.

### 3.43 Staged chains leave the β path, and the error trend changes with them

§3.21 and §3.24's staging results — six stages, OPR 119, "the error does not grow
with stage count" — are all `FlowMatchedCompressor` with an explicit `beta0`
seed. Staging could not use the β-free runtime at all, because
`ECMFMap.from_beta_map` reads the tabulated grids directly and `ScaledMap`
exposed none of them. The same gap as the turbines of §3.40, in another corner:
the map was β-free, the machine built from it was not.

`ScaledMap` now exposes its grids. `scale` is a ratio of corrected flows, so it
multiplies `Wc` and `ecmf`; `PR`, corrected work and efficiency are intensive and
do not scale — a stage is the same machine passing a different flow. Verified
identical to the β path at scale 1.0, 0.5 and 0.25.

**The same harness, both closures, `SubsonicCompressor`:**

| stages | OPR | exit `T₀` | β path | β-free |
| --- | --- | --- | --- | --- |
| 1 | 2.218 | 369 K | −1.648e−06 | **−1.788e−07** |
| 2 | 4.918 | 472 K | −2.454e−06 | **−6.981e−07** |
| 4 | 24.188 | 773 K | −1.697e−06 | **+2.224e−07** |
| 6 | 118.959 | 1265 K | **+4.415e−07** | +2.386e−06 |
| 8 | 585.056 | 2072 K | **+1.198e−06** | +3.156e−06 |

All four β-path values reproduce §3.24 exactly, so the harness is that section's.

**That table is `densify` 9, and the trend it appears to show is not real.** The
repo's tests all run `densify` 9 for speed while the sweeps run 36; §3.24's
staging results inherited it and so did these. §3.42 measures β = 9 as a
**~2.9e−06** interpolation floor against β = 36's 4.4e−08 — the same order as
every number in the column. Re-run at `densify` 36:

| stages | OPR | β path | β-free |
| --- | --- | --- | --- |
| 1 | 2.218 | −1.636e−06 | −1.267e−06 |
| 2 | 4.918 | −2.427e−06 | −1.984e−06 |
| 4 | 24.188 | −1.656e−06 | −1.172e−06 |
| 6 | 118.959 | +4.630e−07 | **+9.441e−07** |
| 8 | 585.053 | +1.208e−06 | **+1.694e−06** |

**The β-free chain does not accumulate.** At `densify` 9 its last three rows read
0.22 → 2.4 → 3.2 e−06 and looked monotone; at 36 they read 1.17 → 0.94 → 1.69
e−06, scatter around ~1.5e−06 with no direction. §3.24's property — per-stage
errors do not accumulate — holds for **both** closures. The claim that it was
specific to the β path, and the mechanism offered for it (neighbouring stages
perturbing each other's measured station), were an explanation for an artefact.

Two further things the re-run shows. **The β path is insensitive to map
resolution here** — 1.648 → 1.636, 2.454 → 2.427, 1.697 → 1.656, 0.4415 → 0.4630
e−06 across a 4× refinement — so its error is dominated by the closure rather
than the table, which is consistent with it carrying an internal β state a better
table cannot help. And **β-free is not uniformly better**: at `densify` 9 it
looked 3–8× tighter, at 36 the two are within ~25% of each other with β-free
ahead. The apparent advantage was mostly the coarse map flattering it.

**What that means for how many nodes can be stacked.** At `densify` 36 both
closures sit at 1–2e−06 across the whole range, with the 6-stage cells inside the
gate and the rest just outside it. Since the error does not grow with stage
count, there is no stage limit visible in this data — the binding constraint is
elsewhere.

**And the numerical limit is not the binding one.** At eight stages the exit
stagnation temperature is **2072 K**, past turbine-entry conditions, and the
repeating-stage assumption — identical corrected work from every stage, which is
what makes OPR compound geometrically to 2.2177⁸ = 585 — describes no machine
anybody builds. The arithmetic stays self-consistent long after the model stops
meaning anything.

**The lesson is the floor, again.** Every staged number in this project — §3.21,
§3.24, and the first version of this section — was computed on a `densify` 9 map
whose own interpolation floor is the same size as the results. §3.42 established
that floor an hour before this section was written and it was not applied here.
A measurement should be read against the resolution it was taken at, and the
repo's tests and its sweeps do not run at the same one.

### 3.42 The two densification axes are not the same quantity

`densify` took one factor for both axes, and 36 came from §3.30's **β-only**
series. `nc_factor` now refines them separately, which makes the question
answerable.

**β is the dominant axis, and 36 is near its minimum.** At Nc = 36:

| β | `HighPqPCompr` on-line | mid-interval | `RadialTurbine` on-line |
| --- | --- | --- | --- |
| 36 | 4.37e−08 | 1.97e−07 | 1.95e−08 |
| 18 | 9.48e−07 | 8.18e−07 | 4.98e−08 |
| 9 | 2.94e−06 | 2.06e−06 | 6.04e−07 |
| 4 | 1.72e−05 | 1.54e−05 | 2.88e−06 |
| 1 | 2.98e−04 | 2.86e−04 | 2.72e−05 |

Smooth, monotone, roughly second order, and the same on and off the tabulated
lines. Halving β costs 22×. That is not headroom.

**The Nc axis needed three attempts to measure, and the failures are the
lesson.** Sampling operating points taken from the *raw* speed list said Nc was
worthless — 36 → 1 identical at 1/26 the memory. True, and useless: a point on a
stored line never blends across speeds, so the axis is inert by construction
there. Sampling the exact *midpoint* between lines then said Nc = 18 was free.
Also wrong: an even `nc_factor` puts the midpoint precisely on a refined node.
Sampling 0.27/0.5/0.73 across the interval, β = 36:

| Nc | MB | `HighPqPCompr` worst | `RadialTurbine` worst | midpoint reading |
| --- | --- | --- | --- | --- |
| 36 | 5.3 | 7.73e−07 | 3.05e−07 | 1.29e−07 |
| 18 | 2.6 | 1.64e−06 | 8.20e−07 | 1.29e−07 |
| 12 | 1.8 | 5.44e−06 | 1.56e−06 | 1.29e−07 |
| 6 | 0.9 | 2.70e−05 | 1.47e−05 | 1.29e−07 |
| 2 | 0.3 | 2.51e−04 | 1.54e−04 | **1.29e−07** |
| 1 | 0.2 | 9.84e−04 | 2.92e−04 | 9.84e−04 |

**The midpoint column reads 1.29e−07 all the way down to Nc = 2, where the true
worst is 2.51e−04 — a factor of 1,900.** It only breaks at Nc = 1, where no
intermediate nodes exist for it to land on. A probe aligned with the grid it is
measuring reports the one position where the interpolation is exact by
construction.

Three sample positions still under-resolve: `RadialTurbine` reads 1.47e−05 at
Nc = 6 and 1.47e−06 at Nc = 4, because where the worst case sits moves with the
grid. **The `worst` column is a lower bound on the worst, not the worst**, and no
specific Nc threshold should be quoted from it.

**And the two axes are not measuring the same kind of error.** Numerically they
are identical — PCHIP on both at densify, linear on both at runtime. Physically
they are not:

| | β | Nc |
| --- | --- | --- |
| between the nodes lies | the same measured speed line, resampled | speed lines **never measured** |
| refinement converges to | the vendor's data | a *model* of variation with speed |
| assumption carried | none — β is a label, the curve is invariant to it | PCHIP across Nc, **at fixed β** |

β is a construct, and that is exactly why refining along it is safe: it
parameterises a curve that was measured, and the curve does not depend on the
parameterisation. Nc lines are physical, and that is exactly why refining
*between* them is not safe in the same sense — it invents states, and the
correspondence it invents them along is fixed β, which is the construct doing
load-bearing work.

**The consequence is the one worth keeping.** This project's own
leave-one-speed-line-out figures put the cross-speed model error at **1.79%
(compressors) and 2.71% (turbines)**. The convergence gate is 1e−06 — *0.0001%*.
So off the tabulated lines the map is four orders of magnitude less certain than
the number being converged to, and a mid-interval "error" of 7.7e−07 means the
solver reproduces the *interpolated* map that well, not that the answer is right
that well. At a tabulated speed the two coincide, because there the map is data.
Between them they do not. Refining Nc buys self-consistency and repeatability; it
does not buy fidelity, and no amount of it will.

### 3.41 The whole library, measured: 1388/1428 across all nineteen maps

Every figure before this section — 261/315, 272/315, **292/315** — is four maps.
`SubsonicCompressor`, `TranssonicCompressor`, `HighPqPCompr`, `TwoStgRadialCompr`.
They were repeatedly described in this document and in conversation as "the
library". There are **nineteen** supplied maps and the library is 1428 cells.

**Compressors and fans, 944/980:**

| map | | map | |
| --- | --- | --- | --- |
| `HPC02` | 49/49 | `LowBPRFan` | 91/91 |
| `HighPqPCompr` | 70/70 | `LowPqPCompr` | 77/77 |
| `IPC01` | 63/63 | `SingleStgRadialCompr` | 77/77 |
| `SubsonicCompressor` | 84/84 | `TranssonicCompressor` | 63/63 |
| `MediumPqPCompr` | 97/98 | `LPC01` | 68/70 |
| `LPC02` | 68/70 | `HPA01` | 62/70 |
| `TwoStgRadialCompr` | 75/98 | | |

**Turbines, 444/448** (§3.40's closure, `densify` 36, inlet Mach derived per cell
from the exit-choke constraint):

| map | | map | |
| --- | --- | --- | --- |
| `SingleStgTurbine` | 35/35 | `Ipt01` | 49/49 |
| `RadialTurbine` | 49/49 | `MediumPqPTurbine` | 70/70 |
| `HighPqPTurbine` | 104/105 | `TwoStgTurbine` | 137/140 |

**Nine of these maps had never been swept**, and they were not easy ones:
`MediumPqPCompr` holds 97/98 with its top three speed lines uninvertible;
`LowBPRFan` 91/91 with 28/28 on refused lines. That is §3.30's ECMF keying
generalising to data it was not developed against, rather than fitting the four
maps it was built on.

**The four originals reproduce exactly on this HEAD** — 84/84, 63/63, 70/70,
75/98, digit for digit against sweeps run five commits earlier, alongside
bit-identical `evaluate` over 1,665 lookups including both off-table branches and
505 design points. So the turbine work cost the compressors nothing, and that is
measured rather than argued.

**`HPA01` is the one compressor map with real deaths, and they are surge.**
Eight cells, spanning PR 2.5 to 7.5 — while PR **11.500** holds on the same
column. Not a pressure ceiling. Measuring `dPR/dECMF` at all 70 design points:

| | count |
| --- | --- |
| slope ≥ 0 and died | 7 |
| slope < 0 and held | 62 |
| slope ≥ 0 but held | **0** |
| slope < 0 but died | 1 |

**69 of 70 classified by slope sign alone.** The lines whose PR peak sits *at*
`f` = 0.000 — Nc 0.500, 0.950, 1.000, 1.050 — hold every cell; the lines whose
peak is interior lose exactly their `f` = 0.0 cell. The single exception has
slope −0.1 against its neighbours' −0.6: falling, but too flat to restore, which
is §3.34 unchanged. `HPA01` is tabulated past its own surge line on six of ten
speed lines, arrived at independently of `TwoStgRadialCompr`.

**So the 40 failures are three known things and nothing else:** the surge branch
(`HPA01` 7, `TwoStgRadialCompr` ~20), the off-table/degenerate ends, and three
turbine cells sitting exactly on the 1e−06 gate at the `densify` 36 floor.
**Zero unexplained deaths in 1428 cells.**

**The correction worth keeping.** "The library" meant four maps for most of this
document's life, and the number was quoted as though it meant nineteen. A
coverage claim needs its denominator stated every time, because the denominator
is the part that silently shrinks.

### 3.40 One key for both machines — and the reason there were two was a bug

§3.39 ran turbines on `FlowMatchedCompressor` with a `beta_map` and an explicit
`beta0` seed: the legacy β-driven closure. So it validated the turbine *physics*
on the path the compressors had already left. Wiring turbines into the β-free
disk exposed something worse than a missing feature.

**The inherited claim.** `ECMFMap`'s own docstring, predating this work, said
ECMF was monotonic on 7 of 22 turbine speed lines against `PR`'s 22 of 22, and
concluded *"a turbine wants its own key"*. §3.38 built a PR-keyed table on that.

**It was measured with the bug §3.38 fixed.** Every turbine τ was a temperature
*rise* instead of a drop and the ECMF factor was inverted. Recomputed over all
six turbine maps:

| ECMF monotonic in β | lines |
| --- | --- |
| with §3.38's broken thermodynamics | **17 / 64** |
| with the correct thermodynamics | **64 / 64** |

**And PR is the one key a turbine disk must not use.** The disk *asserts* a
pressure ratio, so `p₀₂ = p₀₁/PR_measured` is an identity: the source reproduces
the ratio it just read from its own station, and nothing anchors it. Measured on
the PR key, every map converged to a small one-signed residual and stayed there —

| | PR key | ECMF key, `densify` 36 |
| --- | --- | --- |
| `Ipt01` | −3.4e−03 | **+7.1e−08** |
| `RadialTurbine` | −5.6e−03 | **+1.4e−07** |
| `SingleStgTurbine` | −6.0e−03 | **+2.3e−08** |
| `HighPqPTurbine` | −2.7e−03 | **+1.1e−07** |
| `MediumPqPTurbine` | −4.4e−04 | **+3.6e−07** |
| `TwoStgTurbine` | −1.3e−02 | **+3.1e−07** |

— **11 of 11 buildable cells inside the gate**, including `TwoStgTurbine`, which
*died* on the β path of §3.39.

The diagnosis took a wrong turn first, and the wrong turn is what identified it.
Reading the key off the seeded design field showed `exit_offset` 2 carrying a
**95 ppm** bias where offset 3 was exact — §3.36 had chosen 2 for compressors
noting "3 is equally good", and for a compressor it is. Fixing it improved
`Ipt01` fivefold and `RadialTurbine` not at all. A station-reading error would
have moved both. That insensitivity is what pointed at the identity rather than
the reading.

**What this deletes.** The PR-keyed table path, the tabulated `Wc` column, and
the turbine branch in `_exit_ecmf` — which never needed to exist, because that
method measures `p₀₂/p₀₁` from the *field*, and the two machines differ only in
which way up they *store* that ratio. `Wc·√τ/pr` was always ECMF for either.

The one real asymmetry left is recovering `Wc` from the key: `ECMF = Wc·√τ/PR`
for a compressor storing `p₀₂/p₀₁`, and `Wc·√τ·PR` for a turbine storing
`p₀₁/p₀₂`. Verified reproducing the key to **2.22e−16** on both.

**So the runtime is one closure.** Same disk, same ECMF key measured the same
way, same momentum and energy sources, no mass source, no β. What differs is
read from the map: which way up `PR` is stored, and the sign of the work.

**The methodological failure is the point.** §3.38 found the turbine
thermodynamics wrong and fixed them. It did not re-measure the conclusions that
had been *derived* from those thermodynamics, and the ECMF-monotonicity number
was one of them. A fix invalidates every measurement downstream of the bug; that
has to be treated as work the fix creates, not as prior art it inherits. This is
the fourth time in this project a number chosen under one regime turned out not
to survive the next — §3.30's `exit_offset`, §3.36's equivalence test, §3.37's
β labels, and now this.

### 3.39 Turbines run: five of six hold at 1e−09, on the compressor's own closure

The maps of §3.38 loaded but nothing downstream accepted them. Two things stood
in the way and neither was deep.

**`p02 = PR·p01`, at six sites.** Both machines tabulate a ratio above 1 and they
are not the same ratio, so the compressor form on a turbine *raises* the pressure
while the energy source *lowers* the temperature — a disk that compresses and
cools at once. Dimensionally fine, physically impossible, and silent. Now
`_exit_p0(PR, p01, kind)`, with the kind read off a `BetaMap` or inferred from an
`ECMFMap`'s `key_field`, since only a turbine is keyed on `PR`.

**A hard-coded `not a compressor` guard** in `design_from_map`, refusing negative
work. The direction of the temperature change is the machine's definition, so it
is now checked against the machine. The equal-area exit-choke guard becomes
compressor-only: for a turbine the exit is the *low-pressure* station, so
`Φ₂ = Φ₁·PR/√τ > Φ₁` and the exit saturates first — the reverse of a compressor,
where §Phase 1 showed the guard was unreachable.

**Inlet conditions were the substance, not a detail.** At a standard-day inlet
these maps expand to **169–210 K** and the design refuses them, correctly: the
map asks for more enthalpy than the flow carries. At **1600 K / 1200 kPa** all
six build, at 1172–1439 K. `Δh₀ = CW·θ`, so the drop scales with inlet
temperature and only turbine-entry conditions put the exit anywhere sensible.

| turbine | M₁ | PR | p₀ | T₀ | SWx |
| --- | --- | --- | --- | --- | --- |
| `Ipt01` | 0.45 | 1.497 | 1200→802 kPa | 1600→1439 K | −48.0 MW |
| `MediumPqPTurbine` | 0.35 | 1.920 | 1200→625 kPa | 1600→1353 K | −465.9 MW |
| `HighPqPTurbine` | 0.25 | 2.516 | 1200→477 kPa | 1600→1303 K | −3.5 MW |
| `RadialTurbine` | 0.25 | 2.524 | 1200→475 kPa | 1600→1301 K | −4.4 MW |
| `SingleStgTurbine` | 0.25 | 2.719 | 1200→441 kPa | 1600→1238 K | −5.7 MW |
| `TwoStgTurbine` | 0.15 | 3.578 | 1200→335 kPa | 1600→1172 K | −39.4 MW |

The inlet Mach a turbine tolerates falls with its expansion ratio, which is
`Φ₂ = Φ₁·PR/√τ` read backwards.

**Marched to steady state, seeded from the design field:**

| turbine | f = 0.15 | f = 0.5 |
| --- | --- | --- |
| `Ipt01` | −2.4e−09 | −1.6e−09 |
| `RadialTurbine` | −5.1e−10 | −3.0e−10 |
| `SingleStgTurbine` | −6.2e−10 | −1.5e−09 |
| `HighPqPTurbine` | −5.3e−10 | −2.8e−10 |
| `MediumPqPTurbine` | −1.0e−09 | −7.0e−10 |
| `TwoStgTurbine` | died 16200 | died 4658 |

**The closure needed no change at all.** The disk is a momentum source `Fx` on
`q[1]` and an energy source `SWx` on `q[2]`, with no mass source, and that form
is machine-agnostic: it is Δ(pressure force) + Δ(momentum flux), and `W·Δh₀`.
Once `corrected_work` carries the right sign the whole thing runs backwards on
its own. Everything that had to change was a *convention* — which ratio `PR`
names, which direction temperature moves — and none of it was physics.

**`TwoStgTurbine` fails with `mass flux must be positive`** — flow reversal, at
the largest expansion ratio and therefore the lowest inlet Mach the duct will
take. Lowering M₁ further makes it worse (dies at 2336 rather than 16200), so it
is not a sizing knob: a strong source in a slow duct reverses it, which is the
turbine-side counterpart of §3.34's surge branch rather than a defect in the
closure. `f = 0.85` is a build refusal on every map, the exit-choke limit at the
high-expansion end of each speed line.

**Compressors verified unchanged throughout**: 427 design points across all
eleven maps comparing `p02`, `T02`, `Fx`, `SWx`, `area` and `p_back` against
§3.37's `964d177` — **0.000e+00**.

**Open.** A turbine sweep at the scale the compressors got (every speed line ×
seven positions), per-position inlet Mach so the choke-limited column can be
reached at all, and the `TwoStgTurbine` reversal.

### 3.38 Every turbine map was being read as a compressor

Three of the seventeen supplied workbooks would not load — `HighPqPTurbine`,
`RadialTurbine`, `TwoStgTurbine`, all raising `non-physical temperature ratio`.
The obvious reading is that turbines need hot, high-pressure inlet conditions
and were being handed ambient ones. That reading is wrong, and the algebra says
so before any measurement: the check is

```
dh0s = cp·T_ref·(PR^k − 1);   cw = dh0s/η;   τ = 1 + cw/(cp·T_ref)
```

in which **`T_ref` cancels exactly**, leaving `τ = 1 + (PR^k − 1)/η` — a function
of two workbook columns and nothing else. No inlet condition can move it. Inlet
temperature and pressure matter at design and solve, which is code these three
maps never reached.

**What was actually wrong: three errors stacked in one expression.**

| | used | correct for a turbine |
| --- | --- | --- |
| isentropic relation | `cp·T_ref·(PR^k − 1)` — a compression *rise* | `cp·T_ref·(1 − PR^−k)` — an expansion *drop* |
| efficiency | `work = ideal/η`, more than ideal | `work = ideal·η`, less than ideal |
| ECMF factor | `Wc·√τ/PR`, assuming `PR = p₀₂/p₀₁` | `Wc·√τ·PR`, the expansion ratio |

Together they gave **every cell of every turbine map a temperature rise**. A
PR 3, η 0.9 stage came out at **+41% where it must drop 24%**.

**And the crash was the lucky half.** Three maps raised; the other three loaded
and were quietly wrong. `MediumPqPTurbine` at PR 1.0642 with η = −0.0339 derived
**−153,401 J/kg** against an ideal Δh₀ of ≈ 5,190 — a 53% temperature drop across
a 6% pressure ratio — and passed `τ > 0` on magnitude alone. A loud failure on
three files hid a silent one on a fourth.

**Kind is inferred from the machine, not declared.** A compressor throttled
toward surge passes less flow at more pressure ratio; a turbine passes more as
the expansion ratio opens. The sign of `dWc/dPR` therefore separates them, and
over all seventeen workbooks it is unanimous — **positive on 64 of 64 turbine
speed lines, negative on 109 of 109 compressor lines**, no map mixed. Preferred
to the other unanimous signal (compressors carry a `surge_line` sheet, turbines
do not) because that is a filing convention and this is the machine. `kind=`
overrides it.

**η ≤ 0 is repaired on turbines only, and the asymmetry is the finding.** The old
docstring declined to special-case those cells because "the numerator changes
sign with the denominator". Measured: that holds on **12 of 12** such compressor
cells and fails on **35 of 36** turbine cells. On a compressor they all sit at
PR < 1, so the cancellation is real and yields τ > 1 — work in, temperature up,
pressure down, a stalled corner correctly modelled. **It was right about
compressors and wrong only about turbines**, so compressor cells are left exactly
alone; the first version of this fix repaired both and broke `IPC01`, `LPC01` and
`LPC02` by turning that corner into cooling. Turbine repairs interpolate η along
β from the valid cells of the same line, are listed in
`BetaMap.repaired_efficiency`, and a line with no usable cell is refused rather
than invented.

**All six turbines now load, densify, and key on PR:**

| map | grid | η repaired | τ range | max work out |
| --- | --- | --- | --- | --- |
| `HighPqPTurbine` | 50×15 | 18 | 0.585–1.000 | 120.0 kJ/kg |
| `Ipt01` | 20×7 | 0 | 0.802–0.987 | 57.4 kJ/kg |
| `MediumPqPTurbine` | 30×10 | 3 | 0.731–0.999 | 78.0 kJ/kg |
| `RadialTurbine` | 20×7 | 1 | 0.726–0.981 | 79.4 kJ/kg |
| `SingleStgTurbine` | 20×5 | 0 | 0.726–0.901 | 79.2 kJ/kg |
| `TwoStgTurbine` | 20×20 | 14 | 0.599–1.000 | 116.0 kJ/kg |

**The key follows the machine.** ECMF is monotonic in β on 135 of 135 compressor
lines but only **17 of 64** turbine lines, where `PR` manages **64 of 64**.
Neither generalises, so `ECMFMap.from_beta_map` dispatches on `kind`, and the
key array is named `key` with `key_field` saying what it holds. `.ecmf` survives
as a property that returns it for a compressor and **raises** on a PR-keyed map:
handing back `PR` under the name `ecmf` is exactly the class of error that cost
§3.37 an afternoon, and it is worth a loud attribute error to make impossible.

`densify` needed the same lesson. It rebuilt a `BetaMap` without `kind` and
re-derived η and ECMF with the compressor forms, so a densified turbine reverted
to a compressor one call after the loader decided otherwise — silently, since
both forms are dimensionally fine.

**Compressors are untouched, and that is verified rather than asserted.**
Bit-identical to §3.37's `964d177` at load (11 maps × 5 arrays) and through
`evaluate` (15 keys per speed line spanning −0.2 to 1.2 of the range, so both
off-table branches, × 5 returned fields): **0.000e+00**. That included keeping
`ecmf` as a division rather than a reciprocal multiply, which differs in the
last bit — a one-ULP drift that the first draft of this change introduced and
that the comparison caught.

**Still open.** The turbines load and are keyed; they have not been *run*. The
disk closure, the design path and the sweep harness are all written around a
compressor — a turbine's corrected work is negative, which at minimum inverts
what the source term does to the flow — so end-to-end turbine operation is the
next piece, and it is a larger one than this.

### 3.37 A clamped map end *is* a surge condition, and the library closes at 292/315

**The convention, measured, because it was being used backwards.** On all four
compressor maps `Wc` and ECMF both **fall** with β: β = 1 is the low-flow,
high-PR end — **surge** — and β = 0 is choke. The sweep grid's `f`, however, is a
fraction along the **ECMF range** (`_speed_line(nc)[0]` is ECMF, not β), so

> **`f` ≈ 1 − β. `f` = 0.0 is the surge end; `f` = 1.0 is choke.**

Every `f` in §3.30–§3.36 is correct as written. Summaries that re-expressed those
columns "in β" inverted them, and §3.27 and §3.30 each carried one such sentence;
both are fixed above. Recorded because the error is silent — surge and choke both
sit at a table end, so an inverted label still points at a failing column.

**The mechanism.** `np.interp` clamps, so outside the table the characteristic is
**flat**. A flat characteristic is exactly the zero-restoring-force condition
§3.34 identifies as surge — so a clamped lookup *manufactures* an artificial
surge at every table end, and any point that wanders out is pinned there rather
than pushed back. §3.34 saw the step and blamed the map's missing data; the
missing data is not the problem, the missing *gradient* is.

Holding the terminal slope for one speed-line width past each end, `HighPqPCompr`:

| | Nc 0.950 f = 0.0 | Nc 0.950 f = 1.0 | Nc 1.025 f = 0.0 | Nc 1.025 f = 1.0 |
| --- | --- | --- | --- | --- |
| clamped | −7.54e−02 | +2.08e−03 | −2.3e−01 | +5.3e−03 |
| slope held | **−1.03e−10** | **−2.39e−11** | **+3.04e−11** | **+1.74e−10** |

Bounded at one width because §3.8's finding still governs the *value*; it is the
gradient the near field needs. The bound is invisible to a converging run — six
cells reproduce the unbounded result exactly — and it keeps the far field from
being a different way to return a wrong number confidently.

**The library, complete:**

| map | before today | §3.32 | §3.36 | **§3.37** |
| --- | --- | --- | --- | --- |
| `SubsonicCompressor` | 80/84 | 84/84 | 84/84 | **84/84** |
| `TranssonicCompressor` | 20/45, five lines unbuildable | 63/63 | 63/63 | **63/63** |
| `HighPqPCompr` | 21/70 | 46/70 | 56/70 | **70/70** |
| `TwoStgRadialCompr` | — | 68/98 | 69/98 | **75/98** |
| **total** | — | 261/315 | 272/315 | **292/315 (92.7%)** |
| on lines the inverse refuses | 0/91 | 64/91 | 75/91 | **90/91 (99%)** |

`HighPqPCompr` holds **every cell of every speed line, to PR 28.889**, against a
project ceiling of 14.972 this morning.

**Why the radial does not, and why that is not a defect.** The two maps respond
oppositely because their data ends in different places:

| map | β = 1 at the PR peak | past it |
| --- | --- | --- |
| `HighPqPCompr` | **10 / 10 lines** | 0 |
| `TwoStgRadialCompr` | 2 / 14 | **12** |

`HighPqPCompr` stops exactly at surge, so its flat end was purely an artefact —
remove it and the map completes. `TwoStgRadialCompr` is tabulated 0.3–3.4% *past*
its own peak on 12 of 14 lines, so its surge-side columns have `dPR/dECMF ≥ 0`.
The clamp had been flattening that into false passes. Slope sign against outcome,
all 98 cells:

| | count |
| --- | --- |
| slope ≥ 0 and failed | 17 |
| slope < 0 and held | 74 |
| slope ≥ 0 but held | 1 |
| slope < 0 but failed | 6 |

**91 of 98 classified by slope sign alone.** Of the six it misses, three are the
`densify` cells below and three are `f` = 0.15 cells whose slope is negative but
roughly half their neighbours' — §3.34's "too flat to restore". The control is
clean: Nc 1.070 and 1.100, the only two lines whose data stops at the peak, are
**7/7 both**. So 20 of the radial's 23 failures are the model correctly refusing
a statically unstable branch, and closing them would mean truncating the supplied
map at its peak or modelling surge dynamics — not fixing a bug.

**The β-resolution cells, now measured rather than extrapolated.** §3.36 asserted
`densify` 72 would clear them from §3.30's series. Run directly:

| cell | d18 | d36 | **d72** |
| --- | --- | --- | --- |
| Nc 0.600 f 0.65 | 7.07e−06 | 1.25e−06 | **4.51e−07** |
| Nc 0.600 f 0.85 | 4.94e−06 | 2.12e−06 | **7.17e−07** |
| Nc 0.650 f 0.65 | 3.12e−06 | 1.14e−06 | **1.59e−07** |

Monotone in every cell, mean order ≈ 1.8, all three inside the gate at 72.

**And `densify` 72 is now free**, which it was not when §3.36 advised against it.
`evaluate` slices a column six times per call; on a C-ordered array that slice is
strided, so numpy copied the whole speed line before interpolating — the lookup
was O(n_key) in a memcpy, not O(log n_key) in a search. Fortran order makes it a
contiguous view:

| | densify 36 | densify 72 |
| --- | --- | --- |
| `HighPqPCompr` | 35.3 → **26.0 µs** | 107.3 → **25.2 µs** |
| `TwoStgRadialCompr` | 62.4 → **26.5 µs** | 180.8 → **25.7 µs** |

Flat in grid size instead of linear in it, values bit-identical. The earlier
recommendation against 72 was right for the old layout and wrong for this one.

**What `off_table` now means.** §3.34 held that a converged run with a non-zero
count was not to be trusted, because a count meant pinning. It no longer does:
the four cells above converge at 1e−10 to 1e−11 while logging 1281 to 12442
excursions each. Leaving the table during the transient is now normal and
expected; the counter records that it happened, and nothing more.

**One pre-existing test failure, verified not mine.**
`test_flowmatch.py::TestItWorksWhereTheInverseRefuses::test_it_holds_the_operating_point_anyway[0.85]`
fails at PR 1.538 with `W` off by −6.227e−02. Run at HEAD and at 96bd4aa, the
commit that introduced it: **identical to every digit**, so it has failed since
it was written and nothing in §3.32–§3.37 touched it. It exercises
`FlowMatchedCompressor`, whose `exit_offset` defaults to `None` and which
therefore takes the β path that none of this work goes near.

Worth keeping rather than silencing, because of what it measures: `EcmfCompressor`
holds that exact cell — `TranssonicCompressor` Nc 0.880, `f` = 0.85, PR 1.538 —
in every sweep today. The failing test is a limitation of the closure being
replaced, on a point the replacement handles. Suite otherwise: **339 passed, 1
failed, 1 skipped.**

**Open, and what each is worth.** Three turbine maps still fail to load and a
fourth — `MediumPqPTurbine` — loads *silently corrupt*: at PR 1.0642 with
η = −0.0339 the loader derives −153,401 J/kg against an ideal Δh₀ of ≈ 5,190,
implying a 53% temperature drop across a 6% pressure ratio, and passes the τ > 0
check on magnitude alone. The cause is not inlet conditions — `T_REF` cancels
exactly out of `τ = 1 + (PR^k − 1)/η` — but that the loader applies compressor
thermodynamics to turbines: `Δh₀ˢ = cp·T_ref·(PR^k − 1)` is the compression
relation, `work = ideal/η` is the compression convention, and the docstring's
justification for not special-casing η ≤ 0 ("the numerator changes sign with the
denominator") holds on 12 of 12 compressor cells and fails on **35 of 36 turbine
cells**. That, plus keying turbines on PR — monotone on 22/22 lines where ECMF
manages 7/22 — is the turbine work, and it is larger than it looked.

### 3.36 The exit station was reading across the source jump — one cell fixed it

§3.35 measures the exit station under-reading ECMF by 4.6% at PR 4 and 35% at
PR 25, on the exact seeded design field, and leaves the cause open. Decomposing
the reading finds it immediately:

| PR | ``W`` ratio | ``T₀₂`` ratio | ``p₀₂`` ratio | ECMF ratio |
| --- | --- | --- | --- | --- |
| 3.95 | **0.947** | 0.991 | 0.989 | 0.954 |
| 15.88 | **0.763** | 0.985 | 0.986 | 0.768 |
| 21.85 | **0.684** | 0.982 | 0.986 | 0.688 |
| 25.13 | **0.643** | 0.981 | 0.986 | 0.646 |

Stagnation pressure and temperature are right to 1–2%. The whole error is the
**mass flux** — on a field where mass flux is uniform by construction.

**The cause.** `station_state_at` reads a cell's *upstream* face. At
`exit_offset` 1 that face lies between the last forced cell and the first
unforced one, and those states differ by one cell's share of the source: at PR 25
with `n_smear` 7 a **58% pressure jump across a single face**. Roe's dissipation
term scales with the state jump and corrupts the mass flux across it. At
`exit_offset` ≥ 2 both cells straddling the face carry the same state, the jump
is zero, and the flux is exact:

| PR | offset 1 | offset 2 | offset 3 |
| --- | --- | --- | --- |
| 3.95 | 0.9539 | 0.9983 | **1.0000** |
| 15.88 | 0.7681 | 0.9970 | **1.0000** |
| 21.85 | 0.6881 | 0.9972 | **1.0000** |
| 25.13 | 0.6460 | 0.9973 | **1.0000** |

**End to end, on the cells that died:**

| case | offset 1 | offset 2 | offset 3 |
| --- | --- | --- | --- |
| Nc 1.000 f 0.50, PR 21.846 | died 1399 | **+3.7218e−08** | +3.7221e−08 |
| Nc 0.975 f 0.85, PR 16.591 | died **2** | **+1.8625e−07** | +1.8625e−07 |
| Nc 1.025 f 0.50, PR 23.470 | died 1063 | **+8.4293e−08** | +8.4292e−08 |

All with `off_table` = **0**: with a correct reading the demand never leaves the
table, and the failure chain never starts. Offsets 2 and 3 agree to five digits,
as they must — both read a face with no jump across it.

**How it got in, which is the part worth remembering.** §3.30 chose
`exit_offset` 1 by measuring offsets 1, 2 and 3 as "identical to five digits" at
PR 3.36 and taking the closest, because 4 and 8 destabilise through transport
delay. At PR 3.36 the offset-1 error is 4.6% and the closure absorbs it entirely
— so the measurement that chose the default was *structurally incapable of
seeing what it was choosing*. The same shape of mistake as the `ECMFMap`
equivalence test in §3.30, which sampled only tabulated speeds.

Default is now `exit_offset` 2, with 3 equally good and 4+ excluded by §3.30.
`tests/test_flowmatch.py` pins the *jump across the read face* rather than a
converged answer at one pressure ratio, because that is the quantity that
actually differs.

**The full re-sweep confirms it, and the table changes shape.** `HighPqPCompr`
with `exit_offset` 2 — **56/70**, against 46/70 at offset 1 and 21/70 before
§3.32, with refused lines 25/35 against 15/35:

| Nc | f=0.0 | f=0.15 | f=0.35 | f=0.5 | f=0.65 | f=0.85 | f=1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.800 | −3.2e−05 | ok | ok | ok | ok | ok | ok |
| 0.850 | −1.1e−03 | ok | ok | ok | ok | ok | ok |
| 0.900 | −1.2e−02 | ok | ok | ok | ok | ok | +1.2e−05 |
| 0.925 | −3.3e−02 | ok | ok | ok | ok | ok | +1.3e−04 |
| 0.950 | −7.5e−02 | ok | ok | ok | ok | ok | +7.6e−04 |
| 0.975 | −1.3e−01 | ok | ok | ok | ok | ok | +2.1e−03 |
| 1.000 | −2.0e−01 | ok | ok | ok | ok | ok | +4.1e−03 |
| 1.025 | −2.3e−01 | ok | ok | ok | ok | ok | +5.3e−03 |

**Thirteen DIED cells became zero.** Every interior cell on every speed line
holds, to **PR 26.999**. What is left is exactly the two end columns, mirror
signed and monotone in PR — §3.34's off-table clamp, which that section
demonstrates is fixed by moving 1% inside the table (−7.54e−02 → +4.14e−08 on
this map). Not a closure defect.

`TwoStgRadialCompr` reproduces offset 1 almost row for row — **69/98** against
68/98 — because its failures are clamp and surge, not station. Its one gain is
Nc 1.100 f = 0.5 at PR 17.067, which died at offset 1 and holds now. Deaths there
go 8 → 7, and the seven are the f = 0.15 surge column.

**The library, complete:**

| map | before today | after §3.32 | **final** |
| --- | --- | --- | --- |
| `SubsonicCompressor` | 80/84 | 84/84 | **84/84** |
| `TranssonicCompressor` | 20/45, five lines unbuildable | 63/63 | **63/63** |
| `HighPqPCompr` | 21/70 | 46/70 | **56/70** |
| `TwoStgRadialCompr` | — | 68/98 | **69/98** |
| **total** | — | 261/315 | **272/315 (86.3%)** |
| on lines the inverse refuses | 0/91 by construction | 64/91 | **75/91 (82%)** |

*(The `SubsonicCompressor` figure is carried from the offset-1 run: the re-sweep
was stopped at 11 of 12 rows once it had reproduced all 11 identically, the
twelfth having never failed under any configuration.)*

**Every remaining failure is characterised**, which has not been true before:

| count | what | status |
| --- | --- | --- |
| **32** | called "off-table clamp at the exact table ends" — 18 at f = 0.0, 14 at f = 1.0 | **This grouping is wrong; §3.37 splits it.** 14 are a clamp artefact and are now fixed; the rest are the radial's unstable branch and are correct failures. |
| 8 | radial f = 0.15, past the PR peak — 7 deaths and one −7.2e−01 | correct physics, not a defect |
| 3 | map β-resolution near-misses, ~1e−06 (Nc 0.600/0.650, f = 0.65/0.85) | confirmed in §3.37 by measurement rather than extrapolation |

Counted, not estimated. An earlier draft of this table said ~28 / 7 / 3 / ~5, splitting
the clamp across two rows and filing the −7.2e−01 surge cell under the clamp; the
split above is the measured one. **Of the 43, only the 32 are outstanding work** —
and a single clamp fix would take the library to roughly 304/315.

*That estimate was wrong, and §3.37 says how.* The 32 were not one group: 14 were
a clamp artefact and cleared, while the rest are `TwoStgRadialCompr` sitting past
its own PR peak, where failing is correct. The fix landed 292/315, not 304 — and
the shortfall is not missing work but cells that should never have been counted
as recoverable.

**One prediction was wrong and is worth recording.** I expected the choke-end
cells (f = 1.0) to clear, having assigned them to the station. They improve —
+2.0e−04 to +1.3e−04 at Nc 0.925, +2.1e−03 to +7.6e−04 at Nc 0.950 — but do not
clear, because f = 1.0 *is* the choke table end and therefore the clamp. Both end
columns are clamp-limited; the station fix only ever addressed the interior.

### 3.35 The exit station under-reads ECMF, and the error scales with PR

§3.34 leaves one family unexplained: the deaths on `HighPqPCompr` Nc 0.975–1.025
at f = 0.35–0.85. They are not surge (the PR peak is at f = 0.0000 on every one
of those lines, so all interior cells are on the stable branch), not the
off-table clamp (they sit far from either end), and not the acoustic loop —
`key_lag` 0.01/0.03/0.1/0.3 die at steps 1399/2432/1089/1089, delayed
non-monotonically and never prevented. Seeding was checked and is clean: the
seeded disk profile runs Mach 0.155 down to 0.028 with no infeasible inversion.

**Watching one die settles it.** `HighPqPCompr` Nc 1.000, design ECMF 9.2191 in a
table spanning [7.7336, 10.7046]:

| step | W err | PR applied | ECMF key | max M | off-table |
| --- | --- | --- | --- | --- | --- |
| 100 | +3.35e−01 | 26.885 | **7.7334** | 0.878 | 100 |
| 500 | −1.16e−01 | 21.244 | 9.4493 | 0.759 | 190 |
| 1300 | +2.23e+00 | 19.063 | 10.4076 | **1.091** | 190 |
| 1399 | — | — | — | — | died, cell 98 at p = −68.7 |

The key pins at **7.7334 — the table minimum — from step 1**, so the disk applies
the line's *maximum* PR, 26.885 against a design 21.846. Whether a cell survives
is then just how far its design point sits from that maximum: f = 0.15 (design
PR 25.129) takes a 7% over-pressure, survives, and walks back; f = 0.50 takes 23%
and tears itself apart. That is also why the deaths spread inward *from the choke
side* and why PR does not order them — Nc 1.000 f = 0.15 at PR 25.129 holds while
f = 0.35 at PR 23.123 dies.

**The cause is a measurement error, not dynamics.** Reading the exit station on
the *exact seeded design field*, before a single step:

| Nc | PR | ECMF design | ECMF read | ratio |
| --- | --- | --- | --- | --- |
| 0.700 | 3.95 | 9.2125 | 8.7881 | **0.954** |
| 0.950 | 15.88 | 9.4545 | 7.2623 | 0.768 |
| 0.975 | 16.59 | 10.4206 | 7.9130 | 0.759 |
| 1.000 | 21.85 | 9.2191 | 6.3436 | 0.688 |
| 1.000 | 25.13 | 8.1793 | 5.2836 | **0.646** |

**4.6% low at PR 4, 35% low at PR 25.** The table is only ~38% wide, so beyond
PR ~16 the very first reading falls off the bottom of it.

This reframes §3.30–§3.34 rather than merely extending them. The closure has
never read ECMF accurately; at low pressure ratio the error is small enough that
the restoring feedback absorbs it, which is why 261/315 cells hold and why every
converged answer is still right to 1e−07. Above PR ≈ 16 the error exceeds what
the map can absorb and there is nothing left to correct against.

**Not yet identified: why the reading is low.** Two candidates, neither
confirmed. The station reads the face between the last forced cell and the next,
and with `n_smear` = 7 the last forced cell holds the state after 6.5 of 7 source
shares, so a Roe flux built from the two straddling cells need not carry the full
post-disk jump. Against that, `exit_offset` 1, 2 and 3 give identical answers at
PR 3.36 — if it were simply proximity to the smear, moving out should have moved
it. Resolving this is the next step, and it is worth more than any of the
downstream symptoms.

### 3.34 Family A was two mechanisms wearing the same clothes

§3.33's sweep leaves the two β-end columns failing on both high-pressure maps,
mirror-signed and growing smoothly with PR over four decades — 17 cells that look
like one phenomenon. They are two, and only locating each line's **PR peak**
separates them.

**Moving the design point inward from each end.** `HighPqPCompr` Nc 0.950:

| f | 0.00 | 0.01 | 0.02 | 0.05 | 0.10 | 0.20 |
| --- | --- | --- | --- | --- | --- | --- |
| surge end | **−7.54e−02** | +4.14e−08 | +4.23e−08 | +2.03e−08 | +4.40e−08 | +1.35e−08 |

| f | 1.00 | 0.99 | 0.98 | 0.95 | 0.90 | 0.80 |
| --- | --- | --- | --- | --- | --- | --- |
| choke end | **+2.08e−03** | +1.34e−07 | +2.22e−07 | +4.04e−08 | +1.90e−07 | +8.01e−08 |

A **step, not a slope**: ten of ten interior points hold at 1e−08 to 2e−07, at
PR 13.2 to 19.1, on a line that scored 0/7 before §3.32. Only the exact endpoints
fail, and they fail because the design ECMF sits at margin **0.00000** from the
table end — checked directly against the tabulated range.

**Mechanism 1 — off-table clamp.** The map holds no information past its ends and
§3.8 measures extrapolation as the worst error source on these maps, so clamping
is the correct *action*. But a clamped lookup leaves the operating point pinned
with no restoring force outward, and the run converges to the edge of the data
rather than to an answer. `ECMFMap.off_table` and `EcmfCompressor.off_table` now
count it; **a converged run with a non-zero count is not to be trusted.**

**Mechanism 2 — the surge branch, and the model is right.** The same scan on
`TwoStgRadialCompr` Nc 1.000 does *not* recover 1% inside, and the reason is that
this line's PR peak is not at the end:

| line | PR peak at | so f = 0 is |
| --- | --- | --- |
| `HighPqPCompr` Nc 0.950 | f = **0.0000** | at the peak *and* at the table end |
| `TwoStgRadialCompr` Nc 1.000 | f = **0.0948** | **past** the peak, on the rising branch |

| f | PR | against peak 15.981 | result |
| --- | --- | --- | --- |
| 0.00 | 15.841 | past peak | −3.24e−02 |
| 0.02 | 15.841 | past peak | −3.41e−02 |
| 0.05 | 15.914 | past peak | **DIED** |
| 0.10 | 15.981 | *at* the peak | **DIED** |
| **0.20** | 15.689 | 1.8% below | **+2.54e−08 HELD** |

Everything on or beyond the peak fails; the first genuinely stable point holds at
2.5e−08. A positively-sloped branch is what surge *is*, and no throttle holds a
machine there either.

**This closes §3.27's oldest open item.** That section recorded two
`TwoStgRadialCompr` cells with *falling* but nearly flat slopes (−0.014,
−0.0062) that failed anyway, and could not explain them. On Nc 1.000 the PR at
f = 0.15 is 15.8934 against a peak of 15.9808 — **0.5% below peak**. The branch
is falling, but so flat that it carries no restoring force. The flatness was the
answer, and §3.27's own numbers said so.

**Methodological note.** The two mechanisms are indistinguishable in the sweep
table — same column, same sign, same PR scaling — because both maps happen to put
their difficult region at f ≈ 0. Grouping failures by *where they appear* rather
than by *what the map is doing there* produced a single "Family A" that did not
exist. The discriminator was one cheap map computation: locate `argmax(PR)` per
speed line.

**Consequence for the sweep grid.** Testing at f = 0.0 and f = 1.0 exactly means
testing the boundary of the data, and on some lines it also means testing the
statically unstable branch. Both are legitimate robustness probes, but neither is
an operating point, and scoring them alongside interior cells understates the
closure. A grid that reported them separately — and that placed the surge-side
probe relative to each line's own PR peak rather than at a fixed f — would
measure what it intends to.

### 3.33 The filters cannot be made cheaper, and the sweep with both fixes

**Both fixes together, all four maps, same grid as §3.27** — every tabulated
speed line, seven positions including both β ends, `densify` 36, single disk,
201 cells:

| map | cells held | on lines the inverse refuses |
| --- | --- | --- |
| `SubsonicCompressor` | **84/84** | — |
| `TranssonicCompressor` | **63/63** | **35/35** |
| `HighPqPCompr` | 46/70 *(was 21/70)* | 15/35 *(was 0/35)* |
| `TwoStgRadialCompr` | 68/98 | 14/21 |
| **total** | **261/315** | **64/91** |

The two complete maps reproduced their pre-`key_lag` scores *exactly*, which is
what makes the `HighPqPCompr` gain attributable to the fix rather than to run
variation. **PR 26.999 held**, against a previous ceiling of 14.972 (§3.27).

**The filter time constant cannot be reduced.** `dt ≈ 2e−5 s` against
`tau = 1e−2 s` is 500 steps per time constant, which looks like the obvious place
to buy speed. It is not.

| `tau` | control steps | control W | `HighPqPCompr` Nc 0.950, PR 15.883 |
| --- | --- | --- | --- |
| **1e−2** | 9 000 | +4.0038e−07 | **HELD**, 14 500 steps |
| 3e−3 | 9 000 | +3.9954e−07 | died at 1164 |
| 1e−3 | 8 000 | +3.9953e−07 | died at 1122 |
| 3e−4 | 6 500 | +3.9953e−07 | HELD, 35 500 steps |
| 1e−4 | 14 000 | +3.9953e−07 | died at 533 |

Two results, one good and one negative.

**Unit DC gain is now proven to 2e−10 over a hundredfold range.** The converged
answer is invariant to `tau`, so neither filter can be accused of setting the
operating point. That is the property §3.12 claimed and this measures properly.

**But there is no speed on offer.** 1.4× at best, and slower again at 1e−4 —
below ~1e−2 convergence is no longer filter-limited but set by the duct's own
acoustic and convective settling. The earlier expectation of ~10× was wrong, and
the arithmetic that produced it counted only the filter.

**And high pressure ratio becomes erratic**: dies at 3e−3 and 1e−3, survives at
3e−4, dies at 1e−4. Non-monotone in `tau` is marginal stability rather than a
threshold — and that is itself a datum for §3.32's open problem, since it says
those cells sit on ground where small changes flip them.

Keep `inlet_lag = key_lag = 1e-2`. Real speed has to come from the per-step cost
(~200 µs of Python at 201 cells, which the C++ port addresses) or from local time
stepping for steady runs, not from the filters.

### 3.32 The high-PR ceiling was an unfiltered key, not the keying

§3.31 concludes that ECMF keying loses high pressure ratio, and calls it a
capability regression. That conclusion measured a bug of mine, not the closure.

The signature §3.31 was left with — mesh-independent, gain-independent,
PR-dependent — matches a mechanism this project already documents. §3.11–§3.13:
an **unfiltered** input to the disk opens an acoustic feedback loop whose gain
**grows with pressure ratio**. The disk held at PR 1.2 and failed from 1.4 until
the inlet stagnation state was lagged, and lagging `W` as well was needed past
PR 2.0. A filter time constant is physical, so nothing about that shrinks with
the mesh.

`EcmfCompressor` filtered the inlet state through `InletFilter` and then handed
the map a **raw** exit reading. I added an input and did not give it the
treatment §3.12 exists to provide.

Lagging the key, same first-order filter:

| case | no key lag | `key_lag` = 1e−2 |
| --- | --- | --- |
| `HighPqPCompr` Nc 0.850, PR 8.121 | +1.795e−01, never converges | **+5.40e−08 HELD** |
| `TwoStgRadialCompr` Nc 0.900, PR 10.103 | −5.516e−02, never converges | **+1.43e−07 HELD** |
| `HighPqPCompr` Nc 0.950, PR 15.883 | **died at step 387** | **+1.52e−07 HELD** |
| `SubsonicCompressor` Nc 1.000, PR 2.141 | +3.9983e−07 HELD | +4.0038e−07 HELD |

The last row is the one that makes it a fix rather than a knob. A first-order lag
has **unit DC gain**, so it cannot move a converged answer — and it does not, by
5.5e−11. The same argument, and the same test, §3.12 used for the inlet lag.
`key_lag` 1e−1 gives the same answers as 1e−2 to within 6e−08 on every case, over
a tenfold range.

**PR 15.883 held to 1.52e−07, on a line the inverse cannot invert at all.** That
is above the project's previous best of PR 14.972 (§3.27), and it is
simultaneously rank-deficient and higher pressure ratio than anything held
before.

So the two problems were never one problem. Rank deficiency needed ECMF keying;
the pressure-ratio ceiling needed the key filtered. Neither fixes the other, and
§3.31's tables measure the second defect while the first was already solved.

### 3.31 The full sweep: ECMF keying solves rank deficiency completely and loses high pressure ratio *(superseded by §3.32 — its high-PR half measures an unfiltered key)*

§3.30's closure, swept over the same grid as §3.27 — every tabulated speed line,
seven positions including both β ends, `densify` 36, single disk, 201 cells.

| map | PR range | ECMF at t−1 | inlet `Wc` (§3.27) |
| --- | --- | --- | --- |
| `SubsonicCompressor` | 1.1–3.1 | **84/84** | 80/84 |
| `TranssonicCompressor` | 0.9–3.1 | **63/63**, refused lines **35/35** | 20/45; five lines unbuildable |
| `HighPqPCompr` | 3.1–28.9 | **21/70**, refused lines **0/35** | 19/25 on the lines it could build |

**Rank deficiency is solved, completely.** Every line the inverse cannot invert —
five on `TranssonicCompressor`, including Nc 1.144 where `Wc` spans 9.97e−03 over
the whole β range — now holds at *every* position including both β ends. That was
the open problem from §3.26 through §3.29 and it is closed.

**High pressure ratio is not, and is worse than before.** `HighPqPCompr` above
Nc 0.900 is entirely dead: Nc 0.975 and 1.000 and 1.025 are 7/7 DIED, at PR 15 to
29. The inverse closure held **PR 14.972** on this map at Nc 0.925 (§3.27). So
this is a capability *lost*, not merely one not gained.

The boundary is sharp and it is the line's pressure ratio, not its position:

| line PR range | cells held |
| --- | --- |
| ≤ 6.5 | 7/7, every line, all four maps |
| 5–8 | 5–6/7 |
| 6.4–10.2 | 1–3/7 |
| ≥ 8.8 | 0/7 |

**Three explanations tested and refuted.**

*The exit station's discretisation error.* It is real and it grows with PR —
measured against what the disk intends, `p₀₂` is off by −6.6e−04 at PR 2.14,
+2.0e−03 at PR 3.95, +4.0e−03 at PR 6.29, +6.4e−03 at PR 8.12. But refining the
mesh at a failing cell does not help: `HighPqPCompr` Nc 0.850 f 0.5 gives
+2.103e−01 at 201 cells and +2.027e−01 at 401, a 3.6% change. Anything that
shrinks with the mesh is therefore excluded — which also excludes the t−1
transport delay, since `dt` falls with the mesh too.

*The map's loop gain.* §3.9 measured `−dlnPR/dlnECMF` at 0.90–1.10 and I argued
the t−1 read makes it irrelevant. That argument is wrong — a lag turns an
algebraic loop into an explicit iteration `x_{n+1} = f(x_n)`, which still diverges
when `|f′| > 1` — but the gain does not separate the data either:

| line | median gain | PR | result |
| --- | --- | --- | --- |
| `TranssonicCompressor` Nc 1.144 | **1.065** | 2.49 | **7/7** |
| `HighPqPCompr` Nc 0.850 | **1.018** | 8.60 | 1/7 |

A line at gain 1.065 is perfect while one at 1.018 fails. Gain is not the
discriminator; pressure ratio is.

**What is left.** The failure is mesh-independent, gain-independent and
PR-dependent. That is as far as the measurements go, and no fourth hypothesis is
offered here. Note that §3.14 records a limit of the same shape from a different
direction — "the constraint is on the *total* pressure ratio of the duct rather
than the stage" — and §3.16/§3.18 lifted it for the inlet closure via similarity
scaling. `EcmfCompressor` has similarity scaling on by default, so whatever this
is, it is not that.

**Practical consequence.** Neither closure covers the map library. Inlet `Wc`
handles high pressure ratio and refuses 13 of 45 speed lines; ECMF at t−1 handles
every line and dies above PR ≈ 7. Below PR 3 the ECMF closure is flawless over
147 consecutive cells. A component that selects on the map's PR range would cover
everything measured, but that is a workaround, and the high-PR cause should be
found rather than routed around.

### 3.30 Key on ECMF, read it one step behind — and most of §3.26–§3.29 was wrong

Two observations from review, neither of which I had tested:

> "Since we are modelling a Quasi 1D solver with mesh, don't we have access to
> the exit flow rate directly? … maybe we can read from one prior timestep. This
> will introduce a lag but since the simulation is transient and our timesteps
> are not that large, we should be able to model it pretty accurately."

> "The choked lines are not choked in ECMF maps. That's why I tell you to use
> them. And especially if you take the information from one time step prior, the
> loop also doesn't materialize."

Both are right, and together they retire the problem §3.9 opened and §3.26–§3.29
failed to close.

**Why §3.9's objection does not apply across a step.** §3.9 rejected exit-ECMF
keying because the source would read its own output, with measured loop gain
`−dlnPR/dlnECMF` of 0.90–1.10 — above one, where no relaxation converges. That
loop exists *within* a step. Reading the field at the **start** of the step uses a
value produced by the previous step's operating point: there is no algebraic loop
to have a gain.

**And my degeneracy proof was about a different object.** §3.28 argues that
keying on the exit is degenerate because `p₀₂ = PR(β)·p₀₁` identically, so the
residual is identically zero. That is true of ECMF *reconstructed* from the map,
which is the disk agreeing with itself. ECMF *measured from the field* is the
duct's actual response; the two coincide only at the fixed point, and the cells
that fail do not fail at the fixed point — they fail on the way to it.

**Measured, converged mass-flow error, single disk, 201 cells:**

| case | inlet `Wc` (§3.28) | ECMF at t−1 |
| --- | --- | --- |
| `TranssonicCompressor` Nc 1.144 f 0.50 | *no inverse exists* | **+9.43e−08** |
| `TranssonicCompressor` Nc 1.000 f 0.50 | −1.497e−01, 17 825 clamps | **+1.40e−07** |
| `HighPqPCompr` Nc 0.700 f 0.85 | −1.896e−02, 36 846 clamps | **+2.69e−07** |
| `SubsonicCompressor` Nc 1.200 f 0.50 | +1.760e−02 | **+5.40e−08** |
| `SubsonicCompressor` Nc 1.000 f 0.15 | −1.76e−11 | −8.39e−08 |
| `SubsonicCompressor` Nc 1.000 f 0.50 | +6.72e−12 | +4.00e−07 |

Nc 1.144 is the line §3.26 built its whole argument on — `Wc` spans 9.97e−03
across the entire β range while `PR` spans 4.94e−01. It now holds to 9.4e−08.
**Zero clamped steps everywhere**, against tens of thousands.

**The relaxation was never needed.** With the read at t−1, a plain lookup
`β = ECMF_map⁻¹(ECMF(t−1))` gives *bit-identical* answers to the relaxed form on
every case — +5.538e−06 and +1.342e−06 to four figures, same β to 1e−07. The
relaxation converges 1.8–3× faster and is therefore an accelerator, not a
stabiliser, which is the opposite of why §3.28 built it. `gain`, `tau`, the
Newton step, the secant and the clamp logic are all machinery for a problem the
t−1 read removes.

**Station placement is a delay limit, not a bias.** `exit_offset` 1, 2 and 3 give
identical answers to five digits; 4 and 8 do not converge, with clamping
appearing at 8. Transport from disk to station enters the t−1 path, and beyond
~3 cells it destabilises the loop. Offset 1 reads the disk's own exit face.

**The residual is the map's β resolution, and it converges.** 5.54e−06, 1.57e−06
and 2.69e−07 at `densify` 9, 18 and 36 — observed order **2.18**, and on
`SubsonicCompressor` the error passes through zero, so it is a convergent
discretisation rather than a bias. It is invariant to flow mesh (201/401/801
identical to five digits), station offset, smear width, and β dynamics. I
reported it as a property of exit keying; it is not.

**β leaves the runtime.** ECMF is monotonic in β on **135 of 135** speed lines
across all twelve supplied compressor and fan maps, so `β ↔ ECMF` is a bijection
and `PR(ECMF, Nc)` is single-valued. `ECMFMap` re-tabulates at build time and
`EcmfCompressor` does one table read per step with no operating-point state at
all. The order matters: densify in `(β, Nc)` **first**, convert after (D13).

That is not free, and the cost is between speed lines, where fixed-β crossing is
no longer available:

| worst, mid-interval | Sub ×9 | Sub ×36 | Trans ×9 | Trans ×36 |
| --- | --- | --- | --- | --- |
| `PR` | 4.42e−05 | 2.80e−06 | 1.20e−04 | 7.53e−06 |
| `CW` | 2.60e−04 | 1.84e−05 | 2.13e−03 | 1.50e−04 |

~15× for a 4× refinement, so second order, worst at the lowest speeds where the
supplied lines are furthest apart. On a tabulated line the two paths are the same
map exactly.

**Turbines want a different key.** ECMF is monotonic on only **7 of 22** turbine
speed lines, while `PR` is monotonic on **22 of 22**. `SingleStgTurbine` has a
`Wc` span of 7.6–9.6% across its whole β range — §3.26's rank deficiency in a
different component — with `Wc` monotonic on 0/5 lines and `PR` on 5/5. So
`ECMFMap` refuses a turbine at build time with the reason, and turbines get their
own key. Untested question: compressor ECMF contains `W`, which is measured
independently of the source, whereas a turbine keyed on pure `PR` has no
independently measured component. Whether the t−1 read carries it there is
measurable and should be measured, not predicted.

**What this supersedes.**

* §3.26's conclusion — "the operating point must be either downstream-informed or
  a dynamical state" — is half right. Downstream-informed, yes. A dynamical
  state, no.
* §3.28's degeneracy argument holds only for reconstructed ECMF, and its
  ECMF-slope scaling, secant and clamping are all unnecessary.
* §3.29 is wrong twice. Its stated mechanism — that the grip on β is
  `∂Wc_map/∂β`, so a flat line has no isolated root — drops the dominant term:
  `dR/dβ ∝ ∂W_duct/∂β − ∂W_map/∂β`, and the first term is large precisely because
  `PR` varies strongly, which is the same fact that makes ECMF well conditioned.
  The root exists and is well posed on a vertical line. Its span correlation is
  real but the mechanism attached to it is not.

**Open — and §3.31 measures how open.** The full sweep says this closure is
perfect below PR ≈ 3 and dead above PR ≈ 7, losing capability the inlet closure
had. `HighPqPCompr` Nc 0.925 f 0.50 (PR ≈ 13) dies at step 964, and
`TwoStgRadialCompr` Nc 0.600 f 0.85 settles at 2.12e−06, just outside the gate.
Both are lines the inverse could not run at all, so neither is a regression — but
neither is finished either. The full four-map sweep with this closure has not
been run; the numbers above are eight cells, not 315.

### 3.29 What is left is the flat speed lines — the same rank deficiency, weakened

> **Superseded by §3.30, and its mechanism is wrong.** The span correlation
> measured here is real, but the explanation attached to it is not: `dR/dβ` is
> proportional to `∂W_duct/∂β − ∂W_map/∂β` and this section counts only the
> second term. The first is large, so the root exists and is well posed even on a
> vertical line. Keying on ECMF one step behind holds every case below.

The full-map sweep with §3.28's closure runs every tabulated speed line of all
four maps — including the 13 the inverse refuses — at seven positions each,
β = 0 and β = 1 among them. The failures that remain are not scattered. They sit
in one place, and the place identifies the mechanism.

**What predicts failure is how flat the speed line is, not where you sit on it.**
Binning the interior cells by the line's own `Wc` span:

| line's `Wc` span | interior cells | failures | rate |
| --- | --- | --- | --- |
| **< 25%** | 45 | 18 | **40%** |
| ≥ 25% | 90 | 3 | **3.3%** |

and the ordering is close to monotone in the span: the four tightest lines
(`TranssonicCompressor` Nc 1.000 at 4.5%, Nc 0.952 at 7.5%, `SubsonicCompressor`
Nc 1.200 at 8.0%, `HighPqPCompr` Nc 0.750 at 10.0%) carry 12 of the 21 interior
failures between them, while every line above 25% span is clean except three.
Two of those three are the `TwoStgRadialCompr` f = 0.15 cells §3.27 already
attributes to genuine surge.

**This is §3.26's rank deficiency, weakened but not gone.** The residual's grip
on β is `∂Wc_map/∂β`, and the span is exactly that quantity integrated along the
line. So the thing that stops the *inverse* existing is the same thing that stops
the *relaxation* converging — the closure was never going to escape it entirely,
only soften it. What §3.28 buys is that a flat line can now be **run** at all,
and that its endpoints hold; what it does not buy is the interior of the very
flattest lines.

*Corrected from an earlier draft.* This section first claimed "every failure is
in the last 3% of the `Wc` range", generalised from the nine cells then
available. With 175 cells swept that is wrong in both directions: 37 cells above
position 0.95 hold, and three failures sit nowhere near choke — including
`HighPqPCompr` Nc 0.800 at position **0.000**, which is the surge end. Position
along the line gives an elevenfold concentration (27% above 0.95 against 2.4%
below) but no threshold; the span does better and has a mechanism behind it.

**Where both closures fail, the inverse misses by less.** On
`SubsonicCompressor` Nc 1.200, the inverse's near-choke failures are 9.3e−04 and
1.1e−03; the residual closure's are 1.4e−01 and 1.0e−01, and it additionally
fails at f = 0.50 where the inverse holds. The inverse pins β stiffly to the
measured flow, so its failures are near misses; the relaxation lets β wander, so
its failures are excursions. That is a genuine regression in the near-choke band,
bought in exchange for the 13 speed lines that previously could not be run at
all. Both trades are on the table and neither is free.

**The choke-end column holds, and that is still the diagnosis.** At *exactly*
the choke end — β = 0, which is `f` = 1.0 in the sweep grid; see the convention
note in §3.37, this sentence said "β = 1" until it was checked — the point holds
on every map, 37 such cells, no exceptions — `HighPqPCompr` PR 3.149, `TwoStgRadialCompr`
PR 1.759, `TranssonicCompressor` PR 1.441. The same duct, the same pressure
ratio, the same mesh, 0.5% away in flow, fails. The only difference is that at
the end of the line β is pinned by the clamp and has no freedom, and 0.5% inside
it does.

So the remaining failure is in the β dynamics, not in the flow solver, not in the
similarity scaling, and not in the pressure ratio. Nothing about the Euler side
of the problem changes across that boundary.

**And it is the physics of §3.26, arriving from the other side.** That section's
own physical statement is that on a choked line the inlet state carries no
information about position along it. In the residual formulation that reads
`∂Wc_map/∂β → 0`: the residual barely responds to β, so β is nearly *unconstrained*
rather than wrongly constrained. The closure is not computing the wrong answer
there; it is being asked to determine a quantity the measurement cannot pin down.
That is why clamping at the end of the line — removing the freedom entirely —
is the configuration that works.

**Two hypotheses tested and refuted, both by measurement.**

*The step is too small.* On `HighPqPCompr` Nc 0.700 f = 0.85 the step-to-step
difference quotient gives `∂R/∂β = 0.091` against the map slope's 0.60, so the
Newton step looked 6.6× too conservative. Replacing the map slope with that
measured secant made it **13× worse** — W error from −1.90e−02 to +2.42e−01, and
clamps from 36 846 to 51 165, reproduced twice. The reason is that a one-timestep
difference quotient measures the *frozen-flow* sensitivity, before the duct has
responded; that is smaller than the settled sensitivity, so using it as a Newton
denominator systematically over-steps. `c = 1` is a better estimate of the
*steady* duct constant than the instantaneous measurement is. The secant is kept
behind `secant=False` as a recorded negative result.

*The step rate is wrong.* Neither direction helps, which is what rules out
tuning altogether. On the same case:

| β rate | W error | clamps |
| --- | --- | --- |
| `gain` 0.03 | −1.389e−01 | 43 175 |
| `gain` 0.1 | +3.314e−01 | 52 245 |
| `gain` 0.3 | +1.736e−01 | 53 095 |
| **`gain` 1.0** | **−1.896e−02** | **36 846** |
| secant, ≈6.6× faster | +2.423e−01 | 51 165 |

The default is the best of the five and every other is 7–17× worse. More telling
than the ranking is that the slow end is **not monotone** — 0.3, 0.1 and 0.03
give +1.74e−01, +3.31e−01 and −1.39e−01. A converging process ordered by rate
would not do that. None of them converge; the reported number is wherever the run
happened to be at step 119 999. A step-size problem has a step size that fixes
it. This does not, because there is no isolated root to step toward.

Scope: four rates on one cell, `HighPqPCompr` Nc 0.700 f = 0.85. Enough to rule
out rate as *the* explanation there, not enough to claim it for every cell.

*The trace shows a growing oscillation.* It does not. A 2000-step trace of the
same case looked like one; per-step output shows a well-formed update reducing
`R` monotonically, `−1.396e−02 → −6.5e−03` over 39 steps. The coarse samples were
aliasing a slow drift. Recorded because the wrong reading survived two rounds of
reasoning before the finer measurement killed it.


**What would actually close it: the back pressure.** One piece of genuinely
independent information is available near choke and is not being used. `p_back`
is a *boundary condition*, not the disk's output, and together with the
upstream-measured `W` it fixes the total pressure the system demands at the disk
exit — computable without reference to anything the source injected. Then

```
PR_map(β) = PR_demanded(W, p_back)
```

is well posed exactly where `Wc_map(β)` degenerates, because PR is the quantity
that varies along a choked line. This is *not* §3.9's circular closure: that one
read the field downstream of the disk, which is the source's own output, and is
degenerate for the reason §3.28 gives. Reading a boundary condition closes no
loop.

The cost is interface, not physics: a component would need to know its
downstream network, which is what component matching is, and which the engine
work will need anyway. Not built — recorded as the indicated direction.

### 3.28 β as a relaxation on the residual — the map's inverse was never needed

> **Superseded by §3.30.** The central move — stop inverting — was right. The
> degeneracy argument against exit keying applies only to ECMF *reconstructed*
> from `PR(β)`, not to ECMF measured from the field one step behind, and the
> ECMF-slope scaling, the secant and the clamping are all machinery for a
> problem that read removes.

§3.26 left the closure blocked: keying on inlet `Wc` refuses 13 of 45 tabulated
speed lines because on a choked line the whole β range compresses into ~1% of
`Wc`. Both replacements proposed there turned out to be wrong, and the fix is
smaller than either.

**The downstream-error closure is degenerate, not merely awkward.** §3.26's
second candidate was `τ dβ/dt = K·(p₀₂ measured − p₀₂ predicted at β)`. It cannot
work: `p₀₂` is *what this source injects*. In a constant-area duct the scheme
realises the injected total pressure to within its own dissipation, so
`PR_meas ≡ PR(β)` for **every** β and the residual is identically zero. There is
nothing to solve. This is worse than the algebraic loop §3.9 rejected — that one
at least had a root.

**Nothing needed the inverse in the first place.** The steady operating point is
where two curves in `(W, PR)` cross: the map's speed line, falling, and the duct
with a fixed back pressure, rising. A *vertical* line still crosses a rising one
transversally — the intersection is perfectly well posed. Only the algorithm was
ill-posed, because inverting `Wc(β)` asks a question the intersection never asks.

So β becomes a state relaxed on the residual, and nothing is ever inverted:

```
τ dβ/dt = K · [Wc_meas/Wc_map(β) − 1] / (∂lnECMF/∂β)
```

**Which slope goes in the denominator is the entire design.** The step is a
damped Newton step on the residual, so the denominator wants to be `∂R/∂β`.
Writing it out, with `c = ∂lnW/∂lnPR` of the duct — positive, O(1):

```
∂R/∂β ≈ c·∂lnPR/∂β − ∂lnWc/∂β
```

Take `c = 1` and that is exactly `−∂lnECMF/∂β`. ECMF is not one candidate among
three; it is what the coupled residual's derivative *reduces to*. Measured over
all 45 tabulated lines of the four compressor maps:

| slope | reverses sign | floor over all 45 lines |
| --- | --- | --- |
| `∂lnWc/∂β` | on every refused line | **0** — that is what "refused" means |
| `∂lnPR/∂β` | on **20 of 45**, at the surge peak | **0** |
| `∂lnECMF/∂β` | never | **0.225** |

I built the pressure-slope version first, on the reasoning that a compressor
always makes pressure. It is wrong past the surge peak, where PR falls again
toward lower flow: the sign flips and β is driven the wrong way. It killed
`TwoStgRadialCompr` Nc 0.600 at f = 0.15 and left f = 0.85 1.5e−01 out — both
points the inverse holds. The ECMF slope has no such point on any supplied map,
which is the property `q1d.maps` already documents as the reason ECMF exists.

**This is not the exit-ECMF closure §3.9 rejected.** ECMF enters as a *derivative
of the tabulated map at the current β*, never as a measurement. The measured
quantity is still the upstream `Wc`, so §3.14's property survives untouched: the
disk does not read its own output. A slope of a table closes no loop.

**It is the same closure, solved differently.** At steady state `dβ/dt = 0` forces
`Wc_meas = Wc_map(β)` — the identical equation `InletFlowCompressor` inverts. So
the two must agree wherever both can run, and they do. `SubsonicCompressor`
Nc 1.0:

| position | converged β | design β | error |
| --- | --- | --- | --- |
| f = 0.15 | 0.785800 | 0.785800 | −3.7e−09 |
| f = 0.50 | 0.390300 | 0.390300 | +9.6e−13 |
| f = 0.85 | 0.101600 | 0.101600 | −1.2e−10 |
| f = 0.50, seeded β − 0.2 | 0.390300 | 0.390300 | +1.2e−10 |
| f = 0.50, seeded β + 0.2 | 0.390300 | 0.390300 | −1.4e−10 |

The seeded rows matter more than the first three. Without them the table only
shows the closure is inert where it was put; with them it shows the fixed point
*attracts* from 0.2 away in either direction.

### 3.27 The full-map sweep, and surge as a correct failure

225 operating points: four maps, every tabulated speed line, positions 0.15 to
0.85 along each line, plus the **β = 0 and β = 1 ends** — choke and surge
respectively (§3.37; this read "surge and choke" until the convention was
measured) — which an earlier grid had avoided and which are the points worth
having a map for.

**The extremes are not a problem.** `SubsonicCompressor` holds **24/24** at
β = 0 and β = 1, every speed line, up to PR 3.082 at surge (β = 1) and 2.133 at
choke (β = 0).
`TranssonicCompressor` holds **8/8** on its four invertible lines. Wherever the
closure can be evaluated at all, the tabulated ends behave like the interior.

**Pressure ratios far beyond anything earlier.** `HighPqPCompr` holds **PR
14.972** at Nc 0.925, and `TwoStgRadialCompr` holds **PR 11.254** at Nc 0.900 —
single disk, `n_smear` = 7.

**Most surge-side failures are the model being right.** On
`TwoStgRadialCompr` the failures cluster at the surge end (f = 0.15) while
f ≥ 0.35 holds. Taking the static-stability slope there:

| Nc | f | PR | normalised `dp_exit/dW` | sweep |
| --- | --- | --- | --- | --- |
| 0.85 | 0.15 | 9.308 | **+0.0564** rising | died |
| 0.90 | 0.15 | 11.254 | **+0.0895** rising | died |
| 0.90 | 0.35 | 10.830 | −1.2647 falling | held |
| 0.90 | 0.65 | 9.350 | −6.1753 falling | held |

A rising branch is statically unstable — that is what surge *is*, and a real
machine cannot sit there either. **Diverging is the correct answer**, and it is
reassuring that the model finds it in the right place rather than everywhere or
nowhere.

Two surge-side failures are *not* explained this way: Nc 0.65 and 0.80 at
f = 0.15 have falling slopes and should hold. But those slopes are **−0.014 and
−0.0062**, two orders of magnitude flatter than the −1.26 and −6.18 of the
healthy points — marginal ground where the restoring force has nearly vanished.
Whether the model or the margin is at fault there is not yet established.

**A surge-line test is now possible and should exist.** The map's own
`dp_exit/dW = 0` contour is a predicted surge line; the solver's divergence
boundary is a measured one. Comparing them is a physics validation the project
has never had, and it is nearly free given both pieces exist.

### 3.26 Inlet-Wc keying is not ill-conditioned near choke, it is rank-deficient

> **Diagnosis correct, conclusion half wrong — see §3.30.** Inlet `Wc` really is
> rank-deficient on 13 of 45 speed lines. But "the operating point must be either
> downstream-informed or a dynamical state" is only half right: downstream-
> informed yes, a dynamical state no.

The full-map sweep answers a question that had been carried on assertion. Keying
the closure on **inlet** corrected flow refuses **29% of the tabulated speed
lines** across the four maps, and always the top ones:

| map | lines | invertible | refused | refused speeds |
| --- | --- | --- | --- | --- |
| `SubsonicCompressor` | 12 | 12 | 0 | — |
| `TranssonicCompressor` | 9 | 4 | **5** | 0.880 … 1.144 |
| `HighPqPCompr` | 10 | 5 | **5** | 0.900 … 1.025 |
| `TwoStgRadialCompr` | 14 | 11 | **3** | 1.030 … 1.100 |
| **total** | 45 | 32 | **13 (29%)** | |

`SubsonicCompressor` refuses nothing, which is why sweeping it alone gave a
flattering 58/60 and why the problem stayed hidden.

**The refusal is the closure, not the design.** `design_from_map` returns
perfectly good operating points on every refused line — PR 1.820, 1.991, 2.112,
2.159, 2.318 on `TranssonicCompressor`. What raises is
`inlet_closure_is_invertible`.

**And it is rank deficiency, not conditioning.** On a choked speed line the whole
β range compresses into about **1% of inlet `Wc`** while `PR` spans nearly 50%:

| map | Nc | `Wc` spread | ECMF spread | `PR` spread |
| --- | --- | --- | --- | --- |
| `TranssonicCompressor` | 1.144 | **9.97e−03** | 4.88e−01 | 4.94e−01 |
| `HighPqPCompr` | 1.025 | **7.61e−03** | 3.31e−01 | 3.65e−01 |
| `TwoStgRadialCompr` | 1.100 | 4.47e−02 | 4.69e−01 | 4.39e−01 |

The speed line is *vertical in `Wc`* — that is what choked means. Inverting
`Wc → β` there is not merely delicate; there is no inverse, and a truncation
error of order 1% destroys it entirely.

**"Key on ECMF instead" does not work if the ECMF is built from the inlet.**
Since `ECMF_map(β) ≡ Wc_map(β)·√τ(β)/PR(β)`, forming ECMF from a measured inlet
`Wc` gives

```
g(β) = Wc_meas·√τ(β)/PR(β) − ECMF_map(β) = [Wc_meas − Wc_map(β)]·√τ(β)/PR(β)
```

— the *same* root, merely rescaled. ECMF only helps when it is formed from the
**exit** state, which is the circular closure §3.9 rejected.

**The physical statement.** On a choked line the inlet state carries no
information about position along it; the back pressure sets the operating point.
So no instantaneous inlet-only closure can work there, for any numerical method.
The operating point must be either downstream-informed or a dynamical state.

**`UnsteadyMappedCompressor` does not fix this.** Its state equation is
`τ dβ/dt = β_map(Wc) − β`, which still needs the same inversion; it addresses
acoustic amplification (§3.10), not invertibility.

**Two candidate closures, neither built** — *both superseded by §3.28, and the
second is wrong.*

1. *Exit ECMF solved simultaneously.* §3.9 rejected it because the algebraic loop
   has measured gain `−dlnPR/dlnECMF` of 0.90 to 1.10, and above one a
   fixed-point iteration diverges for every relaxation factor. That is an
   argument against fixed-point iteration, not against the closure — and
   `ImplicitStepper` (§3.17) now exists to solve the state and the closure
   together. Not needed: §3.28 gets there without implicit machinery.
2. *β driven by a downstream error.* `τ dβ/dt = K·(p₀₂ measured − p₀₂ predicted
   at β)`. **This cannot work.** `p₀₂` is what the source injects, so
   `PR_meas ≡ PR(β)` for every β to within the scheme's dissipation: the residual
   is identically zero and there is nothing to solve. Worse than the loop §3.9
   rejected — that one at least had a root. See §3.28.

**The premise this section rests on is also wrong, and that is the way out.** It
concludes that "the operating point must be either downstream-informed or a
dynamical state" *because* `Wc → β` cannot be inverted. True — but nothing needs
that inverse. §3.28 keeps the measurement upstream, inverts nothing, and works on
a vertical line.

### 3.25 Re-measuring the open failures — most of them were already fixed

§3.19 left two failures open on `HPC01`: a near-choke limit cycle, and startup
deaths at Nc ≥ 1.0. Both were measured with the closure as it stood then —
cell-centred station sampling, the similarity reference read from the *forced*
cell, `n_smear` = 21, `sample_offset` = 12. All four have since changed (§3.20,
§3.23), so the numbers were stale. Re-measuring cost minutes; fixing what is
already fixed would have cost days.

Current closure, `n_smear` = 1, `sample_offset` = 2, residual convergence:

| Nc | position in `Wc` | PR | §3.19 said | now |
| --- | --- | --- | --- | --- |
| 0.80 | 0.582 | 5.040 | held 1.4e−10 | HELD −1.202e−11 |
| 0.80 | **0.869** | 4.436 | cycle 1.6e−02 | **HELD −2.491e−10** |
| 0.90 | 0.710 | 7.424 | cycle 3.6e−03 | **HELD −3.646e−11** |
| 0.90 | **0.925** | 5.991 | β clamped at 0 | **HELD −1.510e−10** |

**The near-choke limit cycle is gone**, including at 92.5% along the speed line
where β used to clamp. So §3.19's diagnosis — that this was the ICMF
conditioning problem of §3.5 surfacing — was explaining a symptom that the
closure fix removed. The conditioning statement about ICMF remains true; it was
not what caused the cycle.

**The Nc ≥ 1.0 "startup basin" was a smear-width problem.** With `n_smear` = 1
those cases die at step 1–2 rather than 191–791, which is the seed being
non-physical on contact rather than a dynamic instability — and no wonder, since
it asks *one node* for PR 9.45 against a demonstrated single-node capability of
PR 5.04. Widening it:

| Nc | PR | `n_smear` 1 | `n_smear` 7 | `n_smear` 21 |
| --- | --- | --- | --- | --- |
| 1.00 | 9.454 | died 2 | **HELD −4.736e−12** | not conv. −2.18e−02 |
| 1.05 | 10.162 | died 1 | died 12984 | died 335 |

**`n_smear` is non-monotonic**, which §3.20 already saw at Nc 0.8 (1 holds, 3
dies, 21 holds) and is still unexplained. The practical form: **~7 cells is the
sweet spot at high pressure ratio**, and both extremes are worse. One node is
right up to about PR 5; beyond that it wants a few.

**What is actually still open:** Nc 1.05, PR 10.162 — the top speed line — fails
at every smear width tried. That is now the only surviving failure from §3.19.

### 3.19 The residual limit cycle is the ICMF conditioning problem, not the closure

> **Superseded by §3.25.** The cycle described here does not reproduce with
> the current closure — Nc 0.8 at 87% along the speed line now holds to
> −2.5e−10, and Nc 0.9 at 92.5%, where β clamped, holds to −1.5e−10. The
> measurements below are sound and the ICMF conditioning statement is still
> true; the diagnosis linking the two was explaining a symptom that §3.18
> and §3.20 removed.

With the scaling in place, `HPC01` at Nc 0.7 (PR 3.104) converges to a mean
offset of **+6.27e−11** with 1.02e−07 peak-to-peak. Nc 0.8 does not: it settles
into a sustained oscillation of ±8e−3 in mass flow. Since the linearisation
there is −38.96, the cycle is nonlinear, and four hypotheses were tested and
killed before the right one:

| hypothesis | test | result |
| --- | --- | --- |
| filter dynamics set the period | sweep `τ` | period 2.95e−3 s, **0.29 τ** — not the filter |
| duct resonance, exit reflection | absorbing outlet, σ = 0.2, 0.5 | 1.65e−02, 1.69e−02 against 1.63e−02 — no effect |
| duct resonance, inlet reflection | absorbing inlet, σ = 0.2 | 1.63e−02 — **no effect at all** |
| constant area starves the exit | contract to `M2` = 0.30 (`A2/A1` = 0.404) | 1.94e−02 — slightly *worse* |

The period does match `2 × 0.5 / 340 = 2.9e−3 s`, the acoustic round trip of the
half-duct between inlet and disk, but making either end absorbing changes
nothing, so the resonance is a symptom and not the driver.

**What it actually is: the design point sat next to choke.** Every rig here
designs at the midpoint of the speed line's *ECMF* range, and ECMF is a strongly
nonlinear function of `Wc`, so the two midpoints are not the same point. At
Nc 0.8 the line spans `Wc` ∈ [17.39, 23.60] and mid-ECMF lands at `Wc` = 22.785
— **87% of the way to choke, with 3.6% of margin**. The closure inverts *inlet*
`Wc`, so that is the margin that governs. Moving along the line, at Nc 0.8:

| position in `Wc` | PR | peak-to-peak in W | mean offset |
| --- | --- | --- | --- |
| 0.582 | **5.040** | **6.23e−10** | **+1.40e−10** |
| 0.764 | 4.762 | 9.00e−05 | −4.10e−08 |
| 0.87 (mid-ECMF) | 4.436 | 1.63e−02 | −3.56e−04 |

**PR 5.04 holds to 1.4e−10.** The degradation is smooth and monotone in
proximity to choke, which identifies it as the conditioning problem §3.5 already
recorded from the other side: ICMF compresses the whole β range into 3–10% of
mass flow above 72% speed, so near choke the closure inverts a nearly vertical
curve. `InletFlowCompressor` keys on inlet `Wc` deliberately — keying on *exit*
corrected flow closes an algebraic loop through the source's own output, with
measured gain 0.90–1.10 — so this is the price of that choice, and it is only
paid near the choke end.

The same effect fully saturated is the `beta` = 0.0000 clamping seen at Nc 0.9,
with `PR` pinned at the line's end value 4.6161. That is "suspect 1" returning,
for a reason now understood.

**Where the ceiling is now.** Sweeping the top speeds at a comfortable margin:

| Nc | PR | position in `Wc` | result |
| --- | --- | --- | --- |
| 0.7 | 3.104 | 0.5 | **held, 6.27e−11** |
| 0.8 | **5.040** | 0.582 | **held, 1.40e−10** |
| 0.8 | 4.762 | 0.764 | 9.0e−05 cycle |
| 0.9 | 7.424 | 0.710 | survives, 3.6e−03 cycle |
| 0.9 | 6.771 | 0.853 | dies at step 2023 |
| 1.0 | 9.454 | 0.689 | dies at step 791 |
| 1.05 | 10.162 | **0.319** | dies at step 189 |

Two *different* remaining failures, and they must not be conflated:

1. **Near-choke cycling**, above roughly 0.75 of the `Wc` range — the
   conditioning problem above. It degrades smoothly and predictably.
2. **A startup failure at Nc ≥ 1.0**, which is not that: at Nc 1.05 the design
   point sits at 0.319 of the range, nowhere near choke, and the run still dies
   in 189 steps. The linearised design state at Nc 1.0 is **−19.1**, i.e.
   stable, so this is a *basin* problem — the flux-integrated seed is an exact
   steady state of the continuous equations but not of the discrete ones, and at
   PR 9+ the startup transient is large enough to leave the basin.

For (2) the tool is pseudo-transient continuation with `ImplicitStepper`, which
is half-built: Newton stops converging near dt = 5.6e−2 s because the
preconditioner is diagonal (§3.17). The continuation *ramp* is not the tool —
bringing the source up from zero against a back pressure sized for full PR blows
up at steps 286 and 334 where starting at full strength survives.

**Also open.** A first attempt to put `LocalLevelFilter`'s states into the
eigenvalue problem disagreed with its own validated frozen limit (+37.0 against
−38.96 at Nc 0.8) and is therefore **wrong**; its numbers are recorded nowhere,
and the rig needs fixing before the level filter's dynamics can be analysed.

**The continuation ramp is the wrong tool for the startup.** Bringing the source
up from zero against a back pressure sized for full PR blows up at steps 286 and
334 where starting at full strength survives — the weak-source state is nowhere
near the design point and has to travel back. Pseudo-transient continuation with
the implicit stepper is the better route and is only half-built: Newton stops
converging near dt = 5.6e−2 s (~2600 CFL steps) because the preconditioner is
diagonal, which is exactly the limitation §3.17 records.

### 3.20 The whole pressure ratio in one node — shift the reference upstream

§3.18 left the scaling needing a wide smear, because it divided by the pressure
of the cell it was forcing. That is self-referential: a force raises that cell's
own pressure through its own momentum equation, and the loop has gain of order
`(PR−1)/(2·n_smear)` — about 0.1 for a 21-cell smear at PR 5, and above one for
a single cell. Since every high-PR result used `n_smear = 21`, the per-node
pressure rise was only ~1.08, short of the 1.1–1.6 that would let a map vary
without changing the node count at runtime.

The self-interaction is not intrinsic. What the scaling needs is the local
pressure *level*, and it can be read one cell upstream, where the diagonal term
does not exist. Largest eigenvalue of the linearised design state, `HPC01`:

| `n_smear` | per node | source off | reference = forced cell | **reference = one cell up** |
| --- | --- | --- | --- | --- |
| **1** | **5.040** | +86.7 | +88.3 | **−92.3** |
| 3 | 1.715 | +86.7 | +8.1 | **−76.5** |
| 7 | 1.260 | +86.8 | −14.3 | **−77.8** |
| 21 | 1.080 | +87.3 | −34.6 | **−64.8** |

(at Nc 0.8, design at 0.2 of the ECMF range). The shift is not a refinement: it
is what makes a single-cell disk work at all, and it roughly doubles the margin
at wide smears as well. It is still exactly inert at any steady state, so the
operating point does not move and every §3.18 measurement stands.

Two dead ends worth recording, because both look reasonable:

* **Referencing a fixed cell just upstream of the whole smear** (one scalar for
  all forced cells) works beautifully at `n_smear` 1 and 3 and is catastrophic
  at 7 and 21 — +7609 and +1559. Of course: a single upstream cell is a poor
  proxy for the level twenty cells further on, inside a region that has already
  compressed the flow several-fold. The reference must be local *per cell*.
* **Shifting by two cells** rather than one is worse where it matters, +6914 at
  `n_smear = 1`, so the shift wants to be the smallest that removes the diagonal.

**Confirmed nonlinearly**, `HPC01`, 201 cells, 40,000 steps, mass-flow offset and
peak-to-peak over the last half:

| Nc | `n_smear` | PR | per node | reference = forced cell | **reference = one cell up** |
| --- | --- | --- | --- | --- | --- |
| 0.6 | 1 | 2.257 | **2.257** | held, +6.62e−12 | held, **+6.66e−12**, pk-pk **0** |
| 0.6 | 21 | 2.257 | 1.040 | held, +1.008e−10 | held, +1.008e−10, pk-pk 9.6e−15 |
| 0.8 | 1 | 5.040 | **5.040** | **dies at step 513** | held, **+8.58e−12**, pk-pk **0** |
| 0.8 | 3 | 5.040 | 1.715 | dies at 1813 | **dies at 3247** |

So the whole of PR 5.040 goes into **one cell** and sits there to 8.6e−12 with a
peak-to-peak of exactly zero, where the previous form died in 513 steps. That
clears the 1.1–1.6 per-node target by a wide margin and removes the reason the
node count would have had to track the map at runtime.

**The `n_smear = 3` row is an open anomaly and is not explained.** Its linearised
eigenvalue is −76.5, comfortably stable, yet it dies at step 3247 — so it is a
basin failure like the Nc ≥ 1.0 startups of §3.19, not a stability one, and the
behaviour is *non-monotonic* in smear width (1 holds, 3 dies, 21 holds). Until
that is understood, `n_smear` should be taken as 1 or ≳ 7 rather than anything
between, and the reason recorded as "unexplained", not "tuned".

### 3.21 Chaining stages, and the annulus taper that goes with them

With a single node able to carry PR 5.040 (§3.20), staging is the route to engine
overall pressure ratio. §3.14's conclusion — "PR 2.0 splits, PR 4.0 fails at
every split" — was measured on the injection §3.16 has since shown to be wrong,
so the question was open again.

**Constant-area staging works but is physically absurd.** Twelve stages at 1.328
each, OPR 30, `n_smear = 1`: the old injection dies at step 5302, the fixed one
survives. But the exit Mach is **0.023** — the machine discharging into what is
nearly a plenum, because at OPR 30 the density is up thirtyfold and a constant
annulus cannot absorb it. No compressor is built that way; the annulus tapers,
and the more so the higher the pressure ratio.

**With a taper, sized to hold the axial Mach at its inlet value:**

| OPR | stages | stage PR | exit M | `A_n/A_0` |
| --- | --- | --- | --- | --- |
| 4.0 | 4 | 1.414 | 0.450 | 0.311 |
| 14.0 | 8 | 1.391 | 0.450 | 0.108 |
| 30.0 | 12 | 1.328 | 0.450 | **0.057** |

Each disk sits on a constant-area flat — required by §4.5, and also how a real
machine is laid out — with the contraction taken in the gap between rows.

**The taper exposed a first-order error, and the cause is the area profile's
kinks.** A single stage in a tapered duct converged (peak-to-peak 1e−7) to a
*steady* offset of −4.7e−04, where the same disk in a constant-area duct holds
to −1.1e−07. Decomposed:

| case | result |
| --- | --- |
| constant area + disk | held, −1.11e−07 |
| taper, **no disk** | second order: −5.9e−07, −1.4e−07, +1.7e−10 at 301/601/1201 |
| taper + disk | **first order**: −9.37e−04, −4.70e−04, −2.32e−04 |

So neither ingredient is at fault alone; the *interaction* is first order. Two
hypotheses were tested and killed first — the lead-in distance (moving the first
disk downstream changes nothing, −4.58e−04 → −4.92e−04, though it did reveal
that the rig had been sampling the first interior cell) and the width of the
constant-area flat around the disk (3, 10, 25 cells: −4.70e−04, −4.76e−04,
−4.89e−04).

The cause is that the area was piecewise linear, so `dA/dx` is **discontinuous at
every knot** and the limiter clips at those corners. Making the profile C¹ — a
smoothstep blend, zero slope at both ends of each segment, which is also how an
annulus line is actually drawn — gives at 601 cells:

```
    piecewise linear   −4.70e−04
    C1 smoothstep      −2.70e−06        ~170x
```

**Rule for the engine model: every annulus transition wants a C¹ area
distribution.** A corner in the wall line is cheap to draw and expensive to
compute next to a source term.

**Where the cell-centred probe is and is not trustworthy.** `DiskState.W` is
`ρ·u·a_cell` at the sampling cell — a **cell-centred** product, where the scheme
conserves the **face flux**. The 12-stage trace shows the two parting company: it
plateaus at `W_first` = +9.03e−05 while `W_last` = +3.00e−05, and at a steady
state the mass flow is identical at every station, so that threefold difference
is measurement, not physics.

Two separate effects, and it matters which is which:

* **Area gradient.** Measured below: exact in constant area, O(Δx²) and growing
  with `dA/dx` in a contraction.
* **The disk's own cell.** A cell carrying a momentum source has a cell-averaged
  `ρuA` far from the neighbouring flux, because the average represents a jump
  inside the cell. Here those twelve cells read ~+0.158, and 12/601 × 0.158 =
  +3.15e−03 against a measured whole-duct mean of +3.246e−03 — so a naive
  duct-wide average of `cv[1]` is meaningless, and an early attempt at one here
  was.

**So which stations can be believed?** The *first* disk samples at cell 15 and
the inlet flat runs to cell 24, so it sits in constant area and its reading is
faithful. The *last* disk samples at cell 510, inside a taper, and its reading is
not. **The +9.03e−05 plateau is therefore a real mass-flow error**, and the thing
that was artefactual is only the `first`-vs-`last` discrepancy that prompted this
check.

Measured directly, on a contracting duct with **no source**, comparing the spread
of cell-centred `ρuA` across the duct:

| `A₁/A₀` | 301 cells | 601 cells | order |
| --- | --- | --- | --- |
| 1.000 | **7.68e−15** | **9.60e−15** | exact |
| 0.900 | 4.69e−06 | 1.19e−06 | 2 |
| 0.788 | 1.07e−05 | 2.72e−06 | 2 |

In **constant area the two agree to machine precision**; a contraction separates
them by an O(Δx²) amount that grows with the area gradient. Twelve contractions
down to `A/A₀` = 0.057 therefore explain a ~6e−05 spread between stations
without any physical error being present.

Consequences, in order of importance:

* **Nothing in §3.16–§3.20 is affected.** Every one of those measurements is in a
  constant-area duct, which is the 7.68e−15 row. The single-node PR 5.040 hold at
  8.58e−12 stands.
* **Staged OPR 30 converges to +9.0e−05**, monotonically and with no cycle. That
  is a real number, taken at a station in constant area, and it is 90× outside
  the 1e−6 gate — so the train is *stable and slightly wrong*, which is a
  different and far better problem than §3.14 had.
* **The sampling station belongs inside the disk's constant-area flat.** Phase 3
  chose `sample_offset = 12` by measuring the upstream numerical boundary layer
  in a *constant-area* duct, where the gradient effect is identically zero, so
  that choice does not transfer to a tapered machine. In the rig above the flat
  is 7 cells wide and every station after the first sits 12 cells upstream — in
  the taper. That also invalidates the earlier pad = 3/10/25 test, which was
  reading a biased sample.
* **It is not only a diagnostic problem.** The map lookup keys on the sampled
  `W`, so a station in a gradient region displaces the *operating point itself*.
  Invisible here only because the staged rig uses a constant-PR map, where `PR`
  and `Δh₀` do not depend on `Wc`. With a real map it would bite, and that is the
  first thing to fix before staging a mapped machine.

**The fix, and it is forward-compatible.** `Solver.mass_flux_at(cell)` returns
the conserved mass flow from the numerical face fluxes, averaged over the cell's
two bounding faces, and every disk now samples that instead of forming
`ρ·u·a_cell`. It is free: `residual` computes the fluxes *before* calling the
source, so they are cached and waiting.

`Solver.face_fluxes` already existed and its docstring already warned that
cell-centred `ρuA` "is only uniform to `O(dx²)` wherever the area has curvature".
The right probe was there, documented, and the closure used the wrong one anyway
— which is how two rounds got spent arguing with a plateau.

**This must not be justified by "the mass equation has no source".** That is true
today and will stop being true: interstage bleed and turbine cooling air both
remove mass, and then the flux legitimately *steps* along the duct. The property
actually relied on is narrower and survives: the face flux is the conserved
quantity, so it stays the right measure of what passes a station whatever sits
upstream. Two consequences to bank before bleed lands:

* the two-face average is the flow *through* the cell — with a mass source in
  that cell it is the mid-cell value, which is what a station wants;
* the **spread of mass flux along the duct stops being a convergence measure**
  once bleed exists, since it is then legitimately non-zero. `residual_norm` is
  the source-agnostic measure and should take over that job.

#### What the mesh is actually spent on

Worth stating plainly, because the ratio is not what one would guess. The
twelve-stage rig, 1 m duct, 601 cells, `dx` = 1.664 mm:

| region | cells |
| --- | --- |
| inlet duct before stage 1 | 27 |
| **compressor** (`n_smear` = 1 × 12 stages) | **12** |
| duct between and around stages | 484 |
| exit duct after stage 12 | 78 |

**12 cells of compressor against 589 of duct — 2% versus 98%.** Each disk also
sits on a 7-cell constant-area flat (`2·pad + n_smear`) with 44 cells of gap
between stages. The single-disk cases are `ncell` = 201 with the disk at cell 90:
21 cells of compressor at `n_smear` = 21, and **one** in the PR 5.040 case.

For an engine with many components that ratio should invert, and the resolution
should follow the gradients rather than the component count.

**The staged train converges, at better than second order.** Measured properly at
last: OPR 5.478 in six stages, tapered annulus, `n_smear` = 1, every physical
quantity held fixed across meshes — geometry *and* the sampling station — and
each case run until `residual_norm` falls below 1e−11 rather than for a fixed
number of steps.

| `ncell` | `sample_offset` | mass-flow error | flux non-uniformity | steps |
| --- | --- | --- | --- | --- |
| 401 | 8 | +7.230e−05 | 1.6e−09 | 28 000 |
| 601 | 12 | +1.863e−05 | 2.7e−09 | 38 000 |
| 901 | 18 | +4.482e−06 | 4.4e−09 | 51 500 |

Error ratios 3.88 and 4.16 across mesh ratios of 1.499, i.e. observed order
**3.35** and **3.52** — consistent between both pairs and better than the second
order expected. The mass flux is uniform to ~1e−09 throughout, so mass is
conserved exactly and the discrepancy was always with the *analytic design
chain*, never with conservation. Extrapolating puts the 1e−6 gate at roughly
1400 cells.

So staging is sound, and every earlier staged number was instrument or confound
rather than physics.

**The offset is measured in cells, and that is a trap for mesh studies.** §5
Phase 3 chose `sample_offset` in *cells* because the numerical boundary layer is
a stencil artefact and does not shrink under refinement — correct for choosing a
default. But holding it at 12 cells while refining moves the station physically
(3.0% of the duct at 401, 2.0% at 601), and `Fx` and `SWx` are computed from what
it reads, so the source differs on each mesh. The sweep above scales the offset
with the mesh precisely to avoid that; the version that did not came out
non-monotonic (−9.378e−06 then +1.863e−05) and meant nothing.

**Four confounded experiments in a row, and what they have in common.** Worth
recording as a methodological note, because the same mistake wore four different
costumes and cost more time than any of the physics:

1. *pad = 3/10/25 at fixed gap.* Widening the flat also shortens the taper, so
   the area gradient steepens. Read as "flat width does not matter".
2. *pad = 3/14/20 at gap 60.* Same coupling, sharper: pad 20 leaves 19 cells of
   taper instead of 57, roughly tripling `dA/dx`. Read as "putting the station
   in the flat makes it worse" (+6.77e−05, −7.89e−05, −3.07e−04).
3. *ncell = 401/601/901 at fixed step count.* A fixed number of steps is not a
   fixed physical time — the finest mesh got two-thirds the physical time of the
   middle one and was the least converged, which its peak-to-peak showed
   plainly. Read as "the error does not fall with refinement".

4. *ncell = 401/601/901 at `sample_offset` = 12 cells.* Holding the offset in
   cells — which is right for choosing a default, since the boundary layer is a
   stencil artefact — moves the station *physically* under refinement, from 3.0%
   of the duct to 2.0%. `Fx` and `SWx` are computed from what it reads, so the
   source is a different source on each mesh. Came out non-monotonic
   (−9.378e−06 then +1.863e−05) and meant nothing.

In each case two things varied and one was reported. The fixes: hold the taper
length fixed and move only the station; converge on `residual_norm` rather than
a step count — also the measure that survives bleed (§3.22), so it is the right
instrument twice over; and scale *every* physical length with the mesh, the
sampling offset included, when the question is whether the model converges.

With all four controlled, the answer came out clean and positive at order ~3.4.
The lesson is not "be careful" — it is that a sweep is only a sweep if exactly
one thing moves, and the cheapest check is to write down what else changed when
the swept parameter did. **Flat width remains untested**; the two attempts at it
both varied the taper gradient as well.

### 3.24 Engine-scale pressure ratio on a real map

Every staged result before this used `ConstantCompressorMap` — fixed `PR`, fixed
`η`, no dependence on `Wc`. Deliberate, since it isolates the scheme, but it
means the map's restoring slope, the load-bearing physics of §3.16, was absent
from all of it, and a constant-PR stage can neither surge nor choke.

**Why each stage needs its own scaled map.** With one map and a fixed `W`, stage
`k` sees `Wc = W√θ_k/δ_k`. Across a stage of PR ≈ 2, `δ` doubles while `√θ` rises
about 12%, so `Wc` roughly **halves every stage** and leaves the tabulated range
after two. That is not an artefact — it is why a real multistage compressor's
stages are different machines, each sized to its own inlet corrected flow. The
standard device is stage stacking: the same map shape scaled per stage, which
also puts every stage at the same relative point, i.e. a repeating-stage machine.

`SubsonicCompressor`, Nc 1.0, 35% along the speed line, `n_smear` = 1,
`sample_offset` = 2, 601 cells, converged on `residual_norm` < 1e−11:

| stages | stage PR | OPR | exit `T₀` | mass-flow error |
| --- | --- | --- | --- | --- |
| 1 | 2.2177 | 2.218 | 368.7 K | −1.648e−06 |
| 2 | 2.2177 | 4.918 | 471.9 K | −2.454e−06 |
| 4 | 2.2177 | 24.188 | 772.7 K | −1.697e−06 |
| 6 | 2.2177 | **118.959** | 1265.3 K | **+4.415e−07** held |

Mass flux uniform to ~4e−09 throughout, zero reverse-flow events, and — the part
that matters — **the error does not grow with stage count**. Six stages are no
worse than one, so the per-stage errors are not accumulating.

That is past the OPR ≈ 80 that motivated the whole exercise, with real maps and
every stage finding its own operating point.

**Off-design: the stages find each other.** The table above is a *design-point*
result — every stage was seeded where its scaled map was built for, so nothing
had to move and a constant-PR map would have looked identical. Throttling is what
separates them: raise the back pressure and the machine must find a new
equilibrium, with less flow, lower `Wc` at every stage, and therefore more `PR`
from each along its own speed line, until the delivered pressure matches the
load. That is component matching, and the analytic chain gives the answer
independently — walk the stages for a candidate `W`, and solve for the `W` whose
delivered exit static equals the imposed back pressure.

Four stages, `SubsonicCompressor`:

| `p_back`/design | `W` analytic | `W` solver | solver vs analytic |
| --- | --- | --- | --- |
| 1.00 | 51.0188 | 51.0187 | −1.697e−06 |
| 1.02 | 50.9206 | 50.9205 | −2.108e−06 |
| 1.05 | 50.7332 | 50.7331 | −3.177e−06 |
| 0.98 | 51.1014 | 51.1013 | −1.269e−06 |
| 0.95 | 51.2029 | 51.2029 | −8.709e−07 |

**1–3e−06 across the range.** The solver finds the same equilibrium the map
algebra does, which is the property the whole source-term formulation exists to
deliver and which a constant-PR stage cannot test at all.

**The throttle range is narrow, and that is physical.** Above about +4% flow the
*exit* chokes: more flow puts every stage at higher `Wc`, so each delivers less
pressure, so exit `p₀` falls while `W` rises and the exit flow function climbs
faster than linearly. A fixed exit area therefore limits the range. A real engine
does not have one — the turbine nozzle downstream is itself an area — so a wider
excursion needs the next component, not a fix here. It also means the analytic
matching function is one-sided about the design point and cannot be bracketed by
bisection from a fixed interval; it is scanned instead.

**What this is not.** Perfect gas, γ = 1.4 and constant `cp`, with an exit at
1265 K where real-gas effects are large — Phase 4 will move these numbers and
they should not be quoted as physical until it does. One speed line, one map.
Stage stacking puts every stage at the same relative map point, where a real
machine has stage-to-stage variation. No surge-line or choke-line behaviour has
been exercised.

### 3.23 The upstream "boundary layer" was the probe, and it cost 10 cells a row

Phase 3 measured the disk perturbing the field *upstream* of itself, decaying
~10× every two to three cells, and set `sample_offset = 12` to clear it. That
default has been paid on every blade row since, and for an engine with twenty
rows it is a large fraction of the mesh.

It was never in the field. It is the error of averaging a sharp profile over a
cell. The scheme conserves **face fluxes**, and inverting those back to a state
(`Solver.station_state_at`, via `analytic.state_from_flux`) gives a station
reading that is exact right next to the disk. Converged single disk, `p01` error
against the known inlet value:

| `sample_offset` | cell-centred | from the flux |
| --- | --- | --- |
| 1 | −1.933e−04 | **−8.9e−12** |
| 3 | −1.074e−05 | **−9.1e−12** |
| 8 | +1.475e−08 | −9.4e−12 |
| 12 | +3.623e−11 | −9.6e−12 |
| 24 | −1.146e−11 | −1.2e−11 |

Flat at every standoff. It holds at PR 1.2, 1.6 and 2.0, with and without a
downstream taper — five configurations; the sixth (PR 2.0, constant area) is bad
for *both* probes and worsens with offset, which is the signature of an
unconverged field rather than a probe difference.

On the six-stage train the offset drops from 12 to 2 for nothing: −1.273e−05
against −1.383e−05 at 601 cells.

**Careful with the arithmetic, though.** Ten cells a row is real but it is not
the whole budget — the inter-stage `gap` dominates. What the mesh actually buys,
six stages, offset 2, same 1 m duct:

| cells/stage | mass-flow error |
| --- | --- |
| 25 | −2.004e−03 |
| 33 | −6.441e−04 |
| 50 | −1.427e−04 |
| 66 | −5.118e−05 |
| 150 | −4.482e−06 |

So **33–50 cells per blade row** is the sensible engine-model range, giving
1e−4 to 6e−4 — ample for performance work — and a twenty-row engine lands near
800 cells rather than the 3000 the earlier 150-cells-per-stage figure implied.

**The flux probe needs a fallback and this is not padding.** During a startup
transient an intermediate face flux can correspond to *no* physical state, and
the inversion raises on a negative discriminant where the cell-centred product
just returns a number. `station_state_at` catches that and falls back to the
cell-centred reading, which costs nothing at convergence — where the flux value
is the one used, and is exact.

`tests/test_compressor.py` now pins the opposite property to the one it was
written for: every standoff from 1 to 12 clears the gate and they agree with
each other to 1e−9, where the old test asserted a monotone decay.

### 3.22 Designing for bleed and cooling flows, before they arrive

Interstage bleed and turbine cooling air are planned, and they make ``q[0] ≠ 0``.
Several things written down elsewhere in this document quietly assume otherwise,
so the assumptions are collected here while they are cheap to fix.

**What breaks.** Any diagnostic phrased as "the mass flux should be uniform".
With bleed it steps by exactly the mass removed, which is correct behaviour, so
uniformity stops being a convergence measure. `residual_norm` is the
source-agnostic replacement and is already available.

**What survives.** `mass_flux_at` — the face flux is the conserved quantity
regardless of what sources exist, so it remains the right reading of what passes
a station. The justification must be stated that way and not as "the mass
equation has no source", which is the narrower claim that happens to be true
today.

**What has to be built.** A bleed port removes mass at the local stagnation
state, so it is a source in **all three** equations at once,

```
q_bleed = −ṁ_b · [1, u, h₀] / L_port
```

and not merely a mass sink. Removing mass without the matching momentum and
energy would inject spurious momentum and heat — a mistake that would look like
a compressor efficiency error and be hunted for in the wrong place.

**What the similarity rule becomes.** §3.18's rule is "lag the dimensionless
operating point, never the dimensional scale factors". For a compressor the
dimensionless quantities are `PR` and `Δh₀/θ`. For a bleed port the analogous
invariant is a **flow coefficient** — the fraction of passing mass extracted, or
a discharge coefficient against the port pressure ratio — not a mass flow in
kg/s. Tabulating `ṁ_b` in kg/s would reproduce exactly the fixed-rate error that
§3.16 traced through six weeks of symptoms.

**What the reference chain must carry.** The analytic station chain used to size
these rigs assumes `W` is the same at every station. With bleed it must track
`W_{k+1} = W_k − ṁ_{b,k}`, and the annulus areas, back pressure and seeded
profile all follow from that. A rig that forgets it will show a mass-flow error
of exactly the bleed fraction and invite a search for a numerical cause.

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

The original split (5a ICMF + β, 5b ECMF held in reserve) is superseded. D9
retired ICMF as a *cycle-code* lookup coordinate on conditioning grounds, D13
settled the interpolation scheme, and §3.9 showed the solver's closure is a
separate question from either.

- **Phase 5a — ingestion, ECMF, densification.** ✅ complete. `q1d.maps`:
  workbook loading, corrected work and ECMF derived at load, PCHIP refinement of
  the (β, Nc) grid before the conversion (D13), clamp-not-extrapolate in speed.
- **Phase 5b — feasibility-first duct design.** ✅ complete. `q1d.design`:
  a map point plus an inlet Mach number fixes the area, both stagnation states
  and the back pressure; infeasibility is reported with its reason before a
  solver exists. `analytic.state_from_flux` and `design.steady_profile` build
  the discrete steady profile of a smeared disk by integrating the flux balance
  (§3.9a).
- **Phase 5c — a closure the solver can march.** *In progress, and the thing
  that blocks the phase.* Both candidate coordinates fail at the top of these
  maps for opposite reasons — exit corrected flow is circular with loop gain
  above one, inlet corrected flow is non-circular but has `|dlnFx/dlnW|` up to
  29 against a 12-cell sampling delay (§3.9b, §3.9c). Below ~90% speed on
  `SubsonicCompressor` both gains are under one.
- **Phase 5d — the deliverable.** Hold a real map steady at several speeds, show
  the ECMF map with the converged solver points on it, then a throttle sweep
  along one speed line into the surge and choke ends.
- This is where `∂PR/∂Φ₁ < 0` finally supplies aerodynamic stiffness and surge
  dynamics become structurally representable — impossible with a constant-PR
  map, which has zero stiffness by construction. §3.9 is the first evidence
  that the stiffness is *real*: it is what raises the loop gain from 0.10 to
  3.29 at design speed.

**Gate:** speedline reproduction, throttle sweep, and stable operation across
the map.

**A consistency debt this phase creates.** `maps.load_beta_map` derives
corrected work with a *perfect-gas* `cp·T_ref·(PR^k − 1)/η`, while every
invariance figure quoted in §3.6–§3.8 was measured with Cantera. Both are
self-consistent today, because the solver is perfect-gas too. They stop being
so the moment Phase 4 lands, so the 0.13%/0.29% invariance must not be claimed
for the real-gas solver until the map calibration uses the same gas model.

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
| D9 | ~~ICMF + β-lines first; ECMF held in reserve for the choke line~~ **Revised: ECMF from the start** | Measured on the supplied maps (§3.5): ICMF is non-monotonic in β at Nc ≥ 0.72 on the compressor and at *every* speed on the turbine, spanning as little as 3.1%. ECMF is monotonic everywhere and spans 72–163%. ECMF is not a choke-line remedy, it is the only invertible coordinate on most of the map | 2026-07-25 |
| D12 | Maps store **corrected work** `Δh₀/T₀₁` (equivalently the work coefficient `Δh₀/N²`), not η, not τ, not loss | η is singular where PR → 1 with work input; τ is invariant only for a perfect gas and drifts 16.8% at 1200 K. Loss and corrected work are *equally* invariant — at fixed η they are proportional, `loss = Δh₀(1−η)` — so directness decides: `SWx = W·Δh₀` needs corrected work immediately, whereas loss requires solving the isentropic entropy inversion on every evaluation just to subtract it back out (§3.6). Author's proposal, second alternative | 2026-07-25 |
| D11 | R = 287.058 J/kg/K (Cantera, dry air) for all new work; `AIR_LEGACY` keeps 287.1429 | Author's decision. The legacy value is derived from cp=1005, γ=1.4 and is 0.03% off. `AIR_LEGACY` is retained solely so the Phase 1–3 regression numbers stay reproducible; the two must never be mixed in one calculation | 2026-07-25 |
| D10 | Steady-state acceptance only, extended with a hold test | No transient reference data exists yet. A solver that reaches the analytically known point and holds it is accepted for now; revisited when transient data becomes available | 2026-07-25 |
| D13 | Maps are **PCHIP-densified 9× in (β, Nc) at load, before the ECMF conversion**; the runtime lookup stays linear. Speed is **clamped, never extrapolated** | Author's proposal, measured in §3.7. β is not a physical dimension but it is the correspondence label across speed lines, so refining in (β, Nc) and converting afterwards beats refining in ECMF. Dense-plus-linear reproduces direct PCHIP to 0.07% at linear cost on a regular grid, which also ports to C++ as a flat array. Cubic *extrapolation* beyond the tabulated speeds is worse than linear (7.7% vs 4.9%), hence the clamp | 2026-07-25 |

Dependencies: `numpy`, `scipy`, `pandas`, `openpyxl`, `pytest`; `matplotlib`
optional. `scipy` earned its place with D13 (`PchipInterpolator`); root-finding
inside the solver stays hand-written (`rtsafe`) and does not use it.

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

> **The pressure-ratio limit is resolved.** §3.16 identifies the cause and §3.18
> the fix. The three entries immediately below are kept because their
> *measurements* are sound and several are still the best record of what was
> tried, but their conclusions are superseded. Current status: **PR 14.972 held**
> on `HighPqPCompr` at Nc 0.925 and **PR 11.254** on `TwoStgRadialCompr` at
> Nc 0.900, single disk (§3.27). §3.19's near-choke limit cycle and its Nc ≥ 1.0
> startup deaths were **re-measured and are gone** (§3.25).
>
> **The closure limit is resolved too — see §3.30.** §3.26 recorded that
> inlet-`Wc` keying refuses 29% of the tabulated speed lines. The answer is to
> key on **ECMF read one step behind**: no inverse, no operating-point state, and
> §3.9's loop-gain objection does not apply across a step. `TranssonicCompressor`
> Nc 1.144 — the canonical rank-deficient line, `Wc` spanning 9.97e−03 — holds to
> **9.4e−08**. Wherever this document says a speed line "cannot be closed on",
> read it as a property of `InletFlowCompressor`, not of the solver.

- ~~**Fixed to PR 2.0, not beyond.**~~ **Superseded by §3.18.** The inlet lag on
  `(T₀₁, p₀₁, W)` (§3.12, §3.13) holds the operating point to ~2e-10 up to
  PR 2.0. Above that the lag family stops working entirely — no `τ` up to 3 s
  holds PR 2.5. The reason, found later, is that the lag freezes the dimensional
  *level* along with the operating point, turning the disk into a fixed-force,
  fixed-heat-rate device whose steady state is a repeller from PR 2.26.
- ~~**Above PR ≈ 3 the cause is still unknown**~~ (§3.15). **Answered in §3.16.**
  The eliminations recorded there are all correct and all irrelevant: the
  equilibrium itself was linearly unstable, so no numerical ingredient could have
  been the culprit. The "remaining untried option" named there — an implicit
  solve of the disk's operating point — was also wrong, for the same reason.
- **Staging is the right model but does not lift the limit** (§3.14). PR 2.0
  split four ways holds to 3.6e-10, but PR 4.0 fails at every split, so the
  constraint is on the *total* pressure ratio of the duct rather than the stage.
  The evidence points at the train-plus-boundaries, not the disk; the inlet
  boundary condition under a large adverse pressure rise is the next suspect.
- **The actuator-disk source fails above PR ≈ 1.3, independently of any map**
  (§3.10). A constant-PR disk with a closed-form reference holds to 1.9e-10 at
  PR 1.2 and diverges at 1.4, blowing up by 2.2. Everything below about the map
  is downstream of this and much of it is moot; it is retained because the
  measurements are sound even where the diagnosis was not.
- ~~**Phase 5 stops at `Z = |dFx/dW|/c ≈ 1`.**~~ **Superseded by §3.10.** Below it a real map is held to
  1e-11 in mass flow with 1e-12 drift; above it the operating point is not held
  at any mesh from 201 to 1601 cells, at any smear from 1.3% to 20% of the duct,
  at any standoff from 0.12% to 3%, or with any remedy tried (§3.9). On
  `SubsonicCompressor` that is Nc ≤ 0.672 of 12 speed lines; on
  `TranssonicCompressor`, Nc ≤ 0.528 of 9, with the top five refusing the inlet
  closure outright because `Wc` is not invertible there. ~~**Nothing above those
  speeds should be trusted or reported as working.**~~ **No longer true**: the
  refusal was `InletFlowCompressor` needing an inverse that does not exist, and
  §3.28 removes the need for it. The question is no longer
  "where is the bug" — `Z` is the acoustic-impedance criterion and the disk is a
  wave amplifier above it — but "what unsteady compressor response replaces the
  quasi-steady map". That is the next piece of modelling, not the next debugging
  session.
- **An earlier claim in this document was wrong and is corrected in §3.9.** The
  boundary was reported as `|dlnFx/dlnW| ≈ 1`, generalised from
  `SubsonicCompressor` alone. `TranssonicCompressor` holds at −4.41 and −3.38
  while `SubsonicCompressor` fails at −0.95, so the logarithmic gain does not
  separate the data; `Z` does.
- **The lag is a steady-state device only.** A first-order lag with `τ = 1e-2 s`
  restores the hold at gain 0.95, and its unit DC gain means the converged point
  is exact. But `τ` is the same order as the settling time, so it is *not*
  usable for transient work — which is the project's actual purpose. It is
  recorded as a diagnostic that identified the mechanism, not as a solution.
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
