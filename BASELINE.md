# Baseline — `legacy/` reference implementation

Phase 0 gate. Measured 2026-07-25. This is the number to beat and the
behaviour to reproduce.

Configuration as shipped: `ncv = 101` (99 interior cells), constant area
`A = 0.1 m²`, `PR = 1.2`, `η_is = 0.9`, `CFL = 2.5`, `iorder = 2`, 5-stage RK,
`tend = 0.5 s`, source injected at interior cell 50 (global cell 51), lookup
sampled at global cell 52.

## Performance

| metric | value |
| --- | --- |
| wall time | **14.77 s** |
| steps | 10,915 |
| ms / step | 1.35 |
| `dt` (converged) | 4.5511e-05 s |

Per-step cost is **flat** through the run (measured at 1.29–1.35 ms from step
250 onward). Cost does not grow with the developing solution.

> **Measurement warning.** An earlier baseline of 386 s was an artefact of the
> harness, which captured the script's 10,915 per-step `print(t)` calls into an
> in-memory buffer. That is 26x the true cost of the solve. Time the solver
> with the print removed or redirected.

## Accuracy

| quantity | analytic | achieved | error |
| --- | --- | --- | --- |
| W [kg/s] | 21.490743 | 21.492878 | **+0.0099 %** |
| p₀₂/p₀₁ | 1.2 | 1.200036 | +0.0030 % |
| T₀₂ [K] | 305.2701 | 305.2639 | −0.0020 % |

**The legacy solver works.** It reaches the closed-form answer to ~1e-4
relative. Several findings in the original review implied it could not reach a
steady operating point at all; it does.

## Converged field structure

The far field is uniform to ~1e-8. Cells 1–45 all read identically:

```
W = 21.492878   p = 75322.91   pt = 101325.00   Tt = 288.1500   M = 0.66491
```

and cells 55–99 all read identically:

```
W = 21.492878   p = 101325.00  pt = 121593.67   Tt = 305.2639   M = 0.51712
```

This confirms the premise behind the Phase 3 tolerance: in the uniform region
the Roe dissipation vanishes identically and stagnation pressure is preserved
to machine precision, so a 1e-8 gate is reachable.

### Where the 1e-4 error comes from

The source lookup samples global cell 52, which is still inside the region
smeared by the single-cell injection:

| cell | role | pt | Tt | W |
| --- | --- | --- | --- | --- |
| 49 | upstream, clean | 101324.49 | 288.1177 | 21.491863 |
| 50 | upstream of disk | 101265.82 | 288.2373 | 21.448644 |
| 51 | **disk cell** | 108406.12 | 294.8823 | 23.081237 |
| 52 | **lookup station** | 121552.73 | 305.3153 | 21.476938 |
| 55 | downstream, clean | 121593.52 | 305.2649 | 21.492780 |

Cell 52 reads `pt` 40.9 Pa below the true downstream value — a 3.4e-4 deficit.
Since `ecmf ∝ 1/pt`, and `dFx/d(ecmf) ≈ −55 N`, this propagates to ~0.3 N of
force error out of 1731 N, which is the observed magnitude.

Sampling upstream, where the field is uniform to 1e-8, removes this error
source entirely. This quantifies PLAN.md §4.2 finding #2.

(Cell-centred `ρuA` is not constant across the smeared region, and need not be.
The discrete steady state requires the *face* mass flux to be constant, which
it is — every uniform cell on both sides reads 21.492878.)

## Convergence behaviour

Inlet velocity `u₁` settles long before the run ends:

| criterion on \|u₁ − u₁_final\|/u₁ | first reached | fraction of run wasted |
| --- | --- | --- |
| < 1e-4 | step 1,716 (t = 0.0814) | 6.4x |
| < 1e-6 | step 2,565 (t = 0.1200) | 4.3x |
| < 1e-8 | step 3,424 (t = 0.1591) | 3.2x |

**Residual-based stopping is worth 3.2x on its own** — more than every
micro-optimisation in Phase 2 combined. `tend = 0.5` is roughly 3x longer than
the solution needs.

## Implications for Phase 2 priorities

Revised against PLAN.md §5, now that the true cost distribution is known:

1. **Convergence-based stopping — 3.2x.** The single largest lever.
2. **Remove the per-step `print`.** Free, and it was hiding the real cost.
3. `entropy_corr` vectorisation and `np.interp` — together ~10 s at the
   original 33,750-evaluation estimate, but only ~0.7 s of the measured 14.77 s
   at this step count. Still worth doing; no longer the headline.
4. CFL limit investigation — a direct linear saving if the true limit is above
   2.5.

The earlier microbenchmarks (PLAN.md §3.4) measured the routines in isolation
and over-weighted them relative to the full solve. They remain correct as
per-call costs and wrong as a share of runtime.
