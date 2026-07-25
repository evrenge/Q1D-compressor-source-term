# `legacy/` — frozen reference implementation

**These files are frozen. Do not edit them.** They exist so that every later
result can be diffed against the original behaviour. If a rewritten component
disagrees with the reference, the reference is the arbiter of what the original
code did — not of what is correct.

## Contents

| file | role |
| --- | --- |
| `source_map.py` | Offline source-term map generator. Writes `Sources.npy`. |
| `q1d_solver.py` | Quasi-1D Euler solver. Reads `Sources.npy`. |

Run order matters: `source_map.py` must run first and leave `Sources.npy` in
the working directory.

## Provenance

Transcribed from the author's PDF export of the original scripts. The PDF
renderer collapsed newlines, running consecutive statements together on single
lines (`import numpy as npimport matplotlib.pyplot as plt`,
`gamma = 1.4cpgas = 1005.0`), so the statement boundaries were restored by
hand. **No logic, constant, or expression was altered.**

Filenames are assigned here; the originals were not supplied with names.

The docstrings and inline comments appear to have been added by an earlier
annotation pass rather than being the author's own — they are preserved as
found. They are comments only and do not affect behaviour.

### Transcription verification

The reconstruction reproduces values computed independently, before the files
were written, directly from the algebra:

| quantity | independent | `legacy/source_map.py` |
| --- | --- | --- |
| `W_ff` | 21.49074 | 21.490742688 |
| `Fx` at W = 0 | 2026.50 | 2026.5000 |
| `Fx` at W = 23 | 1660.44 | 1660.4446 |
| `SWx` at W = 23 | 395.7 kW | 395731.5694 |
| `ecmf/W` | 0.85773 | 0.857732 |
| `setStatic` iterations at M = 0.9243 | 100 (cap hit) | 100 (cap hit) |

The solver reproduces the analytic steady state to +0.0099% (see
`../BASELINE.md`), which is only possible if the transcription is faithful.

If you ever find a discrepancy against your originals, replace these files and
re-run the baseline — everything downstream is anchored to them.

## Running them

Both scripts open matplotlib figures at the end. Use a non-interactive backend
and a scratch working directory so `Sources.npy` does not land in the repo:

```sh
cd /tmp/scratch
MPLBACKEND=Agg python /path/to/legacy/source_map.py
MPLBACKEND=Agg python /path/to/legacy/q1d_solver.py
```

`q1d_solver.py` prints the simulation time every step (10,915 lines). Capturing
that stream naively can cost far more than the solve itself — the first
baseline measurement was inflated 26x this way. Redirect to `/dev/null` or
strip the `print` when timing.
