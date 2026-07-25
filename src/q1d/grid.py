"""Finite-volume grid for the quasi-1D solver.

Layout, for ``n`` interior cells::

    ghost |  0   |  1   | ...  | n-1  | ghost
    ------+------+------+------+------+------
       f0 |      f1     ...        fn |
          0      1      ...      n-1

* ``n + 1`` faces carry ``a_face``; fluxes live here.
* ``n + 2`` cells carry ``a_cell`` and ``vol``, including one ghost at each end.
* Cell-indexed arrays in the solver are therefore length ``n + 2``, with
  interior cells at ``[1:-1]``.

The legacy script built this with a pair of ``np.insert`` calls whose index
arithmetic (``im``, ``im1``, ``ncv``) had to be re-derived at every use site.
Here the shapes are named and validated once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["Grid"]


@dataclass(frozen=True)
class Grid:
    x_face: np.ndarray  # (n+1,) face positions
    a_face: np.ndarray  # (n+1,) area at faces
    x_cell: np.ndarray  # (n,)   cell centres
    dx: np.ndarray  # (n,)   cell widths
    da: np.ndarray  # (n,)   area change across each interior cell
    a_cell: np.ndarray  # (n+2,) cell areas, ghosts included
    vol: np.ndarray  # (n+2,) cell volumes, ghosts included

    def __post_init__(self) -> None:
        n = self.n_interior
        for name, arr, expected in [
            ("x_face", self.x_face, n + 1),
            ("a_face", self.a_face, n + 1),
            ("x_cell", self.x_cell, n),
            ("dx", self.dx, n),
            ("da", self.da, n),
            ("a_cell", self.a_cell, n + 2),
            ("vol", self.vol, n + 2),
        ]:
            if arr.shape != (expected,):
                raise ValueError(f"{name} has shape {arr.shape}, expected {(expected,)}")
        if np.any(self.dx <= 0.0):
            raise ValueError("grid spacing must be strictly positive")
        if np.any(self.a_face <= 0.0):
            raise ValueError("face areas must be strictly positive")
        if np.any(self.a_cell <= 0.0):
            raise ValueError("cell areas must be strictly positive")
        if np.any(self.vol <= 0.0):
            raise ValueError("cell volumes must be strictly positive")
        expected_da = self.a_face[1:] - self.a_face[:-1]
        if not np.allclose(self.da, expected_da, rtol=0.0, atol=0.0):
            raise ValueError("da must equal a_face[1:] - a_face[:-1]")
        if not np.allclose(self.dx, self.x_face[1:] - self.x_face[:-1], rtol=0.0, atol=0.0):
            raise ValueError("dx must equal the spacing of x_face")

    @property
    def n_interior(self) -> int:
        return len(self.dx)

    @property
    def is_constant_area(self) -> bool:
        return bool(np.all(self.da == 0.0))

    @classmethod
    def uniform(
        cls,
        x0: float,
        x1: float,
        n: int,
        area: float | Callable[[np.ndarray], np.ndarray],
    ) -> Grid:
        """Uniformly spaced grid over ``[x0, x1]`` with ``n`` interior cells.

        ``area`` is either a constant or a callable evaluated at face and cell
        positions. Cell areas are taken from the callable at the cell centre
        rather than averaged from the faces; well-balancedness does not depend
        on that choice, because the geometric source uses
        ``da = a_face[1:] - a_face[:-1]`` which telescopes against the pressure
        flux exactly whatever ``a_cell`` is.
        """
        if n < 3:
            raise ValueError(f"need at least 3 interior cells, got {n}")
        x_face = np.linspace(x0, x1, n + 1)
        x_cell = 0.5 * (x_face[1:] + x_face[:-1])
        a_of = (lambda x: np.full_like(x, float(area))) if np.isscalar(area) else area

        a_face = np.asarray(a_of(x_face), dtype=float)
        a_in = np.asarray(a_of(x_cell), dtype=float)
        dx = x_face[1:] - x_face[:-1]

        # ghost cells mirror the adjacent interior cell's geometry
        a_cell = np.concatenate(([a_in[0]], a_in, [a_in[-1]]))
        vol_in = a_in * dx
        vol = np.concatenate(([vol_in[0]], vol_in, [vol_in[-1]]))

        return cls(
            x_face=x_face,
            a_face=a_face,
            x_cell=x_cell,
            dx=dx,
            da=a_face[1:] - a_face[:-1],
            a_cell=a_cell,
            vol=vol,
        )
