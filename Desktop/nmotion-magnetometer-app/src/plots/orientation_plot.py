"""Gráfico 3D de orientación: caja sólida orientada por cuaternión."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.plots.base_plot import BasePlot


# Caja con 3 lados distintos (paralelepípedo): semiejes X, Y, Z
BOX_HALF = (0.17, 0.37, 0.11)
BOX_VERTICES = np.array([
    [-BOX_HALF[0], -BOX_HALF[1], -BOX_HALF[2]], [BOX_HALF[0], -BOX_HALF[1], -BOX_HALF[2]],
    [BOX_HALF[0],  BOX_HALF[1], -BOX_HALF[2]], [-BOX_HALF[0],  BOX_HALF[1], -BOX_HALF[2]],
    [-BOX_HALF[0], -BOX_HALF[1],  BOX_HALF[2]], [BOX_HALF[0], -BOX_HALF[1],  BOX_HALF[2]],
    [BOX_HALF[0],  BOX_HALF[1],  BOX_HALF[2]], [-BOX_HALF[0],  BOX_HALF[1],  BOX_HALF[2]],
])

BOX_FACES = [
    [0, 1, 2, 3],  # Z-
    [4, 5, 6, 7],  # Z+
    [0, 1, 5, 4],  # Y-
    [2, 3, 7, 6],  # Y+
    [0, 3, 7, 4],  # X-
    [1, 2, 6, 5],  # X+
]

FACE_COLORS = [
    (0.20, 0.50, 0.80, 0.70),
    (0.20, 0.50, 0.80, 0.70),
    (0.20, 0.75, 0.45, 0.70),
    (0.20, 0.75, 0.45, 0.70),
    (0.90, 0.40, 0.20, 0.70),
    (0.90, 0.40, 0.20, 0.70),
]


def _quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Cuaternión unitario (w,x,y,z) -> matriz 3x3."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class OrientationPlot(BasePlot):
    """Renderiza una caja 3D orientada."""

    def __init__(self, width: int = 800, height: int = 600, dpi: int = 100) -> None:
        self._poly_collection: Poly3DCollection | None = None
        super().__init__(width, height, dpi)

    def _setup(self) -> None:
        self.fig = plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#2A2D34")
        self.fig.patch.set_facecolor("#2A2D34")
        self.ax.set_xlim(-0.7, 0.7)
        self.ax.set_ylim(-0.7, 0.7)
        self.ax.set_zlim(-0.7, 0.7)
        self.ax.set_xlabel("X", color="white")
        self.ax.set_ylabel("Y", color="white")
        self.ax.set_zlabel("Z", color="white")
        self.ax.tick_params(colors="white")
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor("white")
        self.ax.yaxis.pane.set_edgecolor("white")
        self.ax.zaxis.pane.set_edgecolor("white")
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)

        verts = [BOX_VERTICES[i].tolist() for i in range(8)]
        face_polys = [[verts[i] for i in face] for face in BOX_FACES]
        self._poly_collection = Poly3DCollection(
            face_polys,
            facecolors=FACE_COLORS,
            edgecolors=(1.0, 1.0, 1.0, 0.9),
            linewidths=1.2,
            zsort="average",
        )
        self.ax.add_collection3d(self._poly_collection)

    def update(self, qw: float, qx: float, qy: float, qz: float) -> None:
        """Actualiza la orientación de la caja."""
        R = _quat_to_matrix(qw, qx, qy, qz)
        rotated = (R @ BOX_VERTICES.T).T
        verts = rotated.tolist()
        face_polys = [[verts[i] for i in face] for face in BOX_FACES]
        if self._poly_collection is not None:
            self._poly_collection.set_verts(face_polys)
