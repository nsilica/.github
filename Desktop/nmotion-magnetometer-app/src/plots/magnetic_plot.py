"""Gráfico 3D de campo magnético: nube de puntos y esfera ajustada."""

from __future__ import annotations

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from src.plots.base_plot import BasePlot

MAX_POINTS = 3000


class MagneticPlot(BasePlot):
    """Renderiza la nube de puntos del magnetómetro y la esfera ajustada."""

    def __init__(self, width: int = 800, height: int = 600, dpi: int = 100) -> None:
        self._points = {
            "x": deque(maxlen=MAX_POINTS),
            "y": deque(maxlen=MAX_POINTS),
            "z": deque(maxlen=MAX_POINTS),
        }
        self._scatter = None
        self._center_dot = None
        self._sphere_surface = None
        self._center = (0.0, 0.0, 0.0)
        self._radius = 0.0
        self._has_fit = False
        super().__init__(width, height, dpi)

    def _setup(self) -> None:
        self.fig = plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#2A2D34")
        self.fig.patch.set_facecolor("#2A2D34")
        self.ax.set_xlabel("Mx", color="white")
        self.ax.set_ylabel("My", color="white")
        self.ax.set_zlabel("Mz", color="white")
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
        self.ax.set_title("Calibración Hard-Iron", color="white")

    def add_point(self, mx: float, my: float, mz: float) -> None:
        """Añade un punto a la nube."""
        self._points["x"].append(mx)
        self._points["y"].append(my)
        self._points["z"].append(mz)

    def clear(self) -> None:
        """Limpia todos los puntos y el ajuste."""
        self._points["x"].clear()
        self._points["y"].clear()
        self._points["z"].clear()
        self._has_fit = False
        self._center = (0.0, 0.0, 0.0)
        self._radius = 0.0
        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None
        if self._center_dot is not None:
            self._center_dot.remove()
            self._center_dot = None
        if self._sphere_surface is not None:
            self._sphere_surface.remove()
            self._sphere_surface = None

    def fit_sphere(self) -> tuple[float, float, float, float] | None:
        """Ajusta una esfera por mínimos cuadrados y devuelve (cx, cy, cz, r)."""
        n = len(self._points["x"])
        if n < 20:
            return None
        xa = np.array(self._points["x"], dtype=float)
        ya = np.array(self._points["y"], dtype=float)
        za = np.array(self._points["z"], dtype=float)

        cx0, cy0, cz0 = xa.mean(), ya.mean(), za.mean()
        r0 = float(np.mean(np.sqrt((xa - cx0) ** 2 + (ya - cy0) ** 2 + (za - cz0) ** 2)))

        def residuals(p: np.ndarray) -> np.ndarray:
            cx_, cy_, cz_, r_ = p
            return np.sqrt((xa - cx_) ** 2 + (ya - cy_) ** 2 + (za - cz_) ** 2) - r_

        try:
            res = least_squares(residuals, x0=[cx0, cy0, cz0, r0], method="lm", max_nfev=50)
        except Exception:  # noqa: BLE001
            return None

        cx, cy, cz, r = res.x[0], res.x[1], res.x[2], abs(res.x[3])
        self._center = (cx, cy, cz)
        self._radius = r
        self._has_fit = True
        return cx, cy, cz, r

    def update(self) -> None:
        """Redibuja la escena con los puntos actuales y la esfera ajustada."""
        if self.ax is None:
            self._setup()

        n = len(self._points["x"])
        if n == 0:
            return

        xs = np.array(self._points["x"], dtype=float)
        ys = np.array(self._points["y"], dtype=float)
        zs = np.array(self._points["z"], dtype=float)

        if self._scatter is not None:
            self._scatter.remove()

        self._scatter = self.ax.scatter(
            xs, ys, zs,
            c=zs,
            cmap="coolwarm",
            s=8,
            alpha=0.8,
            depthshade=True,
        )

        # Ajustar límites con margen
        all_vals = np.concatenate([xs, ys, zs])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = max((vmax - vmin) * 0.1, 50)
        self.ax.set_xlim(vmin - margin, vmax + margin)
        self.ax.set_ylim(vmin - margin, vmax + margin)
        self.ax.set_zlim(vmin - margin, vmax + margin)

        self.fit_sphere()
        if self._has_fit:
            cx, cy, cz = self._center
            r = self._radius

            if self._sphere_surface is not None:
                self._sphere_surface.remove()
            u = np.linspace(0, 2 * np.pi, 24)
            v = np.linspace(0, np.pi, 16)
            sx = cx + r * np.outer(np.cos(u), np.sin(v))
            sy = cy + r * np.outer(np.sin(u), np.sin(v))
            sz = cz + r * np.outer(np.ones_like(u), np.cos(v))
            self._sphere_surface = self.ax.plot_surface(
                sx, sy, sz,
                alpha=0.08,
                color="#FFFFFF",
                linewidth=0,
                antialiased=False,
            )

            if self._center_dot is not None:
                self._center_dot.remove()
            self._center_dot = self.ax.scatter(
                [cx], [cy], [cz],
                marker="+",
                s=200,
                c="#FF6F00",
                linewidths=2.5,
                depthshade=False,
            )
