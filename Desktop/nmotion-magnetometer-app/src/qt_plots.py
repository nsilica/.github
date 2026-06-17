"""Matplotlib canvases embedded in PySide6 widgets."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares

BOX_HALF = (0.17, 0.37, 0.11)
BOX_VERTICES = np.array(
    [
        [-BOX_HALF[0], -BOX_HALF[1], -BOX_HALF[2]],
        [BOX_HALF[0], -BOX_HALF[1], -BOX_HALF[2]],
        [BOX_HALF[0], BOX_HALF[1], -BOX_HALF[2]],
        [-BOX_HALF[0], BOX_HALF[1], -BOX_HALF[2]],
        [-BOX_HALF[0], -BOX_HALF[1], BOX_HALF[2]],
        [BOX_HALF[0], -BOX_HALF[1], BOX_HALF[2]],
        [BOX_HALF[0], BOX_HALF[1], BOX_HALF[2]],
        [-BOX_HALF[0], BOX_HALF[1], BOX_HALF[2]],
    ]
)
BOX_FACES = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
]
FACE_COLORS = [
    (0.20, 0.50, 0.80, 0.70),
    (0.20, 0.50, 0.80, 0.70),
    (0.20, 0.75, 0.45, 0.70),
    (0.20, 0.75, 0.45, 0.70),
    (0.90, 0.40, 0.20, 0.70),
    (0.90, 0.40, 0.20, 0.70),
]
MAX_MAG_POINTS = 3000
CONVERGENCE_WINDOW = 50
CONVERGENCE_THR = 5.0


def _style_3d_axis(ax, title: str | None = None) -> None:
    ax.set_facecolor("#2A2D34")
    ax.figure.patch.set_facecolor("#2A2D34")
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    if title:
        ax.set_title(title, color="white")


def _quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class OrientationCanvas(FigureCanvasQTAgg):
    """Native Qt matplotlib canvas for orientation frames."""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(7.2, 5.0), dpi=100)
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")
        _style_3d_axis(self.ax, "Orientación IMU")
        self.ax.set_xlim(-0.7, 0.7)
        self.ax.set_ylim(-0.7, 0.7)
        self.ax.set_zlim(-0.7, 0.7)
        self.ax.set_xlabel("X", color="white")
        self.ax.set_ylabel("Y", color="white")
        self.ax.set_zlabel("Z", color="white")
        self._poly_collection = Poly3DCollection(
            self._face_polys(BOX_VERTICES),
            facecolors=FACE_COLORS,
            edgecolors=(1.0, 1.0, 1.0, 0.9),
            linewidths=1.2,
            zsort="average",
        )
        self.ax.add_collection3d(self._poly_collection)
        self.figure.tight_layout()

    @staticmethod
    def _face_polys(vertices: np.ndarray) -> list[list[list[float]]]:
        verts = vertices.tolist()
        return [[verts[i] for i in face] for face in BOX_FACES]

    def update_quaternion(self, qw: float, qx: float, qy: float, qz: float) -> None:
        rotated = (_quat_to_matrix(qw, qx, qy, qz) @ BOX_VERTICES.T).T
        self._poly_collection.set_verts(self._face_polys(rotated))
        self.draw_idle()


class MagneticCanvas(FigureCanvasQTAgg):
    """Native Qt matplotlib canvas for magnetic scatter and hard-iron fit."""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(7.2, 5.0), dpi=100)
        super().__init__(self.figure)
        self.ax = self.figure.add_subplot(111, projection="3d")
        _style_3d_axis(self.ax, "Calibración Hard-Iron")
        self.ax.set_xlabel("Mx", color="white")
        self.ax.set_ylabel("My", color="white")
        self.ax.set_zlabel("Mz", color="white")
        self._points = {
            "x": deque(maxlen=MAX_MAG_POINTS),
            "y": deque(maxlen=MAX_MAG_POINTS),
            "z": deque(maxlen=MAX_MAG_POINTS),
        }
        self._scatter = None
        self._trail_scatter = None
        self._center_dot = None
        self._sphere_surface = None
        self._center = (0.0, 0.0, 0.0)
        self._radius = 0.0
        self._center_mod_history: deque[float] = deque(maxlen=CONVERGENCE_WINDOW)
        self._stable = False
        self.figure.tight_layout()

    def add_point(self, mx: float, my: float, mz: float) -> None:
        self._points["x"].append(mx)
        self._points["y"].append(my)
        self._points["z"].append(mz)

    def calibration(self) -> dict[str, object] | None:
        """Return the latest hard-iron sphere fit data."""
        if self._radius <= 0:
            return None
        cx, cy, cz = self._center
        samples_count = len(self._points["x"])
        variation = self.stability_variation()
        return {
            "center_lsb": [cx, cy, cz],
            "center_ut": [cx * 0.15, cy * 0.15, cz * 0.15],
            "radius_lsb": self._radius,
            "radius_ut": self._radius * 0.15,
            "samples_count": samples_count,
            "scale_ut_per_lsb": 0.15,
            "stable": self._stable,
            "stability_variation_lsb": variation,
        }

    def stability_variation(self) -> float | None:
        """Return convergence-window center-module variation in LSB."""
        if len(self._center_mod_history) < CONVERGENCE_WINDOW:
            return None
        return float(max(self._center_mod_history) - min(self._center_mod_history))

    def clear(self) -> None:
        self._points["x"].clear()
        self._points["y"].clear()
        self._points["z"].clear()
        self._center = (0.0, 0.0, 0.0)
        self._radius = 0.0
        self._center_mod_history.clear()
        self._stable = False
        for artist_name in ("_scatter", "_trail_scatter", "_center_dot", "_sphere_surface"):
            artist = getattr(self, artist_name)
            if artist is not None:
                artist.remove()
                setattr(self, artist_name, None)
        self.draw_idle()

    def update_view(self) -> None:
        n = len(self._points["x"])
        if n == 0:
            return

        xs = np.array(self._points["x"], dtype=float)
        ys = np.array(self._points["y"], dtype=float)
        zs = np.array(self._points["z"], dtype=float)

        if self._scatter is not None:
            self._scatter.remove()
        self._scatter = self.ax.scatter(xs, ys, zs, c=zs, cmap="coolwarm", s=6, alpha=0.65)

        self._draw_latest_trail(xs, ys, zs)

        all_vals = np.concatenate([xs, ys, zs])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = max((vmax - vmin) * 0.1, 50)
        self.ax.set_xlim(vmin - margin, vmax + margin)
        self.ax.set_ylim(vmin - margin, vmax + margin)
        self.ax.set_zlim(vmin - margin, vmax + margin)

        if n >= 20:
            self._fit_sphere(xs, ys, zs)
        self._draw_fit()
        self.draw_idle()

    def _draw_latest_trail(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
        if self._trail_scatter is not None:
            self._trail_scatter.remove()
            self._trail_scatter = None
        trail_count = min(10, len(xs))
        if trail_count == 0:
            return
        idx = np.arange(len(xs) - trail_count, len(xs))
        t_norm = np.linspace(0.15, 1.0, trail_count)
        sizes = 16 + t_norm * 44
        colors = np.zeros((trail_count, 4))
        colors[:, 0] = 0.10
        colors[:, 1] = 1.00
        colors[:, 2] = 0.20
        colors[:, 3] = 0.18 + t_norm * 0.72
        colors[-1] = [1.0, 1.0, 1.0, 1.0]
        sizes[-1] = 78
        self._trail_scatter = self.ax.scatter(
            xs[idx],
            ys[idx],
            zs[idx],
            c=colors,
            s=sizes,
            alpha=None,
            depthshade=False,
            edgecolors="#FFFFFF",
            linewidths=0.6,
            zorder=100,
        )

    def _fit_sphere(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
        cx0, cy0, cz0 = xs.mean(), ys.mean(), zs.mean()
        r0 = float(np.mean(np.sqrt((xs - cx0) ** 2 + (ys - cy0) ** 2 + (zs - cz0) ** 2)))

        def residuals(p: np.ndarray) -> np.ndarray:
            cx, cy, cz, r = p
            return np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2) - r

        try:
            res = least_squares(residuals, x0=[cx0, cy0, cz0, r0], method="lm", max_nfev=50)
        except Exception:  # noqa: BLE001
            return
        self._center = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
        self._radius = abs(float(res.x[3]))
        cx, cy, cz = self._center
        self._center_mod_history.append(float(np.sqrt(cx**2 + cy**2 + cz**2)))
        variation = self.stability_variation()
        self._stable = variation is not None and variation < CONVERGENCE_THR

    def _draw_fit(self) -> None:
        if self._radius <= 0:
            return
        cx, cy, cz = self._center
        r = self._radius
        if self._sphere_surface is not None:
            self._sphere_surface.remove()
        if self._center_dot is not None:
            self._center_dot.remove()

        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 16)
        sx = cx + r * np.outer(np.cos(u), np.sin(v))
        sy = cy + r * np.outer(np.sin(u), np.sin(v))
        sz = cz + r * np.outer(np.ones_like(u), np.cos(v))
        self._sphere_surface = self.ax.plot_surface(
            sx,
            sy,
            sz,
            alpha=0.09,
            color="#2E7D32" if self._stable else "#FF6F00",
            linewidth=0,
            antialiased=False,
        )
        self._center_dot = self.ax.scatter(
            [cx],
            [cy],
            [cz],
            marker="+",
            s=220,
            c="#2E7D32" if self._stable else "#FF6F00",
            linewidths=2.8,
            depthshade=False,
        )
