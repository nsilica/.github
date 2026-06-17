"""Matplotlib canvases embedded in PySide6 widgets."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path
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

    def load_calibration_state(
        self,
        center_lsb: list[float],
        radius_lsb: float,
        *,
        points_lsb: list[list[float]] | None = None,
        stable: bool = False,
    ) -> None:
        """Restore a saved calibration and optional point subset."""
        self.clear()
        self.set_calibration(center_lsb, radius_lsb, stable=stable, draw=False)
        if points_lsb:
            for pt in points_lsb:
                self._points["x"].append(float(pt[0]))
                self._points["y"].append(float(pt[1]))
                self._points["z"].append(float(pt[2]))
        self.render_state()

    def set_calibration(
        self,
        center_lsb: list[float],
        radius_lsb: float,
        *,
        stable: bool = False,
        draw: bool = True,
    ) -> None:
        """Set the fitted sphere parameters without clearing points."""
        self._center = (float(center_lsb[0]), float(center_lsb[1]), float(center_lsb[2]))
        self._radius = float(radius_lsb)
        self._stable = bool(stable)
        if stable:
            self._center_mod_history.clear()
            cx, cy, cz = self._center
            mod = float(np.sqrt(cx**2 + cy**2 + cz**2))
            for _ in range(CONVERGENCE_WINDOW):
                self._center_mod_history.append(mod)
        if draw:
            self._draw_fit()
            self.draw_idle()

    def render_state(self) -> None:
        """Redraw current points and fit without refitting."""
        n = len(self._points["x"])
        if n > 0:
            xs = np.array(self._points["x"], dtype=float)
            ys = np.array(self._points["y"], dtype=float)
            zs = np.array(self._points["z"], dtype=float)
            if self._scatter is not None:
                self._scatter.remove()
            self._scatter = self.ax.scatter(
                xs, ys, zs, c=zs, cmap="coolwarm", s=6, alpha=0.65
            )
            self._draw_latest_trail(xs, ys, zs)
            all_vals = np.concatenate([xs, ys, zs])
            vmin, vmax = all_vals.min(), all_vals.max()
            margin = max((vmax - vmin) * 0.1, 50)
            self.ax.set_xlim(vmin - margin, vmax + margin)
            self.ax.set_ylim(vmin - margin, vmax + margin)
            self.ax.set_zlim(vmin - margin, vmax + margin)
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


class NorthCanvas(FigureCanvasQTAgg):
    """Native Qt matplotlib canvas for a magnetic-north compass."""

    _SCALE = 0.75
    _RECT_SCALE = 0.3 * _SCALE
    _CIRCLE_SCALE = 0.18 * _SCALE
    _CLOUD_SIZE = 120
    _SIGMA_PERP = 0.08 * _SCALE
    _TEARDROP_POWER = 4.2
    _CIRCLE_RADIUS = 0.9 * _CIRCLE_SCALE
    _TEARDROP_BASE = 0.9 * _CIRCLE_SCALE + 0.06 * _SCALE
    _TEARDROP_TIP = 0.33 * _SCALE
    _RECT_W = 2.0 * _RECT_SCALE
    _RECT_H = 2.5 * _RECT_SCALE

    # Heading variation plot config
    _VARIATION_WINDOW_S = 5.0
    _VARIATION_DECAY_TAU_S = 0.4
    _VARIATION_Y_DEFAULT_MAX_DEG = 10.0

    def __init__(self) -> None:
        scale = self._SCALE
        self.figure = Figure(figsize=(2.9, 5.15), dpi=100)
        super().__init__(self.figure)
        gs = self.figure.add_gridspec(
            2, 1, height_ratios=[3.0, 1.4], hspace=0.04, left=0.18, right=0.92, top=0.96, bottom=0.08
        )
        self.ax = self.figure.add_subplot(gs[0])
        self.ax.set_xlim(-0.55 * scale, 0.55 * scale)
        self.ax.set_ylim(-0.7 * scale, 0.65 * scale)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_facecolor("#2A2D34")
        self.ax.set_title("Norte Magnético", color="white", fontsize=12)
        self.figure.patch.set_facecolor("#2A2D34")

        self._heading = 0.0
        self._raw_heading = 0.0
        self._cloud_img = None
        self._outline_patch = None

        # Pre-compute grid for the cloud
        x = np.linspace(-0.5 * scale, 0.5 * scale, self._CLOUD_SIZE)
        y = np.linspace(-0.6 * scale, 0.6 * scale, int(self._CLOUD_SIZE * 1.2))
        self._grid_x, self._grid_y = np.meshgrid(x, y)

        self._draw_static()
        self._render_cloud()

        # Heading variation plot
        self._variation_ax = self.figure.add_subplot(gs[1])
        self._variation_ax.set_facecolor("#2A2D34")
        self._variation_ax.tick_params(colors="white", labelsize=7)
        self._variation_ax.spines["bottom"].set_color("white")
        self._variation_ax.spines["top"].set_visible(False)
        self._variation_ax.spines["right"].set_visible(False)
        self._variation_ax.spines["left"].set_color("white")
        self._variation_ax.set_xlim(-self._VARIATION_WINDOW_S, 0.0)
        self._variation_ax.set_ylim(0.0, self._VARIATION_Y_DEFAULT_MAX_DEG)
        self._variation_ax.set_xlabel("s", color="white", fontsize=6)
        self._variation_ax.set_ylabel("°", color="white", fontsize=6, labelpad=2)
        self._variation_ax.tick_params(colors="white", labelsize=6, pad=2)
        (self._variation_line,) = self._variation_ax.plot([], [], color="white", linewidth=1.2)
        self._variation_ax.axhline(
            y=0.0, color="white", linewidth=0.5, alpha=0.3
        )

        self._heading_times: list[float] = []
        self._heading_values: list[float] = []
        self._variation_values: list[float] = []
        self._last_variation_time: float | None = None
        self._last_variation_held: float = 0.0

    def reset_variation(self) -> None:
        """Clear variation history and redraw empty plot."""
        self._heading_times.clear()
        self._heading_values.clear()
        self._variation_values.clear()
        self._last_variation_time = None
        self._last_variation_held = 0.0
        self._variation_line.set_data([], [])
        self._variation_ax.set_ylim(0.0, self._VARIATION_Y_DEFAULT_MAX_DEG)
        self._variation_ax.set_xlim(-self._VARIATION_WINDOW_S, 0.0)
        self.draw_idle()

    def _draw_static(self) -> None:
        # Rounded rectangle, white outline, no fill, 4:5 aspect ratio, esquinas muy redondeadas
        rect = FancyBboxPatch(
            (-self._RECT_W / 2, -self._RECT_H / 2),
            self._RECT_W,
            self._RECT_H,
            boxstyle="round,pad=0.1",
            fill=False,
            edgecolor="white",
            linewidth=1.6,
            zorder=10,
        )
        self.ax.add_patch(rect)

        # Centered circle, white fill, mucho más pequeño
        circle = Circle(
            (0, 0),
            self._CIRCLE_RADIUS,
            fill=True,
            facecolor="white",
            edgecolor="white",
            linewidth=0.6,
            zorder=10,
        )
        self.ax.add_patch(circle)

        # Triangle pointing up (0° reference), 50% más grande
        tri = 0.09 * self._CIRCLE_SCALE * 1.5
        triangle = Polygon(
            [
                [0, self._CIRCLE_RADIUS + tri],
                [-tri * 0.65, self._CIRCLE_RADIUS],
                [tri * 0.65, self._CIRCLE_RADIUS],
            ],
            closed=True,
            facecolor="white",
            edgecolor="white",
            linewidth=0.5,
            zorder=12,
        )
        self.ax.add_patch(triangle)

    def update_heading(
        self, heading_rad: float, smoothing: float = 0.18, now_s: float | None = None
    ) -> None:
        """Update the cloud heading with exponential smoothing and variation plot."""
        # Normalize difference to [-pi, pi]
        diff = (heading_rad - self._heading + math.pi) % (2 * math.pi) - math.pi
        self._heading = (self._heading + smoothing * diff) % (2 * math.pi)
        self._raw_heading = heading_rad
        self._render_cloud()
        if now_s is not None:
            self._update_variation(now_s)
        self.draw_idle()

    def _teardrop_boundary(self, heading_rad: float) -> np.ndarray:
        """Return Nx2 array of points describing the teardrop outline."""
        h = heading_rad
        d_along = np.array([-math.sin(h), math.cos(h)])
        d_perp = np.array([math.cos(h), math.sin(h)])

        n = 120
        phi = np.linspace(-math.pi, math.pi, n)
        # Local shape: tip at phi=0 (+x), base at phi=pi (-x)
        r_local = self._TEARDROP_BASE + (
            self._TEARDROP_TIP - self._TEARDROP_BASE
        ) * np.maximum(0.0, np.cos(phi)) ** self._TEARDROP_POWER
        x_local = r_local * np.cos(phi)
        y_local = r_local * np.sin(phi)

        # Rotate to heading direction
        pts = (
            x_local[:, None] * d_along[None, :]
            + y_local[:, None] * d_perp[None, :]
        )
        return pts

    def _render_cloud(self) -> None:
        if self._cloud_img is not None:
            self._cloud_img.remove()
            self._cloud_img = None
        if self._outline_patch is not None:
            self._outline_patch.remove()
            self._outline_patch = None

        h = self._heading
        d_along = np.array([-math.sin(h), math.cos(h)])
        d_perp = np.array([math.cos(h), math.sin(h)])

        gx, gy = self._grid_x, self._grid_y
        u = gx * d_along[0] + gy * d_along[1]
        v = gx * d_perp[0] + gy * d_perp[1]

        # Teardrop mask in local coordinates
        phi = np.arctan2(v, u)
        r = np.sqrt(u**2 + v**2)
        r_teardrop = self._TEARDROP_BASE + (
            self._TEARDROP_TIP - self._TEARDROP_BASE
        ) * np.maximum(0.0, np.cos(phi)) ** self._TEARDROP_POWER
        # Pequeño overflow del 2% para evitar bug visual en el borde
        teardrop_mask = r <= r_teardrop * 1.02

        # Gradient: sólido azul en el pico, semitransparente cerca del centro
        u_norm = (u + self._TEARDROP_BASE) / (
            self._TEARDROP_TIP + self._TEARDROP_BASE
        )
        u_norm = np.clip(u_norm, 0.0, 1.0)
        axial = u_norm**0.5
        cross = np.exp(-0.5 * (v / self._SIGMA_PERP) ** 2)
        cloud = axial * cross
        cloud *= teardrop_mask

        rgba = np.zeros((*cloud.shape, 4))
        # Color y alpha multiplicados por la máscara para evitar renderizado fuera del outline
        rgba[:, :, 0] = 0.18 * cloud
        rgba[:, :, 1] = 0.58 * cloud
        rgba[:, :, 2] = 0.98 * cloud
        rgba[:, :, 3] = cloud * 0.9

        extent = [
            float(self._grid_x.min()),
            float(self._grid_x.max()),
            float(self._grid_y.min()),
            float(self._grid_y.max()),
        ]
        self._cloud_img = self.ax.imshow(
            rgba,
            extent=extent,
            origin="lower",
            interpolation="bilinear",
            zorder=2,
        )

        # Solid blue outline around the teardrop
        boundary = self._teardrop_boundary(h)
        path = Path(boundary)
        self._outline_patch = PathPatch(
            path,
            facecolor="none",
            edgecolor="#1D85ED",
            linewidth=2.2,
            zorder=11,
        )
        self.ax.add_patch(self._outline_patch)

    def _update_variation(self, now_s: float) -> None:
        heading_deg = math.degrees(self._raw_heading)
        self._heading_times.append(now_s)
        self._heading_values.append(heading_deg)

        # Compute angular diff with circular wrap correction
        if len(self._heading_values) >= 2:
            prev = self._heading_values[-2]
            diff = (heading_deg - prev + 180.0) % 360.0 - 180.0
        else:
            diff = 0.0
        raw = abs(diff)

        # Peak-hold with exponential decay: each plotted value is the maximum
        # between the current raw diff and the previously held value decayed
        # over time. Past plotted points are never modified; they simply scroll
        # out of the window. When the heading stabilizes, the held value decays
        # toward 0.
        if self._last_variation_time is None:
            held = raw
        else:
            dt = now_s - self._last_variation_time
            if dt <= 0.0 or dt > 1.0:
                held = raw
            else:
                decay = math.exp(-dt / self._VARIATION_DECAY_TAU_S)
                held = max(raw, self._last_variation_held * decay)
        self._variation_values.append(held)
        self._last_variation_time = now_s
        self._last_variation_held = held

        # Drop older data outside display window
        cutoff = now_s - self._VARIATION_WINDOW_S
        while self._heading_times and self._heading_times[0] < cutoff:
            self._heading_times.pop(0)
            self._heading_values.pop(0)
            self._variation_values.pop(0)

        if len(self._heading_times) < 2:
            return

        var_times = [t - now_s for t in self._heading_times]
        self._variation_line.set_data(var_times, self._variation_values)
        visible_values = list(self._variation_values)
        if visible_values:
            current_max = max(visible_values)
            y_max = max(current_max * 1.2, 2.0)
            self._variation_ax.set_ylim(0.0, y_max)
        self._variation_ax.set_xlim(-self._VARIATION_WINDOW_S, 0.0)
