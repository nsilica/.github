"""Base para renderizar figuras matplotlib como imágenes para Flet."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BasePlot:
    """Renderiza una figura matplotlib a base64 PNG."""

    def __init__(self, width: int = 800, height: int = 600, dpi: int = 100) -> None:
        self.width = width
        self.height = height
        self.dpi = dpi
        self.fig: plt.Figure | None = None
        self.ax: Any = None
        self._setup()

    def _setup(self) -> None:
        """Sobrescribir en subclases para crear figura y ejes."""
        self.fig, self.ax = plt.subplots(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)

    def render(self) -> str:
        """Devuelve la imagen codificada en base64 (data URI)."""
        if self.fig is None:
            self._setup()
        buf = io.BytesIO()
        # Evitar bbox_inches='tight' reduce bastante el coste de render por frame.
        self.fig.savefig(buf, format="png")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def close(self) -> None:
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
