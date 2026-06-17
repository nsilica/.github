#!/usr/bin/env python3
"""Genera grabaciones de prueba para probar reproducción sin dongle."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recorder import Recording, RecordingManager


def generate_orientation_recording(duration_ms: int = 5000, fps: int = 45) -> Recording:
    """Genera una grabación de orientación con la caja rotando."""
    samples = []
    n = int(duration_ms / 1000 * fps)
    for i in range(n):
        t_ms = int(i * 1000 / fps)
        angle = 2 * math.pi * i / n
        # Cuaternión de rotación alrededor de eje diagonal
        axis = [1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)]
        half = angle / 2
        qw = math.cos(half)
        qx = axis[0] * math.sin(half)
        qy = axis[1] * math.sin(half)
        qz = axis[2] * math.sin(half)
        # A Q15 (sin corrección de montaje, se aplica luego en reproducción)
        scale = 32768.0
        sample = {
            "t_ms": t_ms,
            "seq": i,
            "raw": {
                "q": [int(qw * scale), int(qx * scale), int(qy * scale), int(qz * scale)],
                "acc": [0, 0, 0],
                "mag": [0, 0, 0],
                "flags": 0x13,
            },
            "outputs": {
                "q": [qw, qx, qy, qz],
                "mag_ut": [0.0, 0.0, 0.0],
            },
        }
        samples.append(sample)
    return Recording(mode="orientation", samples=samples)


def generate_magnetic_recording(duration_ms: int = 5000, fps: int = 45) -> Recording:
    """Genera una grabación magnética con puntos sobre una esfera desplazada."""
    samples = []
    n = int(duration_ms / 1000 * fps)
    cx, cy, cz = 30, -220, 55
    r = 80
    for i in range(n):
        t_ms = int(i * 1000 / fps)
        # Muestras sobre esfera con offset hard-iron
        theta = 2 * math.pi * i / n
        phi = math.pi * i / n
        mx = int(cx + r * math.sin(phi) * math.cos(theta))
        my = int(cy + r * math.sin(phi) * math.sin(theta))
        mz = int(cz + r * math.cos(phi))
        sample = {
            "t_ms": t_ms,
            "seq": i,
            "raw": {
                "q": [32767, 0, 0, 0],
                "acc": [0, 0, 0],
                "mag": [mx, my, mz],
                "flags": 0x13,
            },
            "outputs": {
                "q": [1.0, 0.0, 0.0, 0.0],
                "mag_ut": [mx * 0.15, my * 0.15, mz * 0.15],
            },
        }
        samples.append(sample)
    return Recording(mode="magnetic", samples=samples)


def main() -> None:
    manager = RecordingManager()
    rec_o = generate_orientation_recording()
    path_o = manager.save(rec_o, "test_orientacion")
    print(f"Grabación de orientación: {path_o}")

    rec_m = generate_magnetic_recording()
    path_m = manager.save(rec_m, "test_magnetico")
    print(f"Grabación magnética: {path_m}")


if __name__ == "__main__":
    main()
