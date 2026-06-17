"""Punto de entrada de la app nMotion Magnetometer."""

from __future__ import annotations

from src.qt_app import run


def main() -> int:
    """Run the PySide6 desktop application."""
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
