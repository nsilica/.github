"""Persistencia de calibraciones en metadatos (metadata/)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import AppConfig


class CalibrationStore:
    """Carga, guarda y elimina calibraciones magnéticas y de orientación."""

    MAGNETIC_FILE = "magnetic_calibration.json"
    ORIENTATION_FILE = "orientation_calibration.json"

    def __init__(self) -> None:
        self._cfg = AppConfig()
        self.directory = Path(self._cfg.metadata_directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, filename: str) -> Path:
        return self.directory / filename

    def _load_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return None

    def _save_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # --- Magnetic calibration ---

    def load_magnetic_calibration(self) -> dict[str, Any] | None:
        """Load saved magnetic calibration from metadata, migrating old path if needed."""
        path = self._path(self.MAGNETIC_FILE)
        data = self._load_json(path)
        if data is not None:
            return data.get("calibration")

        # Backward compatibility: migrate from recordings directory
        old_path = Path(self._cfg.recording_directory) / "magnetic_hard_iron_calibration.json"
        if old_path.exists():
            try:
                old_data = self._load_json(old_path)
                if old_data is not None:
                    calibration = old_data.get("calibration")
                    points = old_data.get("points_lsb")
                    if isinstance(calibration, dict):
                        self.save_magnetic_calibration(calibration, points)
                        return calibration
            except Exception:  # noqa: BLE001
                return None
        return None

    def save_magnetic_calibration(
        self,
        calibration: dict[str, Any],
        points_lsb: list[list[float]] | None = None,
    ) -> Path:
        """Save magnetic calibration and optional point subset to metadata."""
        payload = {
            "version": 1,
            "type": "magnetic_hard_iron",
            "created_at": datetime.now(UTC).isoformat(),
            "calibration": calibration,
        }
        if points_lsb is not None:
            payload["points_lsb"] = points_lsb
        path = self._path(self.MAGNETIC_FILE)
        self._save_json(path, payload)
        return path

    def delete_magnetic_calibration(self) -> None:
        """Delete saved magnetic calibration metadata."""
        path = self._path(self.MAGNETIC_FILE)
        if path.exists():
            path.unlink()

    # --- Orientation calibration ---

    def load_orientation_reference(self) -> tuple[float, float, float, float] | None:
        """Load saved orientation reference quaternion."""
        path = self._path(self.ORIENTATION_FILE)
        data = self._load_json(path)
        if data is None:
            return None
        q = data.get("reference_q")
        if isinstance(q, list) and len(q) == 4:
            return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return None

    def save_orientation_reference(
        self, reference_q: tuple[float, float, float, float]
    ) -> Path:
        """Save orientation reference quaternion to metadata."""
        payload = {
            "version": 1,
            "type": "orientation_reference",
            "created_at": datetime.now(UTC).isoformat(),
            "reference_q": list(reference_q),
        }
        path = self._path(self.ORIENTATION_FILE)
        self._save_json(path, payload)
        return path

    def delete_orientation_reference(self) -> None:
        """Delete saved orientation reference metadata."""
        path = self._path(self.ORIENTATION_FILE)
        if path.exists():
            path.unlink()

    def clear_all(self) -> None:
        """Delete all calibration metadata files."""
        for filename in (self.MAGNETIC_FILE, self.ORIENTATION_FILE):
            path = self._path(filename)
            if path.exists():
                path.unlink()
