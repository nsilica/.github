"""Gestión de grabaciones JSON: guardar, cargar, listar y metadata."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import AppConfig


@dataclass(slots=True)
class RecordingHeader:
    """Metadatos de una grabación."""

    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    mode: str = ""
    samples_count: int = 0
    duration_ms: int = 0
    device_id: int | None = None


class Recording:
    """Una grabación completa con metadatos y muestras."""

    def __init__(self, mode: str, samples: list[dict[str, Any]]) -> None:
        self.header = RecordingHeader(mode=mode)
        self.samples = samples
        self.header.samples_count = len(samples)
        if samples:
            self.header.duration_ms = int(samples[-1].get("t_ms", 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": asdict(self.header),
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recording:
        header = RecordingHeader(**data.get("header", {}))
        rec = cls.__new__(cls)
        samples = data.get("samples", [])
        if samples:
            first_t = int(samples[0].get("t_ms", 0))
            if first_t != 0:
                for sample in samples:
                    original_t = int(sample.get("t_ms", 0))
                    sample.setdefault("device_t_ms", original_t)
                    sample["t_ms"] = max(0, original_t - first_t)
                header.duration_ms = int(samples[-1].get("t_ms", 0))
        rec.header = header
        rec.samples = samples
        return rec


class RecordingManager:
    """Guarda y carga grabaciones desde el directorio configurado."""

    def __init__(self) -> None:
        self._cfg = AppConfig()
        self.directory = Path(self._cfg.recording_directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def list_recordings(self) -> list[Path]:
        """Lista todos los archivos de grabación ordenados por fecha descendente."""
        files = [p for p in self.directory.iterdir() if p.suffix == ".json" and p.is_file()]
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    def generate_filename(self, name: str | None = None) -> str:
        base = name.strip() if name else "recording"
        base = base.replace(" ", "_").replace("/", "_").replace("\\", "_")
        unique = uuid.uuid4().hex[:6]
        return f"{base}_{unique}.json"

    def save(self, recording: Recording, name: str | None = None) -> Path:
        """Guarda una grabación y devuelve la ruta."""
        filename = self.generate_filename(name)
        path = self.directory / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(recording.to_dict(), f, indent=2)
        return path

    def load(self, path: Path) -> Recording:
        """Carga una grabación desde disco."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return Recording.from_dict(data)

    def delete(self, path: Path) -> None:
        """Elimina una grabación."""
        if path.exists() and path.suffix == ".json":
            path.unlink()
