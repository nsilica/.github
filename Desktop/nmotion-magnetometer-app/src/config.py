"""Carga y acceso a la configuración de la aplicación."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AppConfig:
    """Configuración global de la app, cargada desde config.json."""

    _instance: AppConfig | None = None

    def __new__(cls, path: str | Path | None = None) -> AppConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(path or "config.json")
        return cls._instance

    def _load(self, path: str | Path) -> None:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parent.parent / config_path

        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = self._defaults()

        # Asegurar que el directorio de grabaciones existe
        rec_dir = Path(self._data["recording"].get("directory", "./recordings"))
        if not rec_dir.is_absolute():
            rec_dir = config_path.parent / rec_dir
        rec_dir.mkdir(parents=True, exist_ok=True)
        self._recording_directory_resolved = rec_dir

        # Asegurar que el directorio de metadatos existe
        meta_dir = Path(self._data.get("metadata", {}).get("directory", "./metadata"))
        if not meta_dir.is_absolute():
            meta_dir = config_path.parent / meta_dir
        meta_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_directory_resolved = meta_dir

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {
            "app": {"title": "nMotion Magnetometer", "window_width": 1280, "window_height": 800},
            "serial": {"port": None, "baudrate": 115200, "auto_connect": True},
            "recording": {"directory": "./recordings", "format": "json", "max_samples": 100000},
            "metadata": {"directory": "./metadata"},
            "ui": {
                "background_color": "#F6FBFF",
                "accent_color": "#1D85ED",
                "danger_color": "#CF2E2E",
                "success_color": "#2E7D32",
                "surface_dark": "#2A2D34",
                "text_dark": "#404040",
                "text_light": "#FFFFFF",
            },
        }

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    @property
    def app_title(self) -> str:
        return self._data["app"]["title"]

    @property
    def window_width(self) -> int:
        return int(self._data["app"]["window_width"])

    @property
    def window_height(self) -> int:
        return int(self._data["app"]["window_height"])

    @property
    def serial_port(self) -> str | None:
        return self._data["serial"].get("port")

    @property
    def serial_baudrate(self) -> int:
        return int(self._data["serial"]["baudrate"])

    @property
    def serial_auto_connect(self) -> bool:
        return bool(self._data["serial"].get("auto_connect", True))

    @property
    def recording_directory(self) -> Path:
        return self._recording_directory_resolved

    @property
    def recording_max_samples(self) -> int:
        return int(self._data["recording"].get("max_samples", 100000))

    @property
    def metadata_directory(self) -> Path:
        return self._metadata_directory_resolved

    @property
    def ui_colors(self) -> dict[str, str]:
        return self._data["ui"]
