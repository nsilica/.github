"""Hilo de lectura del dongle nMotion por puerto serie."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import serial
import serial.tools.list_ports

from src.config import AppConfig
from src.dongle_protocol import IMUData, MessageType, decode_imu_data, parse_frame


class SerialWorker:
    """Lee el dongle en un hilo aparte y entrega IMUData a un callback."""

    def __init__(self, on_data: Callable[[IMUData], Any]) -> None:
        self._cfg = AppConfig()
        self._on_data = on_data
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()
        self._connected = False
        self._last_error: str | None = None

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @staticmethod
    def find_dongle_port() -> str | None:
        """Busca el puerto del dongle por VID:PID configurado."""
        cfg = AppConfig()
        vid = int(cfg._data["serial"].get("vid", "0x2FE3"), 16)
        pid = int(cfg._data["serial"].get("pid", "0x4E4D"), 16)

        for p in serial.tools.list_ports.comports():
            if p.vid == vid and p.pid == pid:
                return p.device
            desc = (p.description or "").lower()
            if "nMotion" in (p.description or ""):
                return p.device
        return None

    def start(self, port: str | None = None, baudrate: int | None = None) -> bool:
        """Inicia la lectura serie."""
        if self._thread is not None and self._thread.is_alive():
            return True

        target_port = port or self._cfg.serial_port or self.find_dongle_port()
        target_baud = baudrate or self._cfg.serial_baudrate

        if not target_port:
            self._set_error("No se detectó ningún dongle nMotion.")
            return False

        try:
            self._ser = serial.Serial(
                port=target_port,
                baudrate=target_baud,
                timeout=0.05,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            print(f"[nMotion] Connected serial port: {target_port} @ {target_baud}")
        except Exception as exc:  # noqa: BLE001
            self._set_error(f"No se pudo abrir {target_port}: {exc}")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Detiene la lectura y cierra el puerto."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                try:
                    self._ser.close()
                except Exception:  # noqa: BLE001
                    pass
                self._ser = None
            self._connected = False

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
            self._connected = False

    def _set_connected(self, value: bool) -> None:
        with self._lock:
            self._connected = value

    def _run(self) -> None:
        """Bucle principal de lectura."""
        buf = bytearray()
        empty_loops = 0

        while not self._stop_event.is_set():
            if self._ser is None or not self._ser.is_open:
                self._set_connected(False)
                break

            try:
                if self._ser.in_waiting > 0:
                    data = self._ser.read(self._ser.in_waiting)
                    buf.extend(data)
                    self._set_connected(True)
                    empty_loops = 0
                else:
                    empty_loops += 1
                    if empty_loops > 100:
                        time.sleep(0.001)
                        empty_loops = 0
                    continue
            except serial.SerialException as exc:
                self._set_error(f"Error de lectura: {exc}")
                self._set_connected(False)
                break

            while True:
                msg_type, payload, _ = parse_frame(buf)
                if msg_type is None:
                    break
                if msg_type == MessageType.IMU_DATA and payload is not None:
                    imu = decode_imu_data(payload)
                    if imu is not None:
                        if imu.seq % 100 == 0:
                            print(f"[nMotion] RX seq={imu.seq} mag=({imu.mx},{imu.my},{imu.mz})")
                        try:
                            self._on_data(imu)
                        except Exception:  # noqa: BLE001
                            pass

        self._set_connected(False)
