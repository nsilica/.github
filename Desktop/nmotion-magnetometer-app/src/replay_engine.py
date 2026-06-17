"""Motor de reproducción de grabaciones con timeline y bucle."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from src.recorder import Recording


class ReplayEngine:
    """Reproduce una grabación en bucle con control de posición."""

    def __init__(self, on_frame: Callable[[dict[str, Any]], Any]) -> None:
        self._on_frame = on_frame
        self._recording: Recording | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Empezar pausado
        self._position_ms = 0.0
        self._lock = threading.Lock()
        self._speed = 1.0
        self._loop = True
        self._playing = False

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def duration_ms(self) -> int:
        with self._lock:
            if self._recording is None:
                return 0
            return self._recording.header.duration_ms or (
                self._recording.samples[-1]["t_ms"] if self._recording.samples else 0
            )

    @property
    def position_ms(self) -> float:
        with self._lock:
            return self._position_ms

    def load(self, recording: Recording) -> None:
        """Carga una nueva grabación."""
        self.stop()
        with self._lock:
            self._recording = recording
            self._position_ms = 0.0
            self._playing = False

    def play(self) -> None:
        """Inicia o reanuda la reproducción."""
        if self._recording is None or not self._recording.samples:
            return
        self._pause_event.set()
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        with self._lock:
            self._playing = True

    def pause(self) -> None:
        """Pausa la reproducción."""
        self._pause_event.clear()
        with self._lock:
            self._playing = False

    def stop(self) -> None:
        """Detiene la reproducción y resetea posición."""
        self._stop_event.set()
        self._pause_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        with self._lock:
            self._position_ms = 0.0
            self._playing = False

    def seek(self, position_ms: float) -> None:
        """Mueve el cursor a una posición en ms."""
        duration = self.duration_ms
        with self._lock:
            self._position_ms = max(0.0, min(position_ms, duration))

    def set_speed(self, speed: float) -> None:
        """Cambia la velocidad de reproducción."""
        with self._lock:
            self._speed = max(0.1, min(speed, 4.0))

    def _run(self) -> None:
        """Bucle de reproducción."""
        last_tick = time.monotonic()
        while not self._stop_event.is_set():
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            now = time.monotonic()
            delta_ms = (now - last_tick) * 1000.0
            last_tick = now

            with self._lock:
                recording = self._recording
                speed = self._speed
                loop = self._loop
                current_pos = self._position_ms + delta_ms * speed

            if recording is None or not recording.samples:
                time.sleep(0.01)
                last_tick = time.monotonic()
                continue

            duration = self.duration_ms
            if current_pos >= duration:
                current_pos = 0.0 if loop else float(duration)
                if not loop:
                    self.pause()

            with self._lock:
                self._position_ms = current_pos

            sample = self._interpolate_sample(recording.samples, current_pos)
            if sample is not None:
                with suppress(Exception):
                    self._on_frame(sample)

            time.sleep(1 / 60)  # ~60 fps de refresco UI

    @staticmethod
    def _interpolate_sample(samples: list[dict[str, Any]], position_ms: float) -> dict[str, Any] | None:
        """Devuelve la muestra más cercana a position_ms."""
        if not samples:
            return None
        # Búsqueda lineal simple; samples suele estar ordenado
        best = samples[0]
        best_diff = abs(best.get("t_ms", 0) - position_ms)
        for s in samples[1:]:
            diff = abs(s.get("t_ms", 0) - position_ms)
            if diff < best_diff:
                best_diff = diff
                best = s
            else:
                break
        return best
