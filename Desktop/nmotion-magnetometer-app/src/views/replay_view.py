"""Vista de reproducción: lista de grabaciones y player con timeline."""

from __future__ import annotations

import threading
import time

import flet as ft

from src.plots.magnetic_plot import MagneticPlot
from src.plots.orientation_plot import OrientationPlot
from src.recorder import Recording, RecordingManager
from src.replay_engine import ReplayEngine
from src.theme import CENTER_ALIGNMENT, AppTheme, border_all
from src.views.plot_overlay import PlotOverlay


class ReplayView(ft.View):
    """Vista de selección y reproducción de grabaciones."""

    def __init__(self, on_back: callable) -> None:
        theme = AppTheme()
        super().__init__(
            route="/replay",
            bgcolor=theme.bg,
            padding=0,
        )
        self._on_back = on_back
        self._theme = theme
        self._recorder = RecordingManager()
        self._engine = ReplayEngine(on_frame=self._on_frame)

        self._orientation_plot = OrientationPlot(width=560, height=380, dpi=85)
        self._magnetic_plot = MagneticPlot(width=560, height=380, dpi=85)

        self._overlay: PlotOverlay | None = None
        self._current_recording: Recording | None = None
        self._update_timer: threading.Thread | None = None
        self._stop_timer = threading.Event()
        self._user_seeking = False

        self._recordings_list = ft.Column(spacing=12, scroll=ft.ScrollMode.AUTO, expand=True)
        self.controls = self._build()
        self._refresh_list()

    def _build(self) -> list[ft.Control]:
        return [
            ft.Container(
                expand=True,
                bgcolor=self._theme.bg,
                padding=40,
                content=ft.Row(
                    spacing=32,
                    controls=[
                        # Panel izquierdo
                        ft.Container(
                            width=360,
                            padding=24,
                            border_radius=self._theme.radius_lg,
                            bgcolor=self._theme.card_bg,
                            shadow=self._theme.shadow,
                            border=border_all(1, self._theme.border),
                            content=ft.Column(
                                spacing=24,
                                controls=[
                                    ft.Row(
                                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                        controls=[
                                            self._theme.headline("Reproducción", size=24),
                                            ft.IconButton(
                                                icon=ft.Icons.ARROW_BACK,
                                                icon_color=self._theme.text_dark,
                                                tooltip="Volver",
                                                on_click=lambda _: self._go_back(),
                                            ),
                                        ],
                                    ),
                                    self._theme.muted("Selecciona una grabación guardada.", size=13),
                                    self._recordings_list,
                                ],
                            ),
                        ),
                        # Panel derecho
                        ft.Container(
                            expand=True,
                            alignment=CENTER_ALIGNMENT,
                            content=self._theme.muted("Selecciona una grabación para reproducir", size=16),
                        ),
                    ],
                ),
            ),
        ]

    def _go_back(self) -> None:
        self._close_overlay()
        self._engine.stop()
        self._on_back()

    def _refresh_list(self) -> None:
        self._recordings_list.controls.clear()
        paths = self._recorder.list_recordings()
        if not paths:
            self._recordings_list.controls.append(
                ft.Container(
                    padding=24,
                    alignment=CENTER_ALIGNMENT,
                    content=self._theme.muted("No hay grabaciones guardadas", size=14),
                ),
            )
        else:
            for path in paths:
                try:
                    rec = self._recorder.load(path)
                    self._recordings_list.controls.append(
                        self._recording_card(path, rec),
                    )
                except Exception:  # noqa: BLE001
                    continue


    def _recording_card(self, path, rec: Recording) -> ft.Container:
        from datetime import datetime

        created = rec.header.created_at
        try:
            dt = datetime.fromisoformat(created)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:  # noqa: BLE001
            date_str = created

        return ft.Container(
            padding=16,
            border_radius=self._theme.radius_lg,
            bgcolor=self._theme.card_bg,
            border=border_all(1, self._theme.border),
            animate=ft.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
            on_click=lambda _, p=path: self._play_recording(p),
            content=ft.Column(
                spacing=6,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Text(path.stem, size=14, weight=ft.FontWeight.W_600, color=self._theme.text_dark),
                            ft.IconButton(
                                icon=ft.Icons.DELETE,
                                icon_color=self._theme.danger,
                                tooltip="Eliminar",
                                on_click=lambda e, p=path: self._delete_recording(e, p),
                            ),
                        ],
                    ),
                    ft.Text(f"Modo: {rec.header.mode}", size=12, color=self._theme.text_muted),
                    ft.Text(
                        f"{rec.header.samples_count} muestras · {self._format_ms(rec.header.duration_ms)} · {date_str}",
                        size=11,
                        color=self._theme.text_muted,
                    ),
                ],
            ),
        )

    def _play_recording(self, path: Path) -> None:
        self._close_overlay()
        self._current_recording = self._recorder.load(path)
        mode = self._current_recording.header.mode

        if mode == "orientation":
            title = f"Reproduciendo: {path.stem}"
            image = ft.Image(src=self._orientation_plot.render())
        else:
            title = f"Reproduciendo: {path.stem}"
            image = ft.Image(src=self._magnetic_plot.render())
            # Precargar puntos de la grabación
            for s in self._current_recording.samples:
                mag = s.get("raw", {}).get("mag", [0, 0, 0])
                self._magnetic_plot.add_point(mag[0], mag[1], mag[2])

        self._overlay = PlotOverlay(
            page=self.page,
            title=title,
            image_control=image,
            on_close=self._close_overlay,
            is_replay=True,
        )

        if self._overlay._play_button is not None:
            self._overlay._play_button.on_click = self._toggle_playback
        if self._overlay._timeline is not None:
            self._overlay._timeline.on_change = self._on_timeline_change
            self._overlay._timeline.on_change_end = self._on_timeline_release

        self._overlay.show()

        self._engine.load(self._current_recording)
        self._engine.play()
        self._overlay.set_play_icon(True)
        self._start_ui_timer()

    def _on_frame(self, sample: dict) -> None:
        if self.page is None or self._overlay is None or self._current_recording is None:
            return
        self.page.run(lambda: self._update_frame_ui(sample))

    def _update_frame_ui(self, sample: dict) -> None:
        if self._overlay is None or self._current_recording is None:
            return

        mode = self._current_recording.header.mode
        if mode == "orientation":
            q = sample.get("outputs", {}).get("q", [1.0, 0.0, 0.0, 0.0])
            self._orientation_plot.update(q[0], q[1], q[2], q[3])
            b64 = self._orientation_plot.render()
        else:
            mag = sample.get("raw", {}).get("mag", [0, 0, 0])
            self._magnetic_plot.add_point(mag[0], mag[1], mag[2])
            self._magnetic_plot.update()
            b64 = self._magnetic_plot.render()

        self._overlay.image.src = b64
        self._overlay.image.update()


    def _toggle_playback(self, _: ft.ControlEvent) -> None:
        if self._engine.playing:
            self._engine.pause()
            if self._overlay is not None:
                self._overlay.set_play_icon(False)
        else:
            self._engine.play()
            if self._overlay is not None:
                self._overlay.set_play_icon(True)

    def _on_timeline_change(self, e: ft.ControlEvent) -> None:
        self._user_seeking = True
        if self._overlay is not None and self._overlay._timeline is not None:
            self._engine.pause()
            self._engine.seek(float(self._overlay._timeline.value))

    def _on_timeline_release(self, e: ft.ControlEvent) -> None:
        self._user_seeking = False
        self._engine.play()
        if self._overlay is not None:
            self._overlay.set_play_icon(True)

    def _start_ui_timer(self) -> None:
        self._stop_timer.clear()
        self._update_timer = threading.Thread(target=self._ui_timer_loop, daemon=True)
        self._update_timer.start()

    def _ui_timer_loop(self) -> None:
        while not self._stop_timer.is_set():
            if self.page is not None and self._overlay is not None and not self._user_seeking:
                pos = self._engine.position_ms
                dur = self._engine.duration_ms
                try:
                    self.page.run(lambda p=pos, d=dur: self._overlay.set_timeline(p, d))
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.05)

    def _close_overlay(self) -> None:
        self._stop_timer.set()
        self._engine.stop()
        if self._overlay is not None:
            self._overlay.close()
            self._overlay = None
        self._current_recording = None
        self._magnetic_plot.clear()
        self._orientation_plot = OrientationPlot(width=760, height=560, dpi=90)

    def _delete_recording(self, e: ft.ControlEvent, path: Path) -> None:
        e.stop_propagation()
        self._recorder.delete(path)
        self._refresh_list()

    @staticmethod
    def _format_ms(ms: int) -> str:
        total_seconds = ms / 1000.0
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
