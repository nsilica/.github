"""Vista de medición: lista de modos y overlays de gráficos."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone

import flet as ft

from src.dongle_protocol import IMUData
from src.plots.magnetic_plot import MagneticPlot
from src.plots.orientation_plot import OrientationPlot
from src.recorder import Recording, RecordingManager
from src.serial_worker import SerialWorker
from src.theme import CENTER_ALIGNMENT, AppTheme, border_all
from src.views.plot_overlay import PlotOverlay


class MeasureView(ft.View):
    """Vista de selección de modo de medición."""

    def __init__(self, on_back: callable) -> None:
        theme = AppTheme()
        super().__init__(
            route="/measure",
            bgcolor=theme.bg,
            padding=0,
        )
        self._on_back = on_back
        self._theme = theme
        self._worker = SerialWorker(on_data=self._on_imu_data)
        self._recorder = RecordingManager()

        self._orientation_plot = OrientationPlot(width=560, height=380, dpi=85)
        self._magnetic_plot = MagneticPlot(width=560, height=380, dpi=85)

        self._current_mode: str | None = None
        self._overlay: PlotOverlay | None = None
        self._latest_imu: IMUData | None = None
        self._latest_orientation_q: tuple[float, float, float, float] | None = None
        self._orientation_reference_q: tuple[float, float, float, float] | None = None
        self._latest_lock = threading.Lock()
        self._ui_loop_started = False
        self._content_host = ft.Container(
            expand=True,
            padding=ft.Padding(0, 0, 0, 0),
            alignment=CENTER_ALIGNMENT,
            content=self._theme.muted("Selecciona un modo de medición para comenzar", size=16),
        )
        self._recording_samples: list[dict] = []
        self._recording_start: float | None = None
        self._recording_timer_running = False

        self.controls = self._build()

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
                                            self._theme.headline("Medición", size=24),
                                            ft.IconButton(
                                                icon=ft.Icons.ARROW_BACK,
                                                icon_color=self._theme.text_dark,
                                                tooltip="Volver",
                                                on_click=lambda _: self._go_back(),
                                            ),
                                        ],
                                    ),
                                    self._theme.muted("Selecciona un modo para visualizar en tiempo real.", size=13),
                                    ft.Column(
                                        spacing=16,
                                        controls=[
                                            self._mode_card(
                                                "Medir orientación",
                                                "Visualiza la orientación 3D de la IMU",
                                                ft.Icons.VIEW_IN_AR,
                                                lambda _: self._open_mode("orientation"),
                                            ),
                                            self._mode_card(
                                                "Medir campo magnético",
                                                "Nube de puntos del magnetómetro y ajuste de esfera",
                                                ft.Icons.COMPASS_CALIBRATION,
                                                lambda _: self._open_mode("magnetic"),
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ),
                        # Panel derecho (placeholder)
                        self._content_host,
                    ],
                ),
            ),
        ]

    def _mode_card(self, title: str, subtitle: str, icon: str, on_click: callable) -> ft.Container:
        return ft.ElevatedButton(
            width=320,
            bgcolor=self._theme.card_bg,
            color=self._theme.text_dark,
            elevation=0,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=self._theme.radius_lg),
                padding=ft.Padding(20, 20, 20, 20),
                overlay_color="#1D85ED12",
                bgcolor={
                    ft.ControlState.HOVERED: "#F1F7FF",
                    ft.ControlState.DEFAULT: self._theme.card_bg,
                },
                side={
                    ft.ControlState.HOVERED: ft.BorderSide(1, self._theme.accent),
                    ft.ControlState.DEFAULT: ft.BorderSide(1, self._theme.border),
                },
            ),
            on_click=on_click,
            content=ft.Row(
                spacing=16,
                controls=[
                    ft.Container(
                        width=48,
                        height=48,
                        border_radius=self._theme.radius_md,
                        bgcolor=self._theme.accent,
                        alignment=CENTER_ALIGNMENT,
                        content=ft.Icon(icon, color=self._theme.text_light, size=24),
                    ),
                    ft.Column(
                        spacing=4,
                        expand=True,
                        controls=[
                            ft.Text(title, size=15, weight=ft.FontWeight.W_600, color=self._theme.text_dark),
                            ft.Text(subtitle, size=12, color=self._theme.text_muted),
                        ],
                    ),
                    ft.Icon(ft.Icons.CHEVRON_RIGHT, color=self._theme.text_muted),
                ],
            ),
        )

    def _go_back(self) -> None:
        self._close_overlay()
        self._worker.stop()
        self._on_back()

    def _open_mode(self, mode: str) -> None:
        print(f"[nMotion] Opening mode: {mode}")
        self._current_mode = mode
        self._close_overlay()

        if mode == "orientation":
            self._orientation_reference_q = None
            title = "Medir orientación"
            image = ft.Image(src=self._orientation_plot.render())
            self._overlay = PlotOverlay(
                page=self.page,
                title=title,
                image_control=image,
                on_close=self._close_overlay,
                on_record_toggle=self._toggle_recording,
                action_label="Calibrar orientación",
                on_action=self._calibrate_orientation,
            )
        else:
            title = "Medir campo magnético"
            image = ft.Image(src=self._magnetic_plot.render())
            self._overlay = PlotOverlay(
                page=self.page,
                title=title,
                image_control=image,
                on_close=self._close_overlay,
                on_record_toggle=self._toggle_recording,
                action_label="Resetear mediciones",
                on_action=self._reset_magnetic_measurements,
            )

        self._content_host.content = ft.Container(
            expand=True,
            padding=ft.Padding(12, 12, 12, 12),
            content=self._overlay.overlay,
        )
        self.update()
        self._ensure_ui_loop()

        started = self._worker.start()
        print(f"[nMotion] Serial worker started: {started}")
        if not started and self._worker.last_error:
            self._content_host.content = ft.Container(
                expand=True,
                alignment=CENTER_ALIGNMENT,
                content=ft.Text(self._worker.last_error, color=self._theme.danger, size=14),
            )
            self.update()

    def _close_overlay(self) -> None:
        self._overlay = None
        self._latest_imu = None
        self._content_host.content = self._theme.muted("Selecciona un modo de medición para comenzar", size=16)
        self.update()
        self._stop_recording(discard=True)

    def _on_imu_data(self, imu: IMUData) -> None:
        with self._latest_lock:
            self._latest_imu = imu

        if self._recording_start is not None:
            sample = imu.to_dict()
            sample["recorded_at"] = datetime.now(timezone.utc).isoformat()
            self._recording_samples.append(sample)

    def _update_ui(self, imu: IMUData) -> None:
        if self._overlay is None:
            return

        if imu.seq % 50 == 0:
            print(f"[nMotion] IMU frame seq={imu.seq} mode={self._current_mode}")

        if self._current_mode == "orientation":
            qw, qx, qy, qz = imu.quat_q15
            self._latest_orientation_q = (qw, qx, qy, qz)
            if self._orientation_reference_q is not None:
                qw, qx, qy, qz = self._relative_quaternion(self._orientation_reference_q, (qw, qx, qy, qz))
            self._orientation_plot.update(qw, qx, qy, qz)
            b64 = self._orientation_plot.render()
        else:
            self._magnetic_plot.add_point(imu.mx, imu.my, imu.mz)
            self._magnetic_plot.update()
            b64 = self._magnetic_plot.render()

        self._overlay.image.src = b64
        self._overlay.image.update()
        self._content_host.update()

        if self._recording_start is not None:
            elapsed = (time.time() - self._recording_start) * 1000
            self._overlay.set_timer(self._format_duration(elapsed))

    def _ensure_ui_loop(self) -> None:
        if self.page is None or self._ui_loop_started:
            return
        self._ui_loop_started = True
        if hasattr(self.page, "run_task"):
            self.page.run_task(self._ui_loop)

    async def _ui_loop(self) -> None:
        while True:
            imu: IMUData | None = None
            with self._latest_lock:
                if self._latest_imu is not None:
                    imu = self._latest_imu
                    self._latest_imu = None

            if imu is not None and self._overlay is not None:
                self._update_ui(imu)

            await asyncio.sleep(1 / 24)

    def _calibrate_orientation(self) -> None:
        if self._latest_orientation_q is None:
            return
        self._orientation_reference_q = self._latest_orientation_q

    def _reset_magnetic_measurements(self) -> None:
        self._magnetic_plot = MagneticPlot(width=560, height=380, dpi=85)
        if self._overlay is not None:
            self._overlay.image.src = self._magnetic_plot.render()
            self._overlay.image.update()
            self._content_host.update()


    def _toggle_recording(self) -> None:
        if self._recording_start is None:
            self._start_recording()
        else:
            self._stop_recording(discard=False)

    def _start_recording(self) -> None:
        self._recording_samples = []
        self._recording_start = time.time()
        if self._overlay is not None:
            self._overlay.set_recording(True)
            self._overlay.set_timer("00:00.0")

    def _stop_recording(self, discard: bool) -> None:
        if self._recording_start is None:
            return

        self._recording_start = None
        if self._overlay is not None:
            self._overlay.set_recording(False)

        if discard:
            self._recording_samples = []
            return

        if not self._recording_samples:
            return

        # Modal para nombrar
        name_field = ft.TextField(
            label="Nombre de la grabación",
            hint_text="Ej: calibracion_imu_01",
            autofocus=True,
            border_radius=self._theme.radius_md,
        )

        def on_save(_: ft.ControlEvent) -> None:
            name = name_field.value or "recording"
            recording = Recording(mode=self._current_mode or "unknown", samples=self._recording_samples)
            self._recorder.save(recording, name)
            if hasattr(self.page, "close"):
                self.page.close(dialog)
            else:
                dialog.open = False
                self.page.update()
            self._recording_samples = []

        def on_discard(_: ft.ControlEvent) -> None:
            self._recording_samples = []
            if hasattr(self.page, "close"):
                self.page.close(dialog)
            else:
                dialog.open = False
                self.page.update()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Guardar grabación", weight=ft.FontWeight.W_600),
            content=ft.Column(
                tight=True,
                spacing=16,
                controls=[
                    ft.Text("Introduce un nombre para la grabación."),
                    name_field,
                ],
            ),
            actions=[
                self._theme.danger_button("Descartar", on_click=on_discard),
                self._theme.primary_button("Guardar", on_click=on_save),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            bgcolor=self._theme.card_bg,
            shape=ft.RoundedRectangleBorder(radius=self._theme.radius_lg),
        )
        if hasattr(self.page, "open"):
            self.page.open(dialog)
        else:
            self.page.dialog = dialog
            dialog.open = True
            self.page.update()

    @staticmethod
    def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return (q[0], -q[1], -q[2], -q[3])

    @staticmethod
    def _quat_multiply(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    def _relative_quaternion(
        self,
        reference: tuple[float, float, float, float],
        current: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        return self._quat_multiply(self._quat_conjugate(reference), current)

    @staticmethod
    def _format_duration(ms: float) -> str:
        total_seconds = ms / 1000.0
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
