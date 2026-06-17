"""Ventana modal de gráfico 3D con controles de grabación/reproducción."""

from __future__ import annotations

import flet as ft

from src.theme import CENTER_ALIGNMENT, AppTheme, border_radius_all


class PlotOverlay:
    """Componente visual reutilizable: ventana redondeada oscura con gráfico y controles."""

    def __init__(
        self,
        page: ft.Page,
        title: str,
        image_control: ft.Image,
        on_close: callable,
        on_record_toggle: callable | None = None,
        action_label: str | None = None,
        on_action: callable | None = None,
        is_replay: bool = False,
    ) -> None:
        self.page = page
        self.theme = AppTheme()
        self.is_replay = is_replay
        self._on_close = on_close
        self._on_record_toggle = on_record_toggle
        self._action_label = action_label
        self._on_action = on_action

        self._recording = False
        self._timer_text = ft.Text("00:00.0", size=16, color=self.theme.text_light, weight=ft.FontWeight.W_600)
        self._record_button = self._build_record_button()
        self._play_button: ft.IconButton | None = None
        self._timeline: ft.Slider | None = None
        self._time_label = ft.Text("00:00.0 / 00:00.0", size=12, color=self.theme.text_muted)

        self.image = image_control
        self.image.width = 620
        self.image.height = 420
        self.image.fit = ft.BoxFit.CONTAIN
        self.image.border_radius = border_radius_all(12)
        self.image.gapless_playback = True

        self.overlay = self._build(title)
        self._host: ft.Container | None = None

    def _build_record_button(self) -> ft.IconButton:
        return ft.IconButton(
            icon=ft.Icons.RADIO_BUTTON_UNCHECKED,
            icon_color=self.theme.danger,
            icon_size=28,
            tooltip="Grabar",
            bgcolor=self.theme.text_light,
            on_click=self._toggle_record,
        )

    def _build(self, title: str) -> ft.Container:
        top_controls = [self._record_button, ft.Container(width=8), self._timer_text]

        if self.is_replay:
            self._play_button = ft.IconButton(
                icon=ft.Icons.PAUSE,
                icon_color=self.theme.text_light,
                icon_size=24,
                tooltip="Pausar",
                bgcolor=self.theme.accent,
            )
            self._timeline = ft.Slider(
                min=0,
                max=100,
                value=0,
                divisions=100,
                active_color=self.theme.accent,
                inactive_color=self.theme.text_muted,
                on_change=lambda e: self._on_seek(e.control.value),
            )
            top_controls = [
                self._play_button,
                ft.Container(width=8),
                self._time_label,
                ft.Container(width=16),
                self._timeline,
            ]

        bottom_controls: list[ft.Control] = []
        if self._action_label and self._on_action and not self.is_replay:
            bottom_controls.append(
                ft.Row(
                    alignment=ft.MainAxisAlignment.CENTER,
                    controls=[
                        ft.ElevatedButton(
                            content=self._action_label,
                            bgcolor=self.theme.text_light,
                            color=self.theme.surface_dark,
                            style=ft.ButtonStyle(
                                shape=ft.RoundedRectangleBorder(radius=12),
                                padding=ft.Padding(18, 12, 18, 12),
                                text_style=ft.TextStyle(size=13, weight=ft.FontWeight.W_600),
                            ),
                            on_click=lambda _: self._on_action(),
                        ),
                    ],
                )
            )

        return self.theme.dark_overlay(
            content=ft.Column(
                expand=True,
                spacing=12,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Text(title, size=18, color=self.theme.text_light, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=0,
                                controls=[
                                    *top_controls,
                                    ft.Container(width=16),
                                    ft.IconButton(
                                        icon=ft.Icons.CLOSE,
                                        icon_color=self.theme.text_light,
                                        icon_size=24,
                                        tooltip="Cerrar",
                                        on_click=lambda _: self._on_close(),
                                    ),
                                ],
                            ),
                        ],
                    ),
                    ft.Container(
                        expand=True,
                        bgcolor="#2A2D34",
                        border_radius=12,
                        padding=12,
                        content=ft.Column(
                            expand=True,
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            controls=[self.image],
                        ),
                    ),
                    *bottom_controls,
                ],
            ),
            alignment=CENTER_ALIGNMENT,
            expand=True,
        )

    def _toggle_record(self, _: ft.ControlEvent) -> None:
        if self._on_record_toggle:
            self._on_record_toggle()

    def _on_seek(self, value: float | None) -> None:
        pass

    def set_recording(self, recording: bool) -> None:
        """Actualiza el estado visual del botón de grabar."""
        self._recording = recording
        if recording:
            self._record_button.icon = ft.Icons.RADIO_BUTTON_CHECKED
            self._record_button.icon_color = self.theme.text_light
            self._record_button.bgcolor = self.theme.danger
            self._record_button.tooltip = "Detener grabación"
        else:
            self._record_button.icon = ft.Icons.RADIO_BUTTON_UNCHECKED
            self._record_button.icon_color = self.theme.danger
            self._record_button.bgcolor = self.theme.text_light
            self._record_button.tooltip = "Grabar"
        self._record_button.update()

    def set_timer(self, text: str) -> None:
        self._timer_text.value = text
        self._timer_text.update()

    def set_play_icon(self, playing: bool) -> None:
        if self._play_button is not None:
            self._play_button.icon = ft.Icons.PAUSE if playing else ft.Icons.PLAY_ARROW
            self._play_button.update()

    def set_timeline(self, position: float, duration: float) -> None:
        if self._timeline is not None:
            self._timeline.max = max(duration, 1.0)
            self._timeline.value = position
            self._timeline.update()
        self._time_label.value = f"{self._format_ms(position)} / {self._format_ms(duration)}"
        self._time_label.update()

    @staticmethod
    def _format_ms(ms: float) -> str:
        total_seconds = ms / 1000.0
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def show(self) -> None:
        print("[nMotion] Showing plot overlay")
        self._host = ft.Container(
            expand=True,
            bgcolor="#00000066",
            alignment=CENTER_ALIGNMENT,
            content=self.overlay,
        )
        self.page.overlay.append(self._host)
        self.page.update()

    def close(self) -> None:
        if self._host is None:
            return
        if self._host in self.page.overlay:
            self.page.overlay.remove(self._host)
            self.page.update()
        self._host = None
