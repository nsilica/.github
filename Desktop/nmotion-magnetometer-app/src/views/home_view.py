"""Pantalla de inicio con logo y botones principales."""

from __future__ import annotations

import flet as ft

from src.theme import CENTER_ALIGNMENT, AppTheme


class HomeView(ft.View):
    """Vista inicial de la app."""

    def __init__(self, on_measure: callable, on_replay: callable) -> None:
        theme = AppTheme()
        super().__init__(
            route="/",
            bgcolor=theme.bg,
            padding=0,
        )
        self._on_measure = on_measure
        self._on_replay = on_replay
        self.controls = self._build(theme)

    def _build(self, theme: AppTheme) -> list[ft.Control]:
        return [
            ft.Container(
                expand=True,
                bgcolor=theme.bg,
                alignment=CENTER_ALIGNMENT,
                content=ft.Column(
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=40,
                    controls=[
                        # Logo provisional
                        ft.Container(
                            width=120,
                            height=120,
                            border_radius=theme.radius_full,
                            bgcolor=theme.accent,
                            alignment=CENTER_ALIGNMENT,
                            content=ft.Icon(
                                ft.Icons.SENSORS,
                                size=64,
                                color=theme.text_light,
                            ),
                        ),
                        theme.headline("nMotion", size=36, color=theme.text_dark),
                        theme.muted("Magnetometer & Orientation Lab", size=16),
                        ft.Container(height=20),
                        ft.Row(
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=24,
                            controls=[
                                theme.primary_button(
                                    "Iniciar medición",
                                    icon=ft.Icons.PLAY_ARROW,
                                    on_click=lambda _: self._on_measure(),
                                ),
                                theme.primary_button(
                                    "Reproducir medición",
                                    icon=ft.Icons.REPLAY,
                                    on_click=lambda _: self._on_replay(),
                                ),
                            ],
                        ),
                    ],
                ),
            ),
        ]
