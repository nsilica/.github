"""Design tokens y estilos compartidos de la app."""

from __future__ import annotations

import flet as ft

from src.config import AppConfig


# Helpers para compatibilidad con la API moderna de Flet
CENTER_ALIGNMENT = ft.Alignment(0, 0)


def border_radius_all(value: int) -> ft.BorderRadius:
    return ft.BorderRadius(value, value, value, value)


def padding_symmetric(horizontal: int = 0, vertical: int = 0) -> ft.Padding:
    return ft.Padding(horizontal, vertical, horizontal, vertical)


def border_all(width: int = 1, color: str = "#000000") -> ft.Border:
    side = ft.BorderSide(width, color)
    return ft.Border(side, side, side, side)


class AppTheme:
    """Tokens visuales centralizados."""

    def __init__(self) -> None:
        cfg = AppConfig()
        colors = cfg.ui_colors
        self.bg = colors.get("background_color", "#F6FBFF")
        self.accent = colors.get("accent_color", "#1D85ED")
        self.danger = colors.get("danger_color", "#CF2E2E")
        self.success = colors.get("success_color", "#2E7D32")
        self.surface_dark = colors.get("surface_dark", "#2A2D34")
        self.text_dark = colors.get("text_dark", "#404040")
        self.text_light = colors.get("text_light", "#FFFFFF")
        self.text_muted = "#6B7280"
        self.border = "#E5E7EB"
        self.card_bg = "#FFFFFF"
        self.shadow = ft.BoxShadow(
            spread_radius=0,
            blur_radius=12,
            color="#00000014",
            offset=ft.Offset(0, 4),
        )
        self.radius_sm = 8
        self.radius_md = 12
        self.radius_lg = 16
        self.radius_xl = 24
        self.radius_full = 9999

    def page_theme(self) -> ft.Theme:
        """Tema Material adaptado a los colores de la app."""
        return ft.Theme(
            color_scheme=ft.ColorScheme(
                primary=self.accent,
                on_primary=self.text_light,
                secondary=self.accent,
                on_secondary=self.text_light,
                surface=self.bg,
                on_surface=self.text_dark,
                outline=self.border,
            ),
            font_family="Inter",
            use_material3=True,
        )

    def primary_button(self, text: str, icon: str | None = None, **kwargs) -> ft.ElevatedButton:
        return ft.ElevatedButton(
            content=text,
            icon=icon,
            bgcolor=self.accent,
            color=self.text_light,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=self.radius_lg),
                padding=padding_symmetric(horizontal=32, vertical=18),
                text_style=ft.TextStyle(size=16, weight=ft.FontWeight.W_600),
                elevation=2,
            ),
            **kwargs,
        )

    def danger_button(self, text: str, **kwargs) -> ft.ElevatedButton:
        return ft.ElevatedButton(
            content=text,
            bgcolor=self.danger,
            color=self.text_light,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=self.radius_lg),
                padding=padding_symmetric(horizontal=24, vertical=14),
                text_style=ft.TextStyle(size=14, weight=ft.FontWeight.W_600),
            ),
            **kwargs,
        )

    def ghost_button(self, text: str, icon: str | None = None, **kwargs) -> ft.TextButton:
        return ft.TextButton(
            content=text,
            icon=icon,
            style=ft.ButtonStyle(
                color=self.text_dark,
                shape=ft.RoundedRectangleBorder(radius=self.radius_md),
                padding=padding_symmetric(horizontal=20, vertical=12),
                text_style=ft.TextStyle(size=14, weight=ft.FontWeight.W_500),
            ),
            **kwargs,
        )

    def card_container(self, content: ft.Control, **kwargs) -> ft.Container:
        return ft.Container(
            content=content,
            bgcolor=self.card_bg,
            border_radius=self.radius_lg,
            padding=24,
            shadow=self.shadow,
            border=border_all(1, self.border),
            **kwargs,
        )

    def dark_overlay(self, content: ft.Control, **kwargs) -> ft.Container:
        return ft.Container(
            content=content,
            bgcolor=self.surface_dark,
            border_radius=self.radius_xl,
            padding=20,
            shadow=self.shadow,
            **kwargs,
        )

    def headline(self, text: str, size: int = 28, color: str | None = None) -> ft.Text:
        return ft.Text(
            text,
            size=size,
            weight=ft.FontWeight.W_700,
            color=color or self.text_dark,
            font_family="Inter",
        )

    def body(self, text: str, size: int = 14, color: str | None = None) -> ft.Text:
        return ft.Text(
            text,
            size=size,
            weight=ft.FontWeight.W_400,
            color=color or self.text_dark,
        )

    def muted(self, text: str, size: int = 12) -> ft.Text:
        return ft.Text(
            text,
            size=size,
            weight=ft.FontWeight.W_400,
            color=self.text_muted,
        )
