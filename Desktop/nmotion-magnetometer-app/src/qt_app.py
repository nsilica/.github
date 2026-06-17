"""PySide6 application for nMotion Magnetometer."""

from __future__ import annotations

import math
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.calibration_store import CalibrationStore
from src.config import AppConfig
from src.dongle_protocol import IMUData
from src.qt_plots import MagneticCanvas, NorthCanvas, OrientationCanvas, _quat_to_matrix
from src.recorder import Recording, RecordingManager
from src.serial_worker import SerialWorker

# Offset fijo de montaje para el modo Norte (se resta al heading yaw).
MOUNTING_OFFSET_DEG = 90.0


def _format_ms(ms: float) -> str:
    total_seconds = ms / 1000.0
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def _format_recording_ms(ms: float) -> str:
    total_seconds = ms / 1000.0
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def _mode_name(mode: str) -> str:
    if mode == "orientation":
        return "Orientación"
    if mode == "magnetic":
        return "Calibración Hard Iron"
    if mode == "north":
        return "Norte Magnético"
    return mode


def _button(text: str, *, primary: bool = False, danger: bool = False) -> QPushButton:
    btn = QPushButton(text)
    btn.setMinimumHeight(42)
    cls = "primary" if primary else "danger" if danger else "ghost"
    btn.setProperty("class", cls)
    return btn


class HomePage(QWidget):
    def __init__(self, on_measure: callable, on_replay: callable) -> None:
        super().__init__()
        self._on_measure = on_measure
        self._on_replay = on_replay
        root = QHBoxLayout(self)
        root.setContentsMargins(96, 48, 96, 48)
        root.setSpacing(64)

        left = QVBoxLayout()
        left.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        left.setSpacing(20)
        left.addStretch(1)

        logo = QLabel("nMotion")
        logo.setObjectName("homeTitle")
        logo.setAlignment(Qt.AlignmentFlag.AlignLeft)
        subtitle = QLabel("Magnetometer & Orientation Lab")
        subtitle.setObjectName("muted")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft)

        actions = QHBoxLayout()
        actions.setSpacing(16)
        actions.setAlignment(Qt.AlignmentFlag.AlignLeft)
        measure_btn = _button("Iniciar medición", primary=True)
        replay_btn = _button("Reproducir medición", primary=True)
        measure_btn.setMinimumWidth(220)
        replay_btn.setMinimumWidth(220)
        measure_btn.clicked.connect(self._on_measure)
        replay_btn.clicked.connect(self._on_replay)
        actions.addWidget(measure_btn)
        actions.addWidget(replay_btn)
        actions.addStretch(1)

        left.addWidget(logo)
        left.addWidget(subtitle)
        left.addSpacing(12)
        left.addLayout(actions)
        left.addStretch(1)

        right = QVBoxLayout()
        right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label = QLabel()
        pixmap = QPixmap(str(Path(__file__).resolve().parent.parent / "assets" / "app_logo.png"))
        if not pixmap.isNull():
            scaled = pixmap.scaledToHeight(260, Qt.TransformationMode.SmoothTransformation)
            image_label.setPixmap(scaled)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(image_label)

        root.addLayout(left, 3)
        root.addLayout(right, 2)


class MeasurePage(QWidget):
    def __init__(self, on_back: callable) -> None:
        super().__init__()
        self._on_back = on_back
        self._worker = SerialWorker(on_data=self._on_imu_data)
        self._recorder = RecordingManager()
        self._calibration_store = CalibrationStore()
        self._latest_lock = threading.Lock()
        self._recording_lock = threading.Lock()
        self._latest_imu: IMUData | None = None
        self._latest_orientation_q: tuple[float, float, float, float] | None = None
        self._orientation_reference_q: tuple[float, float, float, float] | None = None
        self._mode: str | None = None
        self._canvas: OrientationCanvas | MagneticCanvas | NorthCanvas | None = None
        self._recording_samples: list[dict[str, Any]] = []
        self._recording_start: float | None = None
        self._recording_base_t_ms: int | None = None
        self._magnetic_frozen = False
        self._magnetic_calibration: dict[str, Any] | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

        root = QHBoxLayout(self)
        root.setContentsMargins(36, 36, 36, 36)
        root.setSpacing(28)

        panel = QFrame()
        panel.setObjectName("card")
        panel.setFixedWidth(360)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(16)

        title_row = QHBoxLayout()
        title = QLabel("Medición")
        title.setObjectName("title")
        back_btn = _button("Volver")
        back_btn.clicked.connect(self.go_back)
        title_row.addWidget(title, 1)
        title_row.addWidget(back_btn)

        self._orientation_btn = _button("Medir orientación", primary=True)
        self._magnetic_btn = _button("Calibración Hard Iron", primary=True)
        self._north_btn = _button("Norte Magnético", primary=True)
        self._orientation_btn.clicked.connect(lambda: self.open_mode("orientation"))
        self._magnetic_btn.clicked.connect(lambda: self.open_mode("magnetic"))
        self._north_btn.clicked.connect(lambda: self.open_mode("north"))

        self._action_btn = _button("Calibrar orientación")
        self._action_btn.clicked.connect(self._run_action)
        self._record_btn = _button("Grabar")
        self._record_btn.clicked.connect(self._toggle_recording)
        self._mag_calibration_btn = _button("Guardar calibración magnética")
        self._mag_calibration_btn.clicked.connect(self._save_magnetic_calibration)
        self._timer_label = QLabel("00:00")
        self._timer_label.setObjectName("plotTimer")
        self._status = QLabel("Selecciona un modo de medición para comenzar")
        self._status.setObjectName("muted")
        self._status.setWordWrap(True)

        self._plot_timer_widget = QWidget()
        plot_timer_layout = QHBoxLayout(self._plot_timer_widget)
        plot_timer_layout.setContentsMargins(0, 0, 0, 0)
        plot_timer_layout.addStretch(1)
        plot_timer_layout.addWidget(self._timer_label)

        self._plot_controls_widget = QWidget()
        plot_controls_layout = QHBoxLayout(self._plot_controls_widget)
        plot_controls_layout.setContentsMargins(0, 0, 0, 0)
        plot_controls_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plot_controls_layout.setSpacing(12)
        plot_controls_layout.addWidget(self._action_btn)
        plot_controls_layout.addWidget(self._record_btn)
        plot_controls_layout.addWidget(self._mag_calibration_btn)

        self._kpi_widget = QFrame()
        self._kpi_widget.setObjectName("kpiPanel")
        self._kpi_widget.setFixedWidth(190)
        kpi_layout = QVBoxLayout(self._kpi_widget)
        kpi_layout.setContentsMargins(12, 12, 12, 12)
        kpi_layout.setSpacing(10)
        self._kpi_labels = [QLabel("-") for _ in range(5)]
        for label in self._kpi_labels:
            label.setObjectName("kpi")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            kpi_layout.addWidget(label)
        kpi_layout.addStretch(1)

        self._plot_body = QWidget()
        self._plot_body_layout = QHBoxLayout(self._plot_body)
        self._plot_body_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_body_layout.setSpacing(12)
        # Canvas occupies index 0; KPI panel is fixed on the right.
        self._plot_body_layout.addWidget(self._kpi_widget)

        panel_layout.addLayout(title_row)
        panel_layout.addWidget(QLabel("Visualiza datos IMU en tiempo real."))
        panel_layout.addWidget(self._orientation_btn)
        panel_layout.addWidget(self._magnetic_btn)
        panel_layout.addWidget(self._north_btn)
        panel_layout.addStretch(1)
        panel_layout.addWidget(self._status)

        self._plot_host = QFrame()
        self._plot_host.setObjectName("plotHost")
        self._plot_layout = QVBoxLayout(self._plot_host)
        self._plot_layout.setContentsMargins(12, 12, 12, 12)
        self._plot_placeholder = QLabel("Selecciona un modo para abrir matplotlib embebido")
        self._plot_placeholder.setObjectName("muted")
        self._plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._plot_layout.addWidget(self._plot_placeholder, 1)
        self._plot_layout.addWidget(self._plot_timer_widget)
        self._plot_layout.addWidget(self._plot_body, 1)
        self._plot_layout.addWidget(self._plot_controls_widget)

        root.addWidget(panel)
        root.addWidget(self._plot_host, 1)
        self._set_controls_enabled(False)
        self._update_mode_buttons()

    def open_mode(self, mode: str) -> None:
        if mode == self._mode and self._canvas is not None:
            return
        self._stop_recording(discard=True)
        self._mode = mode
        self._latest_orientation_q = None
        if mode == "magnetic":
            self._magnetic_frozen = False
        if mode == "orientation":
            canvas: OrientationCanvas | MagneticCanvas | NorthCanvas = OrientationCanvas()
        elif mode == "north":
            canvas = NorthCanvas()
        else:
            canvas = MagneticCanvas()
        self._replace_canvas(canvas)
        if mode == "orientation":
            self._orientation_reference_q = self._calibration_store.load_orientation_reference()
            if self._orientation_reference_q:
                self._set_button_danger(self._action_btn, "Resetear orientación", True)
            else:
                self._action_btn.setText("Calibrar orientación")
                self._action_btn.setProperty("class", "ghost")
                self._action_btn.style().unpolish(self._action_btn)
                self._action_btn.style().polish(self._action_btn)
        elif mode == "north":
            self._load_magnetic_calibration()
            action_text = (
                "Calibración cargada" if self._magnetic_calibration is not None else "Cargar calibración"
            )
            self._set_button_success(
                self._action_btn, action_text, self._magnetic_calibration is not None
            )
        else:
            self._set_button_danger(self._action_btn, "Resetear mediciones", True)
        if mode != "north" and mode != "orientation":
            self._action_btn.setText("Resetear mediciones")
        self._mag_calibration_btn.setVisible(mode == "magnetic")
        self._timer_label.setText("00:00")
        self._update_kpis(None)
        self._set_controls_enabled(True)
        self._update_mode_buttons()
        self._status.setText("Conectando al dongle nMotion...")

        if mode == "magnetic":
            self._load_magnetic_calibration()
            if self._magnetic_calibration is not None and isinstance(
                self._canvas, MagneticCanvas
            ):
                self._canvas.load_calibration_state(
                    self._magnetic_calibration.get("center_lsb", [0.0, 0.0, 0.0]),
                    self._magnetic_calibration.get("radius_lsb", 0.0),
                    points_lsb=self._magnetic_calibration.get("points_lsb"),
                    stable=self._magnetic_calibration.get("stable", False),
                )
                self._magnetic_frozen = True
                self._update_kpis(self._magnetic_calibration)
                self._set_button_success(
                    self._mag_calibration_btn, "Calibración guardada", True
                )
                self._status.setText(
                    "Calibración magnética cargada. Medición congelada."
                )
            else:
                self._set_button_success(
                    self._mag_calibration_btn, "Guardar calibración magnética", False
                )
        started = self._worker.start()
        if started:
            self._timer.start()
            if mode == "orientation" and self._orientation_reference_q:
                self._status.setText("Orientación calibrada cargada.")
            elif mode == "magnetic" and self._magnetic_frozen:
                self._status.setText(
                    "Calibración magnética cargada. Medición congelada."
                )
            else:
                self._status.setText("Conectado. Recibiendo datos en vivo.")
        else:
            self._status.setText(self._worker.last_error or "No se pudo iniciar la lectura serie.")

    def ensure_default_mode(self) -> None:
        if self._mode is None:
            self.open_mode("orientation")

    def go_back(self) -> None:
        self._timer.stop()
        self._worker.stop()
        self._stop_recording(discard=True)
        self._replace_canvas(None)
        self._mode = None
        self._set_controls_enabled(False)
        self._update_mode_buttons()
        self._status.setText("Selecciona un modo de medición para comenzar")
        self._on_back()

    def stop(self) -> None:
        self._timer.stop()
        self._worker.stop()

    def _replace_canvas(self, canvas: OrientationCanvas | MagneticCanvas | NorthCanvas | None) -> None:
        # Remove and destroy the previous canvas (matplotlib canvas is single-use here).
        if self._canvas is not None:
            self._plot_body_layout.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
        self._canvas = canvas
        if canvas is None:
            self._plot_placeholder.setVisible(True)
            self._plot_timer_widget.setVisible(False)
            self._plot_body.setVisible(False)
            self._plot_controls_widget.setVisible(False)
        else:
            self._plot_body_layout.insertWidget(0, canvas, 1)
            self._plot_placeholder.setVisible(False)
            self._plot_timer_widget.setVisible(True)
            self._plot_body.setVisible(True)
            self._plot_controls_widget.setVisible(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._action_btn.setEnabled(enabled)
        self._record_btn.setEnabled(enabled)
        self._mag_calibration_btn.setEnabled(enabled)
        self._plot_timer_widget.setVisible(enabled)
        self._plot_controls_widget.setVisible(enabled)
        self._kpi_widget.setVisible(enabled)

    def _update_mode_buttons(self) -> None:
        self._orientation_btn.setProperty(
            "class", "primarySelected" if self._mode == "orientation" else "primary"
        )
        self._magnetic_btn.setProperty(
            "class", "primarySelected" if self._mode == "magnetic" else "primary"
        )
        self._north_btn.setProperty(
            "class", "primarySelected" if self._mode == "north" else "primary"
        )
        for btn in (self._orientation_btn, self._magnetic_btn, self._north_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _set_button_success(self, btn: QPushButton, text: str, success: bool) -> None:
        btn.setText(text)
        btn.setProperty("class", "success" if success else "ghost")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _set_button_danger(self, btn: QPushButton, text: str, danger: bool) -> None:
        btn.setText(text)
        btn.setProperty("class", "danger" if danger else "ghost")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _on_imu_data(self, imu: IMUData) -> None:
        with self._latest_lock:
            self._latest_imu = imu

    def _tick(self) -> None:
        imu: IMUData | None = None
        with self._latest_lock:
            if self._latest_imu is not None:
                imu = self._latest_imu
                self._latest_imu = None
        if imu is None or self._canvas is None:
            return

        if self._mode == "orientation" and isinstance(self._canvas, OrientationCanvas):
            q = imu.quat_q15
            self._latest_orientation_q = q
            if self._orientation_reference_q is not None:
                q = self._relative_quaternion(self._orientation_reference_q, q)
            self._canvas.update_quaternion(*q)
            self._update_kpis(q)
            roll, pitch, yaw = self._quat_to_euler_deg(q)
            self._record_visual_sample(
                imu,
                {
                    "display_q": list(q),
                    "orientation_reference_q": list(self._orientation_reference_q)
                    if self._orientation_reference_q
                    else None,
                    "roll_deg": roll,
                    "pitch_deg": pitch,
                    "yaw_deg": yaw,
                },
            )
        elif self._mode == "magnetic" and isinstance(self._canvas, MagneticCanvas):
            if self._magnetic_frozen:
                return
            self._canvas.add_point(imu.mx, imu.my, imu.mz)
            self._canvas.update_view()
            calibration = self._canvas.calibration()
            self._update_kpis(calibration)
            self._record_visual_sample(imu, {"magnetic_calibration": calibration})
        elif self._mode == "north" and isinstance(self._canvas, NorthCanvas):
            q = imu.quat_q15
            center_lsb = (
                self._magnetic_calibration.get("center_lsb")
                if self._magnetic_calibration is not None
                else None
            )
            drawing_angle, calibrated = MeasurePage._compute_north_drawing_angle_rad(
                imu.mx, imu.my, imu.mz, q, center_lsb=center_lsb
            )
            self._canvas.update_heading(drawing_angle, now_s=time.monotonic())
            self._update_kpis((drawing_angle, calibrated))
            roll, pitch, yaw = self._quat_to_euler_deg(q)
            visual: dict[str, Any] = {
                "display_q": list(q),
                "north_drawing_deg": math.degrees(drawing_angle),
                "roll_deg": roll,
                "pitch_deg": pitch,
                "yaw_deg": yaw,
            }
            if center_lsb is not None:
                visual["magnetic_center_lsb"] = center_lsb
            self._record_visual_sample(imu, visual)

        with self._recording_lock:
            if self._recording_start is not None:
                elapsed = (time.time() - self._recording_start) * 1000
                self._timer_label.setText(_format_recording_ms(elapsed))

    def _run_action(self) -> None:
        if self._mode == "orientation":
            if self._orientation_reference_q is not None:
                # Reset orientation calibration
                self._orientation_reference_q = None
                self._calibration_store.delete_orientation_reference()
                self._action_btn.setText("Calibrar orientación")
                self._action_btn.setProperty("class", "ghost")
                self._action_btn.style().unpolish(self._action_btn)
                self._action_btn.style().polish(self._action_btn)
                self._update_kpis(None)
                self._status.setText("Orientación reseteada.")
            elif self._latest_orientation_q is not None:
                self._orientation_reference_q = self._latest_orientation_q
                self._calibration_store.save_orientation_reference(self._latest_orientation_q)
                self._set_button_danger(self._action_btn, "Resetear orientación", True)
                self._update_kpis(self._latest_orientation_q)
                self._status.setText("Orientación calibrada contra la pose actual.")
        elif self._mode == "north":
            self._load_magnetic_calibration()
            loaded = self._magnetic_calibration is not None
            self._set_button_success(
                self._action_btn,
                "Calibración cargada" if loaded else "Cargar calibración",
                loaded,
            )
        elif isinstance(self._canvas, MagneticCanvas):
            self._magnetic_frozen = False
            self._magnetic_calibration = None
            self._calibration_store.delete_magnetic_calibration()
            self._set_button_success(
                self._mag_calibration_btn, "Guardar calibración magnética", False
            )
            self._canvas.clear()
            self._update_kpis(None)
            self._status.setText("Mediciones magnéticas reseteadas.")

    def _load_magnetic_calibration(self) -> None:
        self._magnetic_calibration = self._calibration_store.load_magnetic_calibration()
        if self._magnetic_calibration is None:
            self._status.setText("No hay calibración magnética guardada.")
        else:
            self._status.setText("Calibración magnética cargada.")

    def _save_magnetic_calibration(self) -> None:
        if not isinstance(self._canvas, MagneticCanvas):
            return
        calibration = self._canvas.calibration()
        if calibration is None:
            self._status.setText("No hay ajuste de esfera suficiente para guardar calibración.")
            return
        points = list(
            zip(
                list(self._canvas._points["x"]),
                list(self._canvas._points["y"]),
                list(self._canvas._points["z"]),
                strict=True,
            )
        )
        max_points = 1000
        if len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = [points[i] for i in indices]
        path = self._calibration_store.save_magnetic_calibration(calibration, points)
        self._magnetic_frozen = True
        self._set_button_success(self._mag_calibration_btn, "Calibración guardada", True)
        self._status.setText(f"Calibración magnética guardada: {path.name}")

    def _record_visual_sample(self, imu: IMUData, visual: dict[str, Any]) -> None:
        with self._recording_lock:
            if self._recording_start is None:
                return
            if self._recording_base_t_ms is None:
                self._recording_base_t_ms = imu.t_global_ms
            sample = imu.to_dict()
            sample["device_t_ms"] = sample["t_ms"]
            sample["t_ms"] = max(0, imu.t_global_ms - self._recording_base_t_ms)
            sample["recorded_at"] = datetime.now(UTC).isoformat()
            sample["visual"] = visual
            self._recording_samples.append(sample)

    def _update_kpis(self, data: object | None) -> None:
        if self._mode == "orientation":
            if not isinstance(data, tuple):
                values = (
                    ("Roll -", "neutral"),
                    ("Pitch -", "neutral"),
                    ("Yaw -", "neutral"),
                    (
                        "Calibrado" if self._orientation_reference_q else "Sin calibrar",
                        "stable" if self._orientation_reference_q else "unstable",
                    ),
                )
            else:
                roll, pitch, yaw = self._quat_to_euler_deg(data)
                values = (
                    (f"Roll {roll:+.1f}°", "neutral"),
                    (f"Pitch {pitch:+.1f}°", "neutral"),
                    (f"Yaw {yaw:+.1f}°", "neutral"),
                    (
                        "Calibrado" if self._orientation_reference_q else "Sin calibrar",
                        "stable" if self._orientation_reference_q else "unstable",
                    ),
                )
        elif self._mode == "magnetic":
            if not isinstance(data, dict):
                values = (
                    ("Cx -", "neutral"),
                    ("Cy -", "neutral"),
                    ("Cz -", "neutral"),
                    ("R -", "neutral"),
                    ("Inestable", "unstable"),
                )
            else:
                center = data.get("center_lsb", [0.0, 0.0, 0.0])
                radius = float(data.get("radius_lsb", 0.0))
                stable = bool(data.get("stable", False))
                variation = data.get("stability_variation_lsb")
                stability_text = "Estable" if stable else "Inestable"
                if isinstance(variation, float):
                    stability_text = f"{stability_text} Δ {variation:.1f}"
                values = (
                    (f"Cx {float(center[0]):+.1f}", "neutral"),
                    (f"Cy {float(center[1]):+.1f}", "neutral"),
                    (f"Cz {float(center[2]):+.1f}", "neutral"),
                    (f"R {radius:.1f}", "neutral"),
                    (stability_text, "stable" if stable else "unstable"),
                )
        elif self._mode == "north":
            if not isinstance(data, tuple) or len(data) != 2:
                values = (("Norte -", "neutral"),)
            else:
                drawing_angle, calibrated = data
                yaw_deg = math.degrees(drawing_angle)
                values = ((f"Norte {yaw_deg:+.1f}°", "stable" if calibrated else "unstable"),)
        else:
            values = (("-", "neutral"),)
        for idx, label in enumerate(self._kpi_labels):
            if idx < len(values):
                value, state = values[idx]
                label.setText(value)
                label.setProperty("state", state)
                label.setVisible(True)
            else:
                label.setVisible(False)
            label.style().unpolish(label)
            label.style().polish(label)

    @staticmethod
    def _quat_to_euler_deg(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
        qw, qx, qy, qz = q
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (qw * qy - qz * qx)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    @staticmethod
    def _compute_north_drawing_angle_rad(
        mx: float,
        my: float,
        mz: float,
        q: tuple[float, float, float, float],
        center_lsb: list[float] | None = None,
    ) -> tuple[float, bool]:
        """Return (drawing_angle_rad, calibrated) for the north compass.

        The magnetometer reading is rotated into the global/earth frame using
        the quaternion, so the resulting heading is independent of the sensor
        yaw. The drawing angle is that earth-frame heading plus the fixed
        mounting offset. Negative = clockwise from the triangle, positive =
        counter-clockwise.
        """
        calibrated = center_lsb is not None
        if calibrated:
            mx -= float(center_lsb[0])
            my -= float(center_lsb[1])
            mz -= float(center_lsb[2])

        m_corr = np.array([mx, my, mz], dtype=float)
        r = _quat_to_matrix(*q)
        m_earth = r @ m_corr
        heading_mag = math.atan2(m_earth[1], m_earth[0])

        drawing_angle = -(heading_mag + math.radians(MOUNTING_OFFSET_DEG))
        # Normalize to [-pi, pi]
        drawing_angle = (drawing_angle + math.pi) % (2 * math.pi) - math.pi
        return drawing_angle, calibrated

    def _toggle_recording(self) -> None:
        with self._recording_lock:
            recording = self._recording_start is not None
        if recording:
            self._stop_recording(discard=False)
        else:
            with self._recording_lock:
                self._recording_samples = []
                self._recording_start = time.time()
                self._recording_base_t_ms = None
            self._record_btn.setText("Detener grabación")
            self._record_btn.setProperty("class", "danger")
            self._record_btn.style().unpolish(self._record_btn)
            self._record_btn.style().polish(self._record_btn)
            self._timer_label.setText("00:00")

    def _stop_recording(self, *, discard: bool) -> None:
        with self._recording_lock:
            if self._recording_start is None:
                return
            samples = list(self._recording_samples)
            self._recording_samples = []
            self._recording_start = None
            self._recording_base_t_ms = None
        self._record_btn.setText("Grabar")
        self._record_btn.setProperty("class", "ghost")
        self._record_btn.style().unpolish(self._record_btn)
        self._record_btn.style().polish(self._record_btn)
        if discard or not samples:
            return

        name, ok = QInputDialog.getText(self, "Guardar grabación", "Nombre de la grabación:")
        if ok:
            recording = Recording(mode=self._mode or "unknown", samples=samples)
            path = self._recorder.save(recording, name or "recording")
            self._status.setText(f"Grabación guardada: {path.name}")

    @staticmethod
    def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return (q[0], -q[1], -q[2], -q[3])

    @staticmethod
    def _quat_multiply(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
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
        self, reference: tuple[float, float, float, float], current: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        return self._quat_multiply(self._quat_conjugate(reference), current)


class ReplayPage(QWidget):
    def __init__(self, on_back: callable) -> None:
        super().__init__()
        self._on_back = on_back
        self._recorder = RecordingManager()
        self._recording: Recording | None = None
        self._canvas: OrientationCanvas | MagneticCanvas | NorthCanvas | None = None
        self._seeking = False
        self._was_playing_before_seek = False
        self._playing = False
        self._play_started_at = 0.0
        self._play_start_pos = 0.0
        self._position_ms = 0.0
        self._last_pos = 0.0
        self._last_sample_t = -1

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._sync_timeline)

        root = QHBoxLayout(self)
        root.setContentsMargins(36, 36, 36, 36)
        root.setSpacing(28)

        panel = QFrame()
        panel.setObjectName("card")
        panel.setFixedWidth(360)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(12)

        title_row = QHBoxLayout()
        title = QLabel("Reproducción")
        title.setObjectName("title")
        back_btn = _button("Volver")
        back_btn.clicked.connect(self.go_back)
        title_row.addWidget(title, 1)
        title_row.addWidget(back_btn)

        self._list = QListWidget()
        self._list.itemSelectionChanged.connect(self._load_selected)
        delete_btn = _button("Eliminar", danger=True)
        refresh_btn = _button("Refrescar")
        delete_btn.clicked.connect(self._delete_selected)
        refresh_btn.clicked.connect(self.refresh)

        panel_layout.addLayout(title_row)
        panel_layout.addWidget(QLabel("Selecciona una grabación guardada."))
        panel_layout.addWidget(self._list, 1)
        panel_layout.addWidget(delete_btn)
        panel_layout.addWidget(refresh_btn)

        right = QVBoxLayout()
        self._plot_host = QFrame()
        self._plot_host.setObjectName("plotHost")
        self._plot_layout = QVBoxLayout(self._plot_host)
        self._plot_layout.setContentsMargins(12, 12, 12, 12)
        self._replay_body = QWidget()
        self._replay_body_layout = QHBoxLayout(self._replay_body)
        self._replay_body_layout.setContentsMargins(0, 0, 0, 0)
        self._replay_body_layout.setSpacing(12)
        self._replay_kpi_widget = QFrame()
        self._replay_kpi_widget.setObjectName("kpiPanel")
        self._replay_kpi_widget.setFixedWidth(190)
        replay_kpi_layout = QVBoxLayout(self._replay_kpi_widget)
        replay_kpi_layout.setContentsMargins(12, 12, 12, 12)
        replay_kpi_layout.setSpacing(10)
        self._replay_kpi_labels = [QLabel("-") for _ in range(5)]
        for label in self._replay_kpi_labels:
            label.setObjectName("kpi")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            replay_kpi_layout.addWidget(label)
        replay_kpi_layout.addStretch(1)
        # Canvas is inserted at index 0; KPI panel is fixed on the right.
        self._replay_body_layout.addWidget(self._replay_kpi_widget)

        self._replay_placeholder = QLabel("Selecciona un modo para abrir matplotlib embebido")
        self._replay_placeholder.setObjectName("muted")
        self._replay_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._plot_layout.addWidget(self._replay_placeholder, 1)
        self._plot_layout.addWidget(self._replay_body, 1)
        self._replace_canvas(None)

        controls_frame = QFrame()
        controls_frame.setObjectName("replayControls")
        controls = QHBoxLayout(controls_frame)
        controls.setContentsMargins(14, 10, 14, 10)
        controls.setSpacing(14)
        self._play_btn = _button("Play", primary=True)
        self._play_btn.clicked.connect(self._toggle_play)
        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setObjectName("muted")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimumHeight(38)
        self._slider.setMinimum(0)
        self._slider.setMaximum(1)
        self._slider.sliderPressed.connect(self._begin_seek)
        self._slider.sliderMoved.connect(self._seek_preview)
        self._slider.sliderReleased.connect(self._end_seek)
        controls.addWidget(self._play_btn)
        controls.addWidget(self._time_label)
        controls.addWidget(self._slider, 1)

        right.addWidget(self._plot_host, 1)
        right.addWidget(controls_frame)

        root.addWidget(panel)
        root.addLayout(right, 1)
        self.refresh()
        self._set_player_enabled(False)

    def refresh(self) -> None:
        self._list.clear()
        for path in self._recorder.list_recordings():
            try:
                rec = self._recorder.load(path)
            except Exception:  # noqa: BLE001
                continue
            item = QListWidgetItem(
                f"{path.stem}\nModo: {_mode_name(rec.header.mode)} | {rec.header.samples_count} muestras | "
                f"{_format_ms(rec.header.duration_ms)}"
            )
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._list.addItem(item)

    def go_back(self) -> None:
        self.stop()
        self._on_back()

    def stop(self) -> None:
        self._timer.stop()
        self._playing = False
        self._position_ms = 0.0

    def _set_player_enabled(self, enabled: bool) -> None:
        self._play_btn.setEnabled(enabled)
        self._slider.setEnabled(enabled)

    def _replace_canvas(self, canvas: OrientationCanvas | MagneticCanvas | NorthCanvas | None) -> None:
        if self._canvas is not None:
            self._replay_body_layout.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
        self._canvas = canvas
        if canvas is None:
            self._replay_placeholder.setVisible(True)
            self._replay_body.setVisible(False)
        else:
            self._replay_body_layout.insertWidget(0, canvas, 1)
            self._replay_placeholder.setVisible(False)
            self._replay_body.setVisible(True)

    def _load_selected(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            self._load_item(item)

    def _load_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(path, Path):
            return
        self._playing = False
        self._timer.stop()
        self._recording = self._recorder.load(path)
        self._normalize_loaded_recording()
        self._last_pos = 0.0
        self._last_sample_t = -1
        self._position_ms = 0.0
        if self._recording.header.mode == "orientation":
            canvas: OrientationCanvas | MagneticCanvas | NorthCanvas = OrientationCanvas()
        elif self._recording.header.mode == "north":
            canvas = NorthCanvas()
        else:
            canvas = MagneticCanvas()
        self._replace_canvas(canvas)
        self._apply_recorded_calibration_to_canvas()
        self._slider.setMaximum(max(int(self._recording.header.duration_ms), 1))
        self._slider.setValue(0)
        self._time_label.setText(f"00:00 / {_format_ms(self._recording.header.duration_ms)}")
        self._render_position(0.0, force_rebuild=True)
        self._play_start_pos = 0.0
        self._play_started_at = time.monotonic()
        self._play_btn.setText("Play")
        self._set_player_enabled(True)

    def _toggle_play(self) -> None:
        if self._recording is None:
            return
        if self._playing:
            self._position_ms = self._current_replay_position()
            self._playing = False
            self._play_btn.setText("Play")
        else:
            self._play_start_pos = self._position_ms
            self._play_started_at = time.monotonic()
            self._playing = True
            self._timer.start()
            self._play_btn.setText("Pausa")

    def _begin_seek(self) -> None:
        self._seeking = True
        self._was_playing_before_seek = self._playing
        self._position_ms = self._current_replay_position()
        self._playing = False

    def _seek_preview(self, position: int) -> None:
        if self._recording is None:
            return
        pos = float(position)
        self._position_ms = pos
        self._time_label.setText(f"{_format_ms(pos)} / {_format_ms(self._duration_ms())}")
        self._render_position(pos, force_rebuild=True)

    def _end_seek(self) -> None:
        self._seeking = False
        pos = float(self._slider.value())
        self._position_ms = pos
        self._seek_preview(int(pos))
        if self._was_playing_before_seek:
            self._play_start_pos = self._position_ms
            self._play_started_at = time.monotonic()
            self._playing = True
            self._timer.start()
            self._play_btn.setText("Pausa")
        else:
            self._play_btn.setText("Play")

    def _sync_timeline(self) -> None:
        if self._recording is None or self._seeking:
            return
        pos = self._current_replay_position()
        duration = self._duration_ms()
        if pos < self._last_pos:
            if isinstance(self._canvas, MagneticCanvas):
                self._canvas.clear()
                self._last_sample_t = -1
            elif isinstance(self._canvas, NorthCanvas):
                self._canvas.reset_variation()
        self._last_pos = pos
        self._slider.setValue(int(pos))
        self._time_label.setText(f"{_format_ms(pos)} / {_format_ms(duration)}")
        self._render_position(pos, force_rebuild=False)

    def _update_frame(self, sample: dict[str, Any]) -> None:
        if self._recording is None or self._canvas is None:
            return
        if self._recording.header.mode == "orientation" and isinstance(self._canvas, OrientationCanvas):
            q = self._sample_quaternion(sample)
            self._canvas.update_quaternion(q[0], q[1], q[2], q[3])
        elif self._recording.header.mode == "north" and isinstance(self._canvas, NorthCanvas):
            q = self._sample_quaternion(sample)
            mag = sample.get("raw", {}).get("mag", [0, 0, 0])
            cal = sample.get("visual", {}).get("magnetic_calibration")
            center_lsb = cal.get("center_lsb") if isinstance(cal, dict) else None
            drawing_angle, _calibrated = MeasurePage._compute_north_drawing_angle_rad(
                float(mag[0]), float(mag[1]), float(mag[2]), tuple(q), center_lsb=center_lsb
            )
            now_s = float(sample.get("t_ms", 0)) / 1000.0
            self._canvas.update_heading(drawing_angle, now_s=now_s)
        elif isinstance(self._canvas, MagneticCanvas):
            t_ms = int(sample.get("t_ms", 0))
            if t_ms < self._last_sample_t:
                self._canvas.clear()
            self._last_sample_t = t_ms
            mag = sample.get("raw", {}).get("mag", [0, 0, 0])
            self._canvas.add_point(mag[0], mag[1], mag[2])
            cal = sample.get("visual", {}).get("magnetic_calibration")
            if isinstance(cal, dict):
                self._canvas.set_calibration(
                    cal.get("center_lsb", [0.0, 0.0, 0.0]),
                    cal.get("radius_lsb", 0.0),
                    stable=cal.get("stable", False),
                    draw=False,
                )
            self._canvas.render_state()

    def _render_position(self, position_ms: float, *, force_rebuild: bool) -> None:
        if self._recording is None or self._canvas is None:
            return
        if self._recording.header.mode == "orientation":
            q = self._orientation_at(position_ms)
            if q is not None and isinstance(self._canvas, OrientationCanvas):
                self._canvas.update_quaternion(q[0], q[1], q[2], q[3])
                self._update_replay_kpis("orientation", q)
            return
        if self._recording.header.mode == "north":
            if force_rebuild:
                self._rebuild_north_until(position_ms)
                return
            sample = self._sample_at(position_ms)
            if sample is None:
                return
            q = self._sample_quaternion(sample)
            mag = sample.get("raw", {}).get("mag", [0, 0, 0])
            cal = sample.get("visual", {}).get("magnetic_calibration")
            center_lsb = cal.get("center_lsb") if isinstance(cal, dict) else None
            drawing_angle, _calibrated = MeasurePage._compute_north_drawing_angle_rad(
                float(mag[0]), float(mag[1]), float(mag[2]), tuple(q), center_lsb=center_lsb
            )
            now_s = float(sample.get("t_ms", 0)) / 1000.0
            if isinstance(self._canvas, NorthCanvas):
                self._canvas.update_heading(drawing_angle, now_s=now_s)
            self._update_replay_kpis("north", drawing_angle)
            return
        if isinstance(self._canvas, MagneticCanvas):
            if force_rebuild:
                self._rebuild_magnetic_until(position_ms)
                return
            last_cal: dict[str, Any] | None = None
            for sample in self._recording.samples:
                t_ms = int(sample.get("t_ms", 0))
                if t_ms <= self._last_sample_t:
                    continue
                if t_ms > position_ms:
                    break
                mag = sample.get("raw", {}).get("mag", [0, 0, 0])
                self._canvas.add_point(mag[0], mag[1], mag[2])
                self._last_sample_t = t_ms
                cal = sample.get("visual", {}).get("magnetic_calibration")
                if isinstance(cal, dict):
                    last_cal = cal
            if last_cal is not None:
                self._canvas.set_calibration(
                    last_cal.get("center_lsb", [0.0, 0.0, 0.0]),
                    last_cal.get("radius_lsb", 0.0),
                    stable=last_cal.get("stable", False),
                    draw=False,
                )
            self._canvas.render_state()
            self._update_replay_kpis("magnetic", self._canvas.calibration())

    def _rebuild_magnetic_until(self, position_ms: float) -> None:
        if self._recording is None or not isinstance(self._canvas, MagneticCanvas):
            return
        self._canvas.clear()
        self._last_sample_t = -1
        last_cal: dict[str, Any] | None = None
        for sample in self._recording.samples:
            if float(sample.get("t_ms", 0)) > position_ms:
                break
            mag = sample.get("raw", {}).get("mag", [0, 0, 0])
            self._canvas.add_point(mag[0], mag[1], mag[2])
            self._last_sample_t = int(sample.get("t_ms", 0))
            cal = sample.get("visual", {}).get("magnetic_calibration")
            if isinstance(cal, dict):
                last_cal = cal
        if last_cal is not None:
            self._canvas.set_calibration(
                last_cal.get("center_lsb", [0.0, 0.0, 0.0]),
                last_cal.get("radius_lsb", 0.0),
                stable=last_cal.get("stable", False),
                draw=False,
            )
        self._canvas.render_state()
        self._update_replay_kpis("magnetic", self._canvas.calibration())

    def _rebuild_north_until(self, position_ms: float) -> None:
        if self._recording is None or not isinstance(self._canvas, NorthCanvas):
            return
        self._canvas.reset_variation()
        self._last_sample_t = -1
        for sample in self._recording.samples:
            t_ms = float(sample.get("t_ms", 0))
            if t_ms > position_ms:
                break
            q = self._sample_quaternion(sample)
            mag = sample.get("raw", {}).get("mag", [0, 0, 0])
            cal = sample.get("visual", {}).get("magnetic_calibration")
            center_lsb = cal.get("center_lsb") if isinstance(cal, dict) else None
            drawing_angle, _calibrated = MeasurePage._compute_north_drawing_angle_rad(
                float(mag[0]), float(mag[1]), float(mag[2]), tuple(q), center_lsb=center_lsb
            )
            now_s = t_ms / 1000.0
            self._canvas.update_heading(drawing_angle, now_s=now_s)
            self._last_sample_t = int(t_ms)
        self._update_replay_kpis("north", drawing_angle)

    def _sample_at(self, position_ms: float) -> dict[str, Any] | None:
        if self._recording is None or not self._recording.samples:
            return None
        best = self._recording.samples[0]
        best_diff = abs(float(best.get("t_ms", 0)) - position_ms)
        for sample in self._recording.samples[1:]:
            diff = abs(float(sample.get("t_ms", 0)) - position_ms)
            if diff < best_diff:
                best_diff = diff
                best = sample
            else:
                break
        return best

    def _duration_ms(self) -> float:
        if self._recording is None:
            return 0.0
        return float(
            self._recording.header.duration_ms
            or (self._recording.samples[-1].get("t_ms", 0) if self._recording.samples else 0)
        )

    def _current_replay_position(self) -> float:
        if self._recording is None:
            return 0.0
        if not self._playing:
            return self._position_ms
        duration = self._duration_ms()
        pos = self._play_start_pos + (time.monotonic() - self._play_started_at) * 1000.0
        if duration <= 0:
            self._position_ms = 0.0
            return 0.0
        if pos >= duration:
            pos = pos % duration
            self._play_start_pos = pos
            self._play_started_at = time.monotonic()
        self._position_ms = pos
        return pos

    def _orientation_at(self, position_ms: float) -> list[float] | None:
        if self._recording is None or not self._recording.samples:
            return None
        previous = self._recording.samples[0]
        for sample in self._recording.samples[1:]:
            sample_t = float(sample.get("t_ms", 0))
            if sample_t >= position_ms:
                prev_t = float(previous.get("t_ms", 0))
                if sample_t <= prev_t:
                    return self._sample_quaternion(sample)
                alpha = max(0.0, min((position_ms - prev_t) / (sample_t - prev_t), 1.0))
                return self._lerp_quaternion(
                    self._sample_quaternion(previous), self._sample_quaternion(sample), alpha
                )
            previous = sample
        return self._sample_quaternion(previous)

    @staticmethod
    def _sample_quaternion(sample: dict[str, Any]) -> list[float]:
        visual_q = sample.get("visual", {}).get("display_q")
        if visual_q and len(visual_q) == 4:
            return [float(visual_q[0]), float(visual_q[1]), float(visual_q[2]), float(visual_q[3])]
        q = sample.get("outputs", {}).get("q")
        if q and len(q) == 4:
            return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        raw_q = sample.get("raw", {}).get("q", [32768, 0, 0, 0])
        return [
            float(raw_q[0]) / 32768.0,
            -float(raw_q[1]) / 32768.0,
            -float(raw_q[2]) / 32768.0,
            float(raw_q[3]) / 32768.0,
        ]

    @staticmethod
    def _lerp_quaternion(a: list[float], b: list[float], alpha: float) -> list[float]:
        if sum(x * y for x, y in zip(a, b, strict=True)) < 0:
            b = [-x for x in b]
        q = [(1.0 - alpha) * x + alpha * y for x, y in zip(a, b, strict=True)]
        norm = math.sqrt(sum(x * x for x in q))
        if norm < 1e-9:
            return [1.0, 0.0, 0.0, 0.0]
        return [x / norm for x in q]

    def _normalize_loaded_recording(self) -> None:
        if self._recording is None or not self._recording.samples:
            return
        self._recording.samples.sort(key=lambda sample: float(sample.get("t_ms", 0)))
        first_t = float(self._recording.samples[0].get("t_ms", 0))
        if first_t != 0:
            for sample in self._recording.samples:
                original_t = float(sample.get("t_ms", 0))
                sample.setdefault("device_t_ms", original_t)
                sample["t_ms"] = max(0, original_t - first_t)
        self._recording.header.samples_count = len(self._recording.samples)
        self._recording.header.duration_ms = int(self._recording.samples[-1].get("t_ms", 0))

    def _apply_recorded_calibration_to_canvas(self) -> None:
        if self._recording is None or not self._recording.samples or self._canvas is None:
            return
        if isinstance(self._canvas, MagneticCanvas) and self._recording.header.mode == "magnetic":
            for sample in reversed(self._recording.samples):
                cal = sample.get("visual", {}).get("magnetic_calibration")
                if isinstance(cal, dict) and cal.get("radius_lsb", 0.0) > 0.0:
                    self._canvas.set_calibration(
                        cal.get("center_lsb", [0.0, 0.0, 0.0]),
                        cal.get("radius_lsb", 0.0),
                        stable=cal.get("stable", False),
                        draw=False,
                    )
                    break

    def _update_replay_kpis(self, mode: str, data: object | None) -> None:
        if mode == "orientation":
            if not isinstance(data, list):
                values = (("Roll -", "neutral"), ("Pitch -", "neutral"), ("Yaw -", "neutral"))
            else:
                roll, pitch, yaw = MeasurePage._quat_to_euler_deg(tuple(data))
                values = (
                    (f"Roll {roll:+.1f}°", "neutral"),
                    (f"Pitch {pitch:+.1f}°", "neutral"),
                    (f"Yaw {yaw:+.1f}°", "neutral"),
                )
        elif mode == "north":
            if isinstance(data, float):
                yaw_deg = math.degrees(data)
                values = ((f"Norte {yaw_deg:+.1f}°", "neutral"),)
            else:
                values = (("Norte -", "neutral"),)
        elif mode == "magnetic" and isinstance(data, dict):
            center = data.get("center_lsb", [0.0, 0.0, 0.0])
            radius = float(data.get("radius_lsb", 0.0))
            stable = bool(data.get("stable", False))
            variation = data.get("stability_variation_lsb")
            stability_text = "Estable" if stable else "Inestable"
            if isinstance(variation, float):
                stability_text = f"{stability_text} Δ {variation:.1f}"
            values = (
                (f"Cx {float(center[0]):+.1f}", "neutral"),
                (f"Cy {float(center[1]):+.1f}", "neutral"),
                (f"Cz {float(center[2]):+.1f}", "neutral"),
                (f"R {radius:.1f}", "neutral"),
                (stability_text, "stable" if stable else "unstable"),
            )
        elif mode == "magnetic":
            values = (
                ("Cx -", "neutral"),
                ("Cy -", "neutral"),
                ("Cz -", "neutral"),
                ("R -", "neutral"),
                ("Inestable", "unstable"),
            )
        else:
            values = (("-", "neutral"),)
        for idx, label in enumerate(self._replay_kpi_labels):
            if idx < len(values):
                value, state = values[idx]
                label.setText(value)
                label.setProperty("state", state)
                label.setVisible(True)
            else:
                label.setVisible(False)
            label.style().unpolish(label)
            label.style().polish(label)

    def _reset_loaded_recording(self) -> None:
        self._timer.stop()
        self._playing = False
        self._recording = None
        self._position_ms = 0.0
        self._last_pos = 0.0
        self._last_sample_t = -1
        self._slider.setValue(0)
        self._slider.setMaximum(1)
        self._time_label.setText("00:00 / 00:00")
        self._play_btn.setText("Play")
        self._set_player_enabled(False)
        self._replace_canvas(None)
        self._update_replay_kpis("", None)

    def _delete_selected(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(path, Path):
            return
        reply = QMessageBox.question(self, "Eliminar", f"¿Eliminar {path.name}?")
        if reply == QMessageBox.StandardButton.Yes:
            self._recorder.delete(path)
            self._reset_loaded_recording()
            self.refresh()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        cfg = AppConfig()
        self.setWindowTitle(cfg.app_title)
        self.resize(cfg.window_width, cfg.window_height)
        self.setMinimumSize(980, 640)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self._home = HomePage(self.show_measure, self.show_replay)
        self._measure = MeasurePage(self.show_home)
        self._replay = ReplayPage(self.show_home)
        self._stack.addWidget(self._home)
        self._stack.addWidget(self._measure)
        self._stack.addWidget(self._replay)
        self.show_home()

    def show_home(self) -> None:
        self._stack.setCurrentWidget(self._home)

    def show_measure(self) -> None:
        self._measure.ensure_default_mode()
        self._stack.setCurrentWidget(self._measure)

    def show_replay(self) -> None:
        self._replay.refresh()
        self._stack.setCurrentWidget(self._replay)

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802
        self._measure.stop()
        self._replay.stop()
        super().closeEvent(event)


def _stylesheet() -> str:
    return """
    QWidget {
        background: #F6FBFF;
        color: #404040;
        font-family: Inter, Segoe UI, Arial;
        font-size: 13px;
    }
    QLabel {
        background: transparent;
    }
    QLabel#homeTitle {
        font-size: 40px;
        font-weight: 800;
        color: #404040;
    }
    QLabel#title {
        font-size: 24px;
        font-weight: 700;
    }
    QLabel#muted {
        color: #6B7280;
    }
    QLabel#plotTimer {
        color: #FFFFFF;
        background: #00000055;
        border-radius: 10px;
        padding: 6px 10px;
        font-size: 15px;
        font-weight: 700;
    }
    QLabel#kpi {
        color: #FFFFFF;
        background: #1F4F7A;
        border: none;
        border-radius: 10px;
        padding: 18px 12px;
        font-weight: 700;
        min-height: 46px;
    }
    QLabel#kpi[state="stable"] {
        background: #2E7D32;
        border: none;
        color: #FFFFFF;
    }
    QLabel#kpi[state="unstable"] {
        background: #8E2424;
        border: none;
        color: #FFFFFF;
    }
    QFrame#card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
    }
    QFrame#plotHost {
        background: #2A2D34;
        border-radius: 20px;
    }
    QFrame#plotHost QWidget {
        background: transparent;
    }
    QFrame#plotHost QLabel#kpi {
        background: #1F4F7A;
        border: none;
        border-radius: 10px;
    }
    QFrame#plotHost QLabel#kpi[state="stable"] {
        background: #2E7D32;
        border: none;
    }
    QFrame#plotHost QLabel#kpi[state="unstable"] {
        background: #8E2424;
        border: none;
    }
    QFrame#kpiPanel {
        background: transparent;
        border: none;
        border-radius: 14px;
    }
    QFrame#replayControls {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
    }
    QPushButton {
        border-radius: 12px;
        padding: 10px 18px;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        color: #404040;
        font-weight: 600;
    }
    QPushButton[class="primary"] {
        background: #1D85ED;
        color: #FFFFFF;
        border: 1px solid #1D85ED;
    }
    QPushButton[class="primarySelected"] {
        background: #176FC8;
        color: #FFFFFF;
        border: 1px solid #176FC8;
    }
    QPushButton[class="danger"] {
        background: #CF2E2E;
        color: #FFFFFF;
        border: 1px solid #CF2E2E;
    }
    QPushButton[class="success"] {
        background: #2E7D32;
        color: #FFFFFF;
        border: 1px solid #2E7D32;
    }
    QFrame#plotHost QPushButton[class="ghost"] {
        background: #FFFFFF;
        color: #2A2D34;
        border: 1px solid #FFFFFF;
    }
    QFrame#plotHost QPushButton[class="primary"] {
        background: #1D85ED;
        color: #FFFFFF;
        border: 1px solid #1D85ED;
    }
    QFrame#plotHost QPushButton[class="danger"] {
        background: #CF2E2E;
        color: #FFFFFF;
        border: 1px solid #CF2E2E;
    }
    QFrame#plotHost QPushButton[class="success"] {
        background: #2E7D32;
        color: #FFFFFF;
        border: 1px solid #2E7D32;
    }
    QPushButton:disabled {
        background: #E5E7EB;
        color: #9CA3AF;
        border: 1px solid #E5E7EB;
    }
    QListWidget {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 8px;
    }
    QListWidget::item {
        padding: 10px;
        border-radius: 8px;
    }
    QListWidget::item:selected {
        background: #E8F3FF;
        color: #404040;
    }
    QSlider::groove:horizontal {
        height: 8px;
        background: #D1D5DB;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #1D85ED;
        width: 20px;
        height: 20px;
        margin: -7px 0;
        border-radius: 10px;
    }
    """


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(_stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
