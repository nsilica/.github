#!/usr/bin/env python3
"""
Calibración Hard-Iron del magnetómetro AK09916 a través del dongle nMotion.

El dongle recibe IMUpack v2 por BLE (23 bytes) y lo reenvía por USB-CDC con
el protocolo binario:

    [AA] [55] [msg_type] [length] [payload...] [crc16_lo] [crc16_hi]

Este script parsea mensajes MSG_IMU_DATA (0x01, payload 32 bytes), extrae
mx/my/mz (offsets 25-30) y muestra la esfera magnética en 3D para estimar
el offset hard-iron por least-squares.

Sensibilidad AK09916: 0.15 µT/LSB. Rango completo ±32767 LSB.
"""

import collections
import argparse
import signal
import struct
import threading
import time

import numpy as np
from scipy.optimize import least_squares

MAX_MAG_POINTS = 10_000
MAX_PLOT_POINTS = 300
DRAW_INTERVAL_S = 1.0 / 60.0

SYNC_BYTES = b"\xAA\x55"
MSG_IMU_DATA = 0x01

DONGLE_VID = 0x2FE3
DONGLE_PID = 0x4E4D


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE usado por el dongle (poly 0x1021, init 0xFFFF)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def find_ports() -> list[str]:
    import serial.tools.list_ports

    candidates = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if (p.vid == DONGLE_VID and p.pid == DONGLE_PID) or \
           "nMotion" in (p.description or "") or \
           "2fe3" in hwid and "4e4d" in hwid:
            candidates.append(p.device)
    return candidates


def select_port(port_arg: str | None = None) -> str:
    import serial.tools.list_ports

    if port_arg:
        return port_arg
    candidates = find_ports()
    if not candidates:
        print("No se detectó dongle nMotion. Puertos disponibles:")
        all_ports = serial.tools.list_ports.comports()
        for i, p in enumerate(all_ports):
            print(f"  {i + 1}. {p.device} - {p.description}")
        if not all_ports:
            raise RuntimeError("No hay puertos serie")
        choice = input("Selecciona número: ").strip()
        return all_ports[int(choice) - 1].device
    if len(candidates) == 1:
        print(f"Dongle nMotion detectado en: {candidates[0]}")
        return candidates[0]
    print("Varios dongles detectados:")
    for i, p in enumerate(candidates):
        print(f"  {i + 1}. {p}")
    choice = input("Selecciona número: ").strip()
    return candidates[int(choice) - 1]


def parse_dongle_frame(buf: bytearray):
    """
    Busca una trama completa en buf. Devuelve (msg_type, payload, bytes_consumidos)
    o (None, None, 0) si no hay trama completa.
    """
    while True:
        if len(buf) < 4:
            return None, None, 0
        try:
            idx = next(i for i in range(len(buf) - 1)
                       if buf[i] == SYNC_BYTES[0] and buf[i + 1] == SYNC_BYTES[1])
        except StopIteration:
            # Conservar solo el último byte por si es el inicio de sync
            consumed = max(0, len(buf) - 1)
            del buf[:consumed]
            return None, None, consumed

        if idx > 0:
            del buf[:idx]
            continue

        msg_type = buf[2]
        length = buf[3]
        frame_len = 4 + length + 2
        if len(buf) < frame_len:
            return None, None, 0

        frame = bytes(buf[:frame_len])
        payload = bytes(buf[4:4 + length])
        crc_rx = frame[-2] | (frame[-1] << 8)
        crc_calc = crc16_ccitt(frame[2:-2])
        del buf[:frame_len]

        if crc_rx == crc_calc:
            return msg_type, payload, frame_len
        # CRC incorrecto: seguir buscando a partir del byte siguiente


def dongle_reader_thread(ser, stop_event, latest_mag, mag_lock, has_mag,
                         mag_history, mag_scatter_buf, last_point_time):
    import serial

    buf = bytearray()

    def on_mag(mx, my, mz):
        with mag_lock:
            latest_mag[0], latest_mag[1], latest_mag[2] = mx, my, mz
            has_mag[0] = True
            last_point_time[0] = time.time()
        mag_history["mx"].append(mx)
        mag_history["my"].append(my)
        mag_history["mz"].append(mz)
        mag_scatter_buf["x"].append(mx)
        mag_scatter_buf["y"].append(my)
        mag_scatter_buf["z"].append(mz)
        mag_scatter_buf["t"].append(time.time())

    while not stop_event.is_set():
        try:
            if ser.in_waiting > 0:
                buf.extend(ser.read(ser.in_waiting))
                while True:
                    msg_type, payload, consumed = parse_dongle_frame(buf)
                    if consumed == 0:
                        break
                    if msg_type == MSG_IMU_DATA and len(payload) == 32:
                        try:
                            # payload: imu_id(1) device_id(4) seq(2) t_global(4)
                            #          qw qx qy qz ax ay az mx my mz flags
                            fields = struct.unpack("<BIHIhhhhhhhhhhB", payload)
                            mx, my, mz = fields[11], fields[12], fields[13]
                            on_mag(mx, my, mz)
                        except struct.error:
                            pass
        except serial.SerialException:
            break
        except Exception:
            # Mantener lectura ante errores puntuales
            pass
        time.sleep(0.001)


def probe_serial(port: str, baud: int, seconds: float) -> int:
    import serial

    ser = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=0.1,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )
    print(f"Probe {port} @ {baud} durante {seconds:.1f}s...")
    buf = bytearray()
    count = 0
    last = None
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buf.extend(chunk)
            while True:
                msg_type, payload, consumed = parse_dongle_frame(buf)
                if consumed == 0:
                    break
                if msg_type == MSG_IMU_DATA and len(payload) == 32:
                    try:
                        fields = struct.unpack("<BIHIhhhhhhhhhhB", payload)
                        mx, my, mz = fields[11], fields[12], fields[13]
                        count += 1
                        last = (mx, my, mz)
                        if count <= 5 or count % 25 == 0:
                            print(f"#{count:04d} raw=({mx:+d},{my:+d},{mz:+d}) "
                                  f"uT=({mx * 0.15:+.2f},{my * 0.15:+.2f},{mz * 0.15:+.2f})")
                    except struct.error:
                        pass
    finally:
        ser.close()
    print(f"Muestras MAG: {count}. Ultima: {last}")
    return 0 if count > 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Calibración Hard-Iron del AK09916 vía dongle nMotion"
    )
    parser.add_argument("--port", default=None, help="Puerto serie del dongle (ej: COM12)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--probe", type=float, default=0.0,
                        help="Solo probar recepcion durante N segundos, sin GUI")
    args = parser.parse_args()

    port = select_port(args.port)
    if args.probe > 0:
        return probe_serial(port, args.baud, args.probe)

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button as MplButton
    except ImportError:
        print("Se necesita matplotlib: pip install matplotlib")
        return

    try:
        import serial
        ser = serial.Serial(
            port=port,
            baudrate=args.baud,
            timeout=0.05,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
    except Exception as e:
        print(f"Error abriendo puerto: {e}")
        return

    print("Conectado al dongle nMotion.")
    print("Rota la IMU en todas las orientaciones para capturar la esfera.")
    print("Cierra la ventana o Ctrl+C para salir.")

    running = [True]

    def _signal_handler(sig, frame):
        running[0] = False

    signal.signal(signal.SIGINT, _signal_handler)

    latest_mag = [0, 0, 0]
    mag_lock = threading.Lock()
    has_mag = [False]
    last_point_time = [0.0]
    mag_history = {k: collections.deque(maxlen=MAX_PLOT_POINTS) for k in ("mx", "my", "mz")}
    mag_scatter_buf = {
        "x": collections.deque(maxlen=MAX_MAG_POINTS),
        "y": collections.deque(maxlen=MAX_MAG_POINTS),
        "z": collections.deque(maxlen=MAX_MAG_POINTS),
        "t": collections.deque(maxlen=MAX_MAG_POINTS),
    }
    stop_event = threading.Event()
    reader = threading.Thread(
        target=dongle_reader_thread,
        args=(ser, stop_event, latest_mag, mag_lock, has_mag, mag_history,
              mag_scatter_buf, last_point_time),
        daemon=True,
    )
    reader.start()

    plt.ion()
    fig = plt.figure(figsize=(16, 7))
    fig.subplots_adjust(left=0.18, bottom=0.14, wspace=0.38)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_xlabel("Mx (LSB)")
    ax3d.set_ylabel("My (LSB)")
    ax3d.set_zlabel("Mz (LSB)")
    ax3d.set_title("Magnetómetro — Hard Iron Calibration (dongle nMotion)")

    scatter = [None]
    scatter_trail = [None]
    scatter_center = [None]
    sphere_surface = [None]

    _u = np.linspace(0, 2 * np.pi, 24)
    _v = np.linspace(0, np.pi, 16)
    _SX = np.outer(np.cos(_u), np.sin(_v))
    _SY = np.outer(np.sin(_u), np.sin(_v))
    _SZ = np.outer(np.ones_like(_u), np.cos(_v))

    ax_txt_vals = fig.add_axes([0.005, 0.62, 0.165, 0.30])
    ax_txt_vals.set_axis_off()
    val_text = ax_txt_vals.text(
        0.05, 0.95, "Mx: —\nMy: —\nMz: —",
        transform=ax_txt_vals.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.95),
    )
    count_text = ax_txt_vals.text(
        0.05, 0.18, "Puntos: 0",
        transform=ax_txt_vals.transAxes,
        ha="left", va="bottom",
        fontsize=9, color="#555555",
    )

    ax_txt_center = fig.add_axes([0.005, 0.22, 0.165, 0.38])
    ax_txt_center.set_axis_off()
    center_text = ax_txt_center.text(
        0.05, 0.95,
        "Centro estimado\n  cx: —\n  cy: —\n  cz: —\nRadio: —",
        transform=ax_txt_center.transAxes,
        ha="left", va="top",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffde7", edgecolor="#f0c040", alpha=0.97),
    )

    ax_led = fig.add_axes([0.03, 0.10, 0.12, 0.10])
    ax_led.set_axis_off()
    ax_led.set_xlim(0, 1)
    ax_led.set_ylim(0, 1)
    led_circle = plt.Circle((0.30, 0.50), 0.22, color="#c62828", zorder=2)
    ax_led.add_patch(led_circle)
    led_label = ax_led.text(
        0.58, 0.50, "inestable",
        ha="left", va="center",
        fontsize=8, fontweight="bold", color="#333333",
    )

    colors_m = {"mx": "#E53935", "my": "#43A047", "mz": "#1E88E5"}
    colors_c = {"cx": "#FF6F00", "cy": "#AB47BC", "cz": "#00ACC1"}
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Magnetómetro en tiempo real")
    ax2.set_ylabel("LSB (int16)")
    ax2.set_xlabel("Muestras")
    ax2.grid(True, linestyle="--", alpha=0.5)
    mlines = {k: ax2.plot([], [], label=k, color=c, linewidth=1.2)[0]
              for k, c in colors_m.items()}
    clines = {k: ax2.plot([], [], label=k, color=c, linewidth=1.5,
                          linestyle="--")[0]
              for k, c in colors_c.items()}
    ax2.legend(loc="upper right", fontsize=8)

    ax_btn_clear = fig.add_axes([0.38, 0.02, 0.24, 0.07])
    btn_clear = MplButton(ax_btn_clear, "Limpiar puntos")

    def on_clear(_):
        with mag_lock:
            for key in mag_scatter_buf:
                mag_scatter_buf[key].clear()
            if scatter[0] is not None:
                scatter[0].remove()
                scatter[0] = None
            if scatter_trail[0] is not None:
                scatter_trail[0].remove()
                scatter_trail[0] = None
            if scatter_center[0] is not None:
                scatter_center[0].remove()
                scatter_center[0] = None
            if sphere_surface[0] is not None:
                sphere_surface[0].remove()
                sphere_surface[0] = None
            count_text.set_text("Puntos: 0")
            center_text.set_text("Centro estimado\n  cx: —\n  cy: —\n  cz: —\nRadio: —")

    btn_clear.on_clicked(on_clear)

    center_history = {k: collections.deque(maxlen=MAX_PLOT_POINTS) for k in ("cx", "cy", "cz")}
    CONVERGENCE_WINDOW = 50
    CONVERGENCE_THR = 5.0
    center_mod_history = collections.deque(maxlen=CONVERGENCE_WINDOW)

    def on_close(_):
        running[0] = False

    fig.canvas.mpl_connect("close_event", on_close)

    try:
        while running[0]:
            with mag_lock:
                mx, my, mz = latest_mag
                has_data = has_mag[0]

            if has_data:
                val_text.set_text(f"Mx: {mx:6d}\nMy: {my:6d}\nMz: {mz:6d}")

                xs = list(mag_scatter_buf["x"])
                ys = list(mag_scatter_buf["y"])
                zs = list(mag_scatter_buf["z"])
                ts = list(mag_scatter_buf["t"])
                n = min(len(xs), len(ys), len(zs), len(ts))
                xs, ys, zs, ts = xs[:n], ys[:n], zs[:n], ts[:n]

                if n > 0:
                    count_text.set_text(f"Puntos: {n}")

                    now = time.time()
                    mz_arr = np.array(zs, dtype=float)
                    mz_mean = mz_arr.mean()

                    COLOR_RED = np.array([0.85, 0.10, 0.10, 1.0])
                    COLOR_DARKBLUE = np.array([0.08, 0.18, 0.55, 1.0])
                    rgba = np.where(
                        (mz_arr < mz_mean)[:, None],
                        COLOR_RED,
                        COLOR_DARKBLUE,
                    )

                    ts_arr = np.array(ts, dtype=float)
                    TRAIL_S = 2.0
                    trail_mask = (now - ts_arr) < TRAIL_S

                    non_trail = ~trail_mask
                    if scatter[0] is not None:
                        scatter[0].remove()
                        scatter[0] = None
                    if non_trail.any():
                        scatter[0] = ax3d.scatter(
                            np.array(xs)[non_trail],
                            np.array(ys)[non_trail],
                            np.array(zs)[non_trail],
                            c=rgba[non_trail],
                            s=4,
                            alpha=None,
                            depthshade=True,
                        )

                    if scatter_trail[0] is not None:
                        scatter_trail[0].remove()
                        scatter_trail[0] = None
                    if trail_mask.any():
                        trail_idx = np.where(trail_mask)[0]
                        age = now - ts_arr[trail_idx]
                        t_norm = 1.0 - age / TRAIL_S
                        sizes = 4 + t_norm * 28
                        alphas = 0.3 + t_norm * 0.7
                        trail_rgba = np.zeros((len(trail_idx), 4))
                        trail_rgba[:, 1] = 0.85
                        trail_rgba[:, 2] = 0.15
                        trail_rgba[:, 3] = alphas
                        trail_rgba[-1] = [0.1, 1.0, 0.2, 1.0]
                        sizes[-1] = 55

                        scatter_trail[0] = ax3d.scatter(
                            np.array(xs)[trail_idx],
                            np.array(ys)[trail_idx],
                            np.array(zs)[trail_idx],
                            c=trail_rgba,
                            s=sizes,
                            alpha=None,
                            depthshade=False,
                            zorder=100,
                        )

                    all_vals = xs + ys + zs
                    vmin, vmax = min(all_vals), max(all_vals)
                    margin = max((vmax - vmin) * 0.1, 50)
                    ax3d.set_xlim(vmin - margin, vmax + margin)
                    ax3d.set_ylim(vmin - margin, vmax + margin)
                    ax3d.set_zlim(vmin - margin, vmax + margin)

                    if n >= 20:
                        xa = np.array(xs, dtype=float)
                        ya = np.array(ys, dtype=float)
                        za = np.array(zs, dtype=float)

                        cx0, cy0, cz0 = xa.mean(), ya.mean(), za.mean()
                        r0 = float(np.mean(np.sqrt((xa - cx0) ** 2 + (ya - cy0) ** 2 + (za - cz0) ** 2)))

                        def _residuals(p):
                            cx_, cy_, cz_, r_ = p
                            return np.sqrt((xa - cx_) ** 2 + (ya - cy_) ** 2 + (za - cz_) ** 2) - r_

                        res = least_squares(
                            _residuals,
                            x0=[cx0, cy0, cz0, r0],
                            method="lm",
                            max_nfev=50,
                        )
                        cx, cy, cz, r = res.x[0], res.x[1], res.x[2], abs(res.x[3])

                        center_text.set_text(
                            f"Centro estimado\n"
                            f"  cx: {cx:+.1f}\n"
                            f"  cy: {cy:+.1f}\n"
                            f"  cz: {cz:+.1f}\n"
                            f"Radio:  {r:.1f}"
                        )

                        center_history["cx"].append(cx)
                        center_history["cy"].append(cy)
                        center_history["cz"].append(cz)
                        mod_c = float(np.sqrt(cx ** 2 + cy ** 2 + cz ** 2))
                        center_mod_history.append(mod_c)

                        if len(center_mod_history) >= CONVERGENCE_WINDOW:
                            variation = float(np.max(center_mod_history) - np.min(center_mod_history))
                            if variation < CONVERGENCE_THR:
                                led_circle.set_color("#2e7d32")
                                led_label.set_text("estable")
                            else:
                                led_circle.set_color("#c62828")
                                led_label.set_text("inestable")

                        if sphere_surface[0] is not None:
                            sphere_surface[0].remove()
                        sphere_surface[0] = ax3d.plot_surface(
                            cx + r * _SX,
                            cy + r * _SY,
                            cz + r * _SZ,
                            alpha=0.08,
                            color="#FF6F00",
                            linewidth=0,
                            antialiased=False,
                            zorder=1,
                        )

                        if scatter_center[0] is not None:
                            scatter_center[0].remove()
                        scatter_center[0] = ax3d.scatter(
                            [cx], [cy], [cz],
                            marker="+",
                            s=200,
                            c="#FF6F00",
                            linewidths=2.5,
                            depthshade=False,
                            zorder=200,
                        )

            n_hist = len(mag_history["mx"])
            if n_hist > 0:
                x_data = list(range(n_hist))
                for key, mline in mlines.items():
                    mline.set_data(x_data, list(mag_history[key]))
                n_c = len(center_history["cx"])
                if n_c > 0:
                    xc = list(range(n_c))
                    for key, cline in clines.items():
                        cline.set_data(xc, list(center_history[key]))
                ax2.relim()
                ax2.autoscale_view()
                ax2.set_xlim(0, max(n_hist, n_c, 10))

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(DRAW_INTERVAL_S)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupción recibida. Cerrando...")
    finally:
        running[0] = False
        stop_event.set()
        reader.join(timeout=0.5)
        if ser.is_open:
            ser.close()
        plt.close(fig)
        plt.ioff()

    print("Puerto cerrado. Hasta luego.")


if __name__ == "__main__":
    main()
