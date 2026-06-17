#!/usr/bin/env python3
"""
Visualización en tiempo real: caja 3D orientada según el cuaternión de la IMU
recibida a través del dongle nMotion.

El dongle reenvía por USB-CDC el protocolo binario:
    [AA] [55] [msg_type] [length] [payload...] [crc16_lo] [crc16_hi]

Este script parsea MSG_IMU_DATA (0x01, payload 32 bytes), extrae el cuaternión
Q15 (offsets 11-18), aplica la corrección de montaje del chip y dibuja una caja
3D sólida orientada en tiempo real.
"""

import argparse
import math
import struct
import threading
import time

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

IMU_DATA_PAYLOAD_LEN = 32
Q15_SCALE = 32768.0

UART_SYNC_1 = 0xAA
UART_SYNC_2 = 0x55
UART_MSG_TYPE_IMU_DATA = 0x01
UART_MSG_TYPE_IMU_EVENT = 0x02

DONGLE_VID = 0x2FE3
DONGLE_PID = 0x4E4D


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def q15_to_float(q_int: int) -> float:
    return q_int / Q15_SCALE


def quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> list:
    """Cuaternión unitario (w,x,y,z) -> matriz 3x3."""
    w, x, y, z = qw, qx, qy, qz
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def mat_vec(R: list, v: list) -> list:
    return [R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2] for i in range(3)]


# Caja con 3 lados distintos (paralelepípedo): semiejes X, Y, Z
BOX_HALF = (0.17, 0.37, 0.11)
BOX_VERTICES = [
    [-BOX_HALF[0], -BOX_HALF[1], -BOX_HALF[2]], [BOX_HALF[0], -BOX_HALF[1], -BOX_HALF[2]],
    [BOX_HALF[0],  BOX_HALF[1], -BOX_HALF[2]], [-BOX_HALF[0],  BOX_HALF[1], -BOX_HALF[2]],
    [-BOX_HALF[0], -BOX_HALF[1],  BOX_HALF[2]], [BOX_HALF[0], -BOX_HALF[1],  BOX_HALF[2]],
    [BOX_HALF[0],  BOX_HALF[1],  BOX_HALF[2]], [-BOX_HALF[0],  BOX_HALF[1],  BOX_HALF[2]],
]

BOX_FACES = [
    [0, 1, 2, 3],  # cara Z-
    [4, 5, 6, 7],  # cara Z+
    [0, 1, 5, 4],  # cara Y-
    [2, 3, 7, 6],  # cara Y+
    [0, 3, 7, 4],  # cara X-
    [1, 2, 6, 5],  # cara X+
]

FACE_COLORS = [
    (0.20, 0.50, 0.80, 0.60),  # Z± azul
    (0.20, 0.50, 0.80, 0.60),
    (0.20, 0.75, 0.45, 0.60),  # Y± verde
    (0.20, 0.75, 0.45, 0.60),
    (0.90, 0.40, 0.20, 0.60),  # X± naranja
    (0.90, 0.40, 0.20, 0.60),
]

DRAW_INTERVAL_S = 1.0 / 60.0


def find_ports() -> list[str]:
    import serial.tools.list_ports

    candidates = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if (p.vid == DONGLE_VID and p.pid == DONGLE_PID) or \
           "nMotion" in (p.description or "") or \
           ("2fe3" in hwid and "4e4d" in hwid):
            candidates.append(p.device)
    return candidates


def select_port(port_arg: str | None = None) -> str | None:
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


def process_buffer(buffer: bytearray, on_quat):
    """
    Parsea tramas AA 55 con CRC. Para cada IMU_DATA (0x01) válido llama
    on_quat(qw, qx, qy, qz) con el cuaternión en float.
    """
    while True:
        sync_idx = -1
        for i in range(len(buffer) - 1):
            if buffer[i] == UART_SYNC_1 and buffer[i + 1] == UART_SYNC_2:
                sync_idx = i
                break
        if sync_idx < 0:
            if len(buffer) > 2:
                buffer = buffer[-1:]
            return buffer

        if sync_idx > 0:
            buffer = buffer[sync_idx:]

        if len(buffer) < 4:
            return buffer
        msg_type = buffer[2]
        length = buffer[3]
        frame_len = 4 + length + 2
        if len(buffer) < frame_len:
            return buffer

        frame = bytes(buffer[:frame_len])
        buffer = buffer[frame_len:]

        crc_rx = frame[-2] | (frame[-1] << 8)
        crc_calc = crc16_ccitt(frame[2:-2])
        if crc_rx != crc_calc:
            continue

        if msg_type == UART_MSG_TYPE_IMU_DATA and length == IMU_DATA_PAYLOAD_LEN:
            payload = frame[4:4 + IMU_DATA_PAYLOAD_LEN]
            try:
                fields = struct.unpack("<BIHIhhhhhhhhhhB", payload)
                qw_i, qx_i, qy_i, qz_i = fields[4], fields[5], fields[6], fields[7]
                # Corrección de montaje: chip rotado 180° alrededor del eje Z del PCB
                qw_f = q15_to_float(qw_i)
                qx_f = -q15_to_float(qx_i)
                qy_f = -q15_to_float(qy_i)
                qz_f = q15_to_float(qz_i)
                if on_quat:
                    on_quat(qw_f, qx_f, qy_f, qz_f)
            except struct.error:
                pass

    return buffer


def build_rotated_verts(qw, qx, qy, qz):
    """Rota los vértices con el cuaternión (ya corregido, normalizado)."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-9:
        return [list(v) for v in BOX_VERTICES]
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    R = quat_to_rotation_matrix(w, x, y, z)
    return [mat_vec(R, v) for v in BOX_VERTICES]


def serial_reader_thread(ser, stop_event, latest_quat, quat_lock, has_quat):
    import serial

    buffer = bytearray()

    def on_quat(qw, qx, qy, qz):
        with quat_lock:
            latest_quat[0], latest_quat[1], latest_quat[2], latest_quat[3] = qw, qx, qy, qz
            has_quat[0] = True

    while not stop_event.is_set():
        try:
            if ser.in_waiting > 0:
                buffer.extend(ser.read(ser.in_waiting))
                buffer = process_buffer(buffer, on_quat)
        except serial.SerialException:
            break
        except Exception:
            pass
        time.sleep(0.001)


def main():
    parser = argparse.ArgumentParser(description="Visualización 3D de orientación IMU vía dongle nMotion")
    parser.add_argument("--port", default=None, help="Puerto serie del dongle (ej: COM12)")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("Se necesita matplotlib: pip install matplotlib")
        return

    port_name = select_port(args.port)
    if port_name is None:
        return

    try:
        import serial
        ser = serial.Serial(
            port=port_name,
            baudrate=args.baud,
            timeout=0.05,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
    except Exception as e:
        print(f"Error abriendo puerto: {e}")
        return

    print(f"Conectado a {port_name}. Cierra la ventana o Ctrl+C para salir.")

    latest_quat = [1.0, 0.0, 0.0, 0.0]
    quat_lock = threading.Lock()
    has_quat = [False]
    stop_event = threading.Event()
    reader = threading.Thread(
        target=serial_reader_thread,
        args=(ser, stop_event, latest_quat, quat_lock, has_quat),
        daemon=True,
    )
    reader.start()

    plt.ion()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Orientación IMU  [{port_name}]")

    verts = build_rotated_verts(1.0, 0.0, 0.0, 0.0)
    face_polys = [[verts[i] for i in face] for face in BOX_FACES]
    poly_col = Poly3DCollection(
        face_polys,
        facecolors=FACE_COLORS,
        edgecolors=(0.0, 0.0, 0.0, 0.8),
        linewidths=1.2,
        zsort="average",
    )
    ax.add_collection3d(poly_col)

    running = [True]

    def on_close(_):
        running[0] = False

    fig.canvas.mpl_connect("close_event", on_close)

    try:
        while running[0]:
            with quat_lock:
                q = list(latest_quat)
                has_data = has_quat[0]

            if has_data:
                verts = build_rotated_verts(q[0], q[1], q[2], q[3])
                face_polys = [[verts[i] for i in face] for face in BOX_FACES]
                poly_col.set_verts(face_polys)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(DRAW_INTERVAL_S)
    except KeyboardInterrupt:
        pass
    finally:
        running[0] = False
        stop_event.set()
        reader.join(timeout=0.5)
        if ser.is_open:
            ser.close()
        plt.close(fig)
        plt.ioff()
    print("Puerto cerrado.")


if __name__ == "__main__":
    main()
