"""Protocolo binario del dongle nMotion: parser, CRC y modelos de datos."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from enum import IntEnum


class MessageType(IntEnum):
    """Tipos de mensaje del protocolo UART del dongle."""

    IMU_DATA = 0x01
    IMU_EVENT = 0x02
    IMU_BATTERY = 0x03


@dataclass(frozen=True, slots=True)
class IMUData:
    """Frame IMU_DATA decodificado."""

    imu_id: int
    device_id: int
    seq: int
    t_global_ms: int
    qw: int
    qx: int
    qy: int
    qz: int
    ax: int
    ay: int
    az: int
    mx: int
    my: int
    mz: int
    flags: int
    received_at: float

    @property
    def quat_q15(self) -> tuple[float, float, float, float]:
        """Cuaternión corregido para visualización (chip rotado 180° en Z)."""
        scale = 32768.0
        return (
            self.qw / scale,
            -self.qx / scale,
            -self.qy / scale,
            self.qz / scale,
        )

    @property
    def mag_ut(self) -> tuple[float, float, float]:
        """Magnetómetro en µT."""
        return (self.mx * 0.15, self.my * 0.15, self.mz * 0.15)

    def to_dict(self) -> dict:
        """Serialización para grabación."""
        q = self.quat_q15
        mag = self.mag_ut
        return {
            "t_ms": self.t_global_ms,
            "seq": self.seq,
            "raw": {
                "q": [self.qw, self.qx, self.qy, self.qz],
                "acc": [self.ax, self.ay, self.az],
                "mag": [self.mx, self.my, self.mz],
                "flags": self.flags,
            },
            "outputs": {
                "q": list(q),
                "mag_ut": list(mag),
            },
        }


SYNC_BYTES = b"\xAA\x55"
IMU_DATA_PAYLOAD_LEN = 32


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


def parse_frame(buf: bytearray) -> tuple[int | None, bytes | None, int]:
    """
    Extrae una trama válida del buffer.

    Returns:
        (msg_type, payload, bytes_consumed). Si no hay trama completa,
        devuelve (None, None, 0).
    """
    consumed_total = 0
    while True:
        if len(buf) < 4:
            return None, None, consumed_total

        sync_idx = -1
        for i in range(len(buf) - 1):
            if buf[i] == SYNC_BYTES[0] and buf[i + 1] == SYNC_BYTES[1]:
                sync_idx = i
                break

        if sync_idx < 0:
            drop = max(0, len(buf) - 1)
            del buf[:drop]
            consumed_total += drop
            return None, None, consumed_total

        if sync_idx > 0:
            del buf[:sync_idx]
            consumed_total += sync_idx
            continue

        msg_type = buf[2]
        length = buf[3]
        frame_len = 4 + length + 2
        if len(buf) < frame_len:
            return None, None, consumed_total

        frame = bytes(buf[:frame_len])
        payload = bytes(buf[4 : 4 + length])
        crc_rx = frame[-2] | (frame[-1] << 8)
        crc_calc = crc16_ccitt(frame[2:-2])
        del buf[:frame_len]
        consumed_total += frame_len

        if crc_rx == crc_calc:
            return msg_type, payload, consumed_total
        # CRC inválido: continuar buscando desde el siguiente byte


def decode_imu_data(payload: bytes) -> IMUData | None:
    """Decodifica un payload MSG_IMU_DATA de 32 bytes."""
    if len(payload) != IMU_DATA_PAYLOAD_LEN:
        return None
    try:
        fields = struct.unpack("<BIHIhhhhhhhhhhB", payload)
    except struct.error:
        return None

    return IMUData(
        imu_id=fields[0],
        device_id=fields[1],
        seq=fields[2],
        t_global_ms=fields[3],
        qw=fields[4],
        qx=fields[5],
        qy=fields[6],
        qz=fields[7],
        ax=fields[8],
        ay=fields[9],
        az=fields[10],
        mx=fields[11],
        my=fields[12],
        mz=fields[13],
        flags=fields[14],
        received_at=time.time(),
    )
