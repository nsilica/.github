# nMotion Magnetometer App

Aplicación desktop multiplataforma para visualizar en tiempo real la orientación y el campo magnético de las IMUs nMotion a través del dongle USB-BLE.

## Características

- **Pantalla de inicio** con acceso a dos modos: medición y reproducción.
- **Medición en tiempo real**:
  - Orientación 3D de la IMU (cuaternión Q15 corregido de montaje).
  - Calibración Hard Iron del AK09916 con ajuste esférico.
  - Guardado de calibración hard-iron en `recordings/magnetic_hard_iron_calibration.json`.
  - Grabación de datos completos (raw + outputs) en JSON.
- **Reproducción** de grabaciones con timeline draggable y reproducción en bucle.
- Interfaz desktop con PySide6 y matplotlib embebido nativamente en Qt.

## Estructura

```text
nmotion-magnetometer-app/
├── config.json                 # Configuración de puerto, path de grabaciones, colores
├── pyproject.toml              # Dependencias y metadatos
├── recordings/                 # Grabaciones guardadas por defecto
├── src/
│   ├── main.py                 # Punto de entrada PySide6
│   ├── qt_app.py               # UI Qt, navegación, medición y replay
│   ├── qt_plots.py             # Canvases matplotlib embebidos en Qt
│   ├── config.py               # Carga de configuración
│   ├── theme.py                # Design tokens y estilos
│   ├── dongle_protocol.py      # Parser del protocolo binario AA55 + CRC
│   ├── serial_worker.py        # Hilo de lectura del dongle
│   ├── recorder.py             # Guardar/cargar grabaciones JSON
│   ├── replay_engine.py        # Motor de reproducción con timeline
│   ├── views/                  # Vistas Flet legacy no usadas por el entrypoint actual
│   └── plots/                  # Renderizadores legacy basados en PNG/base64
└── README.md
```

## Requisitos

- Python 3.12+
- Dongle nMotion conectado por USB (VID:PID `0x2FE3:0x4E4D`)
- Dependencias gestionadas con `uv` o `pip`

## Instalación

```powershell
# Con uv (recomendado)
uv sync

# O con pip
pip install -e ".[dev]"
```

## Ejecución en desarrollo

```powershell
uv run python -m src.main
```

## Configuración

Edita `config.json`:

```json
{
  "serial": {
    "port": null,           // null = auto-detectar dongle
    "baudrate": 115200
  },
  "recording": {
    "directory": "./recordings",
    "max_samples": 100000
  }
}
```

## Build del ejecutable .exe

La app ya no depende de Flet/Flutter. Para generar un ejecutable se recomienda PyInstaller:

```powershell
uv add --dev pyinstaller
uv run pyinstaller --noconfirm --windowed --name nMotionMagnetometer main.py
```

El ejecutable se generará en `dist/nMotionMagnetometer/`.

## Formato de grabación

Cada grabación se guarda como un JSON con metadatos en `header` y la lista de muestras:

```json
{
  "header": {
    "version": 1,
    "created_at": "2026-06-16T13:45:00",
    "mode": "magnetic",
    "samples_count": 1250,
    "duration_ms": 5234
  },
  "samples": [
    {
      "t_ms": 0,
      "seq": 1,
      "raw": {
        "q": [32767, 0, 0, 0],
        "acc": [120, -200, 40],
        "mag": [50, -230, 60],
        "flags": 19
      },
      "outputs": {
        "q": [0.9999, -0.0, -0.0, 0.0],
        "mag_ut": [7.5, -34.5, 9.0]
      }
    }
  ]
}
```

## Notas de implementación

- Los gráficos 3D se muestran con `FigureCanvasQTAgg`, embebidos directamente como widgets Qt.
- La lectura del puerto serie y la reproducción corren en hilos aparte; la UI consume los datos con `QTimer` y señales Qt.
- El cuaternión se corrige con la montura del chip: `(qw, -qx, -qy, +qz)`.

## Licencia

Proyecto interno nMotion.
