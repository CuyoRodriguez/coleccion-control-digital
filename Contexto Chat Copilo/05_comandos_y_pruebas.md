# Comandos y pruebas

Todos se ejecutan en PowerShell desde la raíz del proyecto
(`C:\Users\seb10\Documents\Codigo Pagina`).

## Suite de pruebas SG90 (29 pruebas, ~2 min)

```powershell
$env:PYTHONUTF8=1
uv run --offline --python 3.11 --with numpy==1.24.3 --with streamlit==1.65.0 --with streamlit-option-menu --with scipy==1.15.3 --with plotly==7.1.0 --with matplotlib==3.7.1 --with control --with pyserial --with joblib --with scikit-learn==1.2.2 python -m unittest scripts.test_reproduccion_sg90 scripts.test_teach_pendant_sg90
```

- `PYTHONUTF8=1` evita errores con acentos y símbolos (°, ², etc.).
- Las pruebas no abren puertos USB reales.

## Servidor local (Streamlit)

```powershell
streamlit run app.py --server.port 8503
```

Salud: `http://localhost:8503/_stcore/health` debe responder `ok`.

## Auditor y telemetría

```powershell
python -m scripts.auditar_trayectoria_sg90 "C:\ruta\hardware\trayectoria.json"
python -m scripts.auditar_trayectoria_sg90 "C:\ruta\hardware\trayectoria.json" --grafica telemetria.png --vertice 95 --vertice 190
```

Sale con código 1 si alguna curva supera los límites.

## Visión

```powershell
python -m pip install "opencv-python-headless==4.11.0.86"
python -m scripts.medir_punta_sg90 video_prueba.mp4 --esquinas 100 80 590 95 560 390 120 375 --hoja-mm 297 210 --color rojo --salida medicion_punta.csv
python -m scripts.comparar_trayectorias_sg90 medicion_punta.csv "C:\ruta\hardware\trayectoria.json" --hombro-mm 148.5 10 --inicio-video-s 1.5 --grafica comparacion.png
```

Con `uv` (OpenCV no está en el entorno base):

```powershell
uv run --python 3.11 --with numpy==1.24.3 --with opencv-python-headless==4.11.0.86 python -m scripts.medir_punta_sg90 ...
uv run --offline --python 3.11 --with numpy==1.24.3 --with matplotlib==3.7.1 --with pyserial python -m scripts.comparar_trayectorias_sg90 ...
```

## Reproducir en el robot real

```powershell
# Solo validar el JSON, sin abrir puertos:
python assets\hardware_sg90\reproducir_simulacion_sg90.py --trayectoria trayectoria.json --seco
# Eco sin hardware:
python assets\hardware_sg90\reproducir_simulacion_sg90.py --trayectoria trayectoria.json --puerto loop://
# Robot real (pide confirmación antes de mover):
python assets\hardware_sg90\reproducir_simulacion_sg90.py --trayectoria trayectoria.json --puerto COM3
```

Para JSON v2 el firmware debe responder `LISTO US2`.

## Arduino CLI (compilar / subir firmware)

```powershell
$cli = "C:\Users\seb10\AppData\Local\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe"
& $cli compile --fqbn arduino:avr:uno <carpeta_brazo_sg90>
& $cli upload -p COM3 --fqbn arduino:avr:uno <carpeta_brazo_sg90>
```

El sketch debe estar en una carpeta con su mismo nombre (`brazo_sg90/brazo_sg90.ino`).
Requiere la librería del PCA9685 (Adafruit PWM Servo Driver). El ZIP de
exportación del simulador ya trae el firmware.

## Git

```powershell
git --no-pager status --short
git add <archivos>; git commit -m "mensaje"; git push
```

Incluir en los commits el trailer:
`Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
