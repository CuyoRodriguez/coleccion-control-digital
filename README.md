# Sebastian Enriquez Ramos - Workpage

Aplicación multipágina de Streamlit para reunir recursos de ingeniería, Control Digital y proyectos personales.

## Ejecución local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address localhost
```

## Brazo SG90 de 2 GDL

- **Control HMI** permite probar poses y editar la secuencia virtual. El
  movimiento manual por USB físico requiere una instancia local iniciada con
  el comando anterior y el modo de control **Cinemático**.
- **Código del robot → Paquete** descarga el firmware Arduino de hombro/codo
  (D9/D10) y el enviador Python con sus módulos.
- **TXT / HTML** exportan el resumen matemático; el HTML con video requiere
  una trayectoria calculada.
- La web pública ofrece simulación, descarga de código y eco serial `loop://`,
  pero no expone puertos físicos. En local, el primer comando físico apunta a
  90°/90° sin rampa; desconectar no equivale a una parada de emergencia.

## Publicación en Streamlit Community Cloud

Conecta este repositorio en Streamlit Community Cloud y configura `app.py` como archivo principal. Las dependencias se instalan desde `requirements.txt`.
