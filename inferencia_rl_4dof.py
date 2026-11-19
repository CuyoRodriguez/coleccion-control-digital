import numpy as np


class MotorInferencia4DOF:
    """Actor NumPy desacoplado: 22 observaciones crudas -> 3 acciones incrementales.

    Formato NPZ v13: n_capas, W0/b0..., media y var. Normalizacion
    VecNormalize con epsilon=1e-8 y recorte a +/-10; Tanh solo en capas
    ocultas, salida lineal recortada a [-1,1]. No construye observaciones,
    no integra la planta, no controla la pinza ni envia comandos fisicos.
    """

    def __init__(self):
        self._capas: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
        self._media: np.ndarray | None = None
        self._var: np.ndarray | None = None

    @staticmethod
    def _arreglo(datos, clave: str) -> np.ndarray:
        if clave not in datos:
            raise ValueError(f'Falta {clave} en el modelo NPZ.')
        arreglo = np.asarray(datos[clave])
        if arreglo.dtype.kind not in 'fiu' or not np.all(np.isfinite(arreglo)):
            raise ValueError(f'{clave} debe contener numeros reales finitos.')
        return arreglo.astype(np.float64)

    def cargar_modelo(self, ruta_npz) -> None:
        """Lee y valida un NPZ sin pickle. Los errores se propagan al llamador."""
        with np.load(ruta_npz, allow_pickle=False) as datos:
            if 'n_capas' not in datos:
                raise ValueError('Falta n_capas en el modelo NPZ.')
            n = np.asarray(datos['n_capas'])
            if (n.shape != () or n.dtype.kind not in 'iu' or int(n) < 1
                    or 2 * int(n) + 3 > len(datos.files)):
                raise ValueError('n_capas debe ser un entero positivo con sus pesos y sesgos.')
            media, var = self._arreglo(datos, 'media'), self._arreglo(datos, 'var')
            if media.shape != (22,) or var.shape != (22,) or np.any(var < 0):
                raise ValueError('Se requieren media y var de 22 valores, con var no negativa.')
            capas = []
            entradas = 22
            for i in range(int(n)):
                pesos, sesgo = self._arreglo(datos, f'W{i}'), self._arreglo(datos, f'b{i}')
                if (pesos.ndim != 2 or pesos.shape[0] == 0 or pesos.shape[1] != entradas
                        or sesgo.shape != (pesos.shape[0],)):
                    raise ValueError(f'Dimensiones incompatibles en la capa {i}.')
                capas.append((pesos, sesgo))
                entradas = pesos.shape[0]
            if entradas != 3:
                raise ValueError('El actor 4 GDL debe producir tres acciones, no cuatro.')
        self._media, self._var, self._capas = media, var, tuple(capas)

    def predecir_accion(self, observacion) -> np.ndarray:
        """Recibe un vector sin normalizar de 22 valores y devuelve shape (3,)."""
        if self._capas is None or self._media is None or self._var is None:
            raise RuntimeError('Ejecuta cargar_modelo(ruta_npz) antes de predecir.')
        obs = np.asarray(observacion)
        if obs.dtype.kind not in 'fiu' or obs.shape != (22,) or not np.all(np.isfinite(obs)):
            raise ValueError('La observacion debe ser un vector de 22 numeros reales finitos.')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            x = np.clip((obs.astype(np.float64) - self._media) / np.sqrt(self._var + 1e-8), -10, 10)
            for pesos, sesgo in self._capas[:-1]:
                x = np.tanh(x @ pesos.T + sesgo)
            pesos, sesgo = self._capas[-1]
            accion = np.clip(x @ pesos.T + sesgo, -1, 1)
        if not np.all(np.isfinite(accion)):
            raise FloatingPointError('La inferencia produjo una accion no finita.')
        return accion.astype(np.float32)
