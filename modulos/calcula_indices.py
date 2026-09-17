"""
Módulo para el cálculo matricial de índices espectrales (NDVI, NDRE, etc.)
a partir de bandas individuales de imágenes multiespectrales locales (dron).
"""

import os
import sys
from typing import Any

import numpy as np
import rasterio
import rasterio.errors
import spyndex
from rasterio.warp import Resampling, calculate_default_transform, reproject

# Mapeo de parámetros Spyndex a nombres de archivo esperados
MAPA_BANDAS: dict[str, str] = {
    "B": "blue.tif",
    "G": "green_560.tif",
    "G1": "green_531.tif",
    "R": "red.tif",
    "N": "nir.tif",
    "RE1": "red_edge_705.tif",
    "RE2": "red_edge_740.tif",
    # 'S2': 'swir2.tif'
}

# Constantes usadas por Spyndex
CONSTANTES_SPYNDEX: list[str] = [
    "g",
    "L",
    "C1",
    "C2",
    "c",
    "cexp",
    "nexp",
    "alpha",
    "beta",
    "epsilon",
    "fdelta",
    "gamma",
    "omega",
    "sla",
    "slb",
    "k",
    "p",
    "sigma",
]

CRS_SALIDA: str = "EPSG:32630"
NODATA_VAL: float = -9999.0


def carga_bandas(
    ruta_entrada: str
    | os.PathLike,  # Usamos 'X | Y' (PEP 604, Python 3.10+) en lugar de Union[X, Y] (Ruff UP007)
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """
    Cargo los rasters de las bandas espectrales desde el directorio especificado.
    """
    print(f"Leyendo rasters del directorio {ruta_entrada}...")

    if not os.path.exists(ruta_entrada):
        raise FileNotFoundError(f"El directorio de entrada no existe: {ruta_entrada}")

    datos_bandas: dict[str, dict[str, np.ndarray]] = {}
    info_referencia: dict[str, Any] | None = None  # Usamos 'T | None' en lugar de Optional[T] (PEP 604, Ruff UP007)

    for param, archivo in MAPA_BANDAS.items():
        ruta_completa = os.path.join(ruta_entrada, archivo)
        if not os.path.exists(ruta_completa):
            # Opcional: Imprimir advertencia si falta alguna banda crítica
            # print(f"Advertencia: Archivo no encontrado: {archivo} (Para parámetro {param})")
            continue

        try:
            with rasterio.open(ruta_completa) as src:
                data = src.read(1)
                mask = src.read_masks(1) == 0  # True donde es nodata/masked
                datos_bandas[param] = {"data": data, "mask": mask}

                # Uso la banda Roja como referencia, o la primera que encuentro
                if info_referencia is None or param == "R":
                    info_referencia = {
                        "crs": src.crs,
                        "transform": src.transform,
                        "width": src.width,
                        "height": src.height,
                        "bounds": src.bounds,
                    }
        except (rasterio.errors.RasterioIOError, OSError) as e:
            print(f"Error al leer el archivo {archivo}: {e}")

    if not datos_bandas:
        raise FileNotFoundError("No se encontraron bandas válidas en el directorio.")

    if info_referencia is None:
        raise FileNotFoundError(
            "No se pudo obtener información de referencia espacial (falta banda roja u otras)."
        )

    return datos_bandas, info_referencia


def procesa_indices(
    datos_bandas: dict[str, dict[str, np.ndarray]],
    info_ref: dict[str, Any],
    ruta_salida: str | os.PathLike,
    lista_indices: list[str],
    crs_salida: str = CRS_SALIDA,
) -> None:
    """Calculo la lista de índices espectrales usando Spyndex y guardo los resultados en formato GeoTIFF.

    Args:
        datos_bandas (dict[str, dict[str, np.ndarray]]): Diccionario con las matrices de datos y máscaras por banda.
        info_ref (dict[str, Any]): Información espacial de referencia (CRS, transformación, dimensiones, límites).
        ruta_salida (str | os.PathLike): Directorio donde se guardarán los archivos GeoTIFF resultantes.
        lista_indices (list[str]): Lista de nombres de índices a calcular (ej: ['NDVI', 'GNDVI']).
        crs_salida (str, opcional): Sistema de Referencia de Coordenadas de salida. Por defecto 'EPSG:32630'.
    """
    os.makedirs(ruta_salida, exist_ok=True)

    # Preparo parámetros para spyndex (bandas + constantes)
    params = {k: v["data"] for k, v in datos_bandas.items()}

    for const in CONSTANTES_SPYNDEX:
        if const in spyndex.constants:
            params[const] = float(spyndex.constants[const].value)

    # Información espacial de origen
    src_crs = info_ref["crs"]
    src_transform = info_ref["transform"]
    src_width = info_ref["width"]
    src_height = info_ref["height"]
    src_bounds = info_ref["bounds"]

    # Calculo transformación de salida
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, crs_salida, src_width, src_height, *src_bounds
    )

    # Creo máscara combinada de todas las bandas cargadas
    # (Si un píxel es inválido en CUALQUIER banda cargada, se enmascara)
    combined_mask: np.ndarray | None = None
    for info in datos_bandas.values():
        if combined_mask is None:
            combined_mask = info["mask"].copy()
        else:
            combined_mask |= info["mask"]

    for indice in lista_indices:
        print(f"Calculando índice: {indice}")
        try:
            calculo = spyndex.computeIndex(index=indice, params=params)
        except Exception as e:  # noqa: BLE001
            print(f"Error calculando {indice}: {e}")
            continue

        # Aseguro tipo y aplico máscara
        calculo = np.array(calculo, dtype=np.float32)

        if combined_mask is not None:
            # Me aseguro que las dimensiones coincidan (spyndex a veces devuelve formas distintas si inputs difieren)
            if calculo.shape == combined_mask.shape:
                calculo[combined_mask] = NODATA_VAL
            else:
                print(f"Advertencia: Dimensiones no coinciden para máscara en {indice}")

        print(f"Guardando índice: {indice}")

        destination = np.zeros((dst_height, dst_width), dtype=np.float32)

        reproject(
            source=calculo,
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=crs_salida,
            resampling=Resampling.nearest,
            src_nodata=NODATA_VAL,
            dst_nodata=NODATA_VAL,
        )

        archivo_salida = os.path.join(ruta_salida, f"{indice}.tif")
        with rasterio.open(
            archivo_salida,
            "w",
            driver="GTiff",
            height=dst_height,
            width=dst_width,
            count=1,
            dtype=rasterio.float32,
            crs=crs_salida,
            transform=dst_transform,
            nodata=NODATA_VAL,
        ) as dst:
            dst.write(destination, 1)


def calcula_indices(
    ruta_entrada: str | os.PathLike,
    ruta_salida: str | os.PathLike,
    lista_indices: list[str] | None = None,
    crs_salida: str = CRS_SALIDA,
) -> None:
    """Orquesto la lectura de bandas y el cálculo de índices de vegetación.

    Args:
        ruta_entrada (str | os.PathLike): Ruta al directorio de entrada con las bandas GeoTIFF.
        ruta_salida (str | os.PathLike): Ruta al directorio donde se exportarán los índices procesados.
        lista_indices (list[str], opcional): Lista con los nombres de los índices a calcular. Por defecto None.
        crs_salida (str, opcional): Sistema de Referencia de Coordenadas de salida. Por defecto 'EPSG:32630'.
    """
    if lista_indices is None:
        print("No se especificaron índices para calcular.")
        return

    datos_bandas, info_ref = carga_bandas(ruta_entrada)
    procesa_indices(
        datos_bandas, info_ref, ruta_salida, lista_indices, crs_salida=crs_salida
    )
    print(f"Proceso completado. Índices guardados en {ruta_salida}")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        ruta_in = sys.argv[1]
        ruta_out = sys.argv[2]
        # El tercer argumento se espera como una cadena separada por comas: "NDVI,GNDVI"
        indices = sys.argv[3].split(",")
        crs = sys.argv[4] if len(sys.argv) >= 5 else CRS_SALIDA
        calcula_indices(ruta_in, ruta_out, indices, crs_salida=crs)
    else:
        print(
            "Uso: python -m modulos.calcula_indices <ruta_entrada> <ruta_salida> <indice1,indice2,...> [crs_salida]"
        )
