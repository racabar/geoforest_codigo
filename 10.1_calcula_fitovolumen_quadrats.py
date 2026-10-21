"""
Script para calcular el fitovolumen (m³/ha) para cada quadrat y fecha a partir
de la serie temporal de rasters de fitovolumen generados por el UAV.

El resultado se guarda en un único archivo CSV consolidado.
"""

import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

# Ocultar advertencias de depreciación si es necesario
warnings.filterwarnings("ignore", category=FutureWarning)


def calcular_fitovolumen_por_quadrat(ruta_tif, gdf_parcelas, id_columna="id_quadrat"):
    """
    Calcula la suma de los valores de un raster de fitovolumen para cada parcela.

    Args:
        ruta_tif (Path): Ruta al archivo GeoTIFF de fitovolumen.
        gdf_parcelas (gpd.GeoDataFrame): GeoDataFrame con las geometrías de las parcelas.
        id_columna (str): Nombre de la columna identificadora en gdf_parcelas.

    Returns:
        pd.DataFrame: DataFrame con id_columna y el valor de fitovolumen calculado.
    """
    resultados = []

    with rasterio.open(ruta_tif) as src:
        # Asegurar que el CRS del vector coincida con el del raster
        if gdf_parcelas.crs != src.crs:
            gdf_parcelas = gdf_parcelas.to_crs(src.crs)

        for _, parcela in gdf_parcelas.iterrows():
            geom = parcela["geometry"]
            id_parcela = parcela[id_columna]

            try:
                # Recortar el raster con la geometría de la parcela
                out_image, _ = mask(src, [geom], crop=True)
                nodata_value = src.nodata

                # Enmascarar valores nodata para excluirlos de la suma
                if nodata_value is not None:
                    out_image = np.ma.masked_equal(out_image, nodata_value)

                # Sumar todos los valores de los píxeles dentro de la parcela
                # El raster ya está en cm³, por lo que la suma es el volumen total en cm³
                volumen_total_cm3 = np.sum(out_image)

                resultados.append({
                    id_columna: id_parcela,
                    "fitovol_uav_cm3": volumen_total_cm3
                })
            except ValueError:
                # Esto puede ocurrir si la parcela está fuera del extent del raster
                resultados.append({
                    id_columna: id_parcela,
                    "fitovol_uav_cm3": 0
                })

    return pd.DataFrame(resultados)


if __name__ == "__main__":
    # Directorios raíz
    ruta_base = Path(__file__).resolve().parent
    ruta_entradas = ruta_base / "entradas"
    ruta_salidas = ruta_base / "salidas"

    CAPA_QUADRATS = "daliasQuadrats_32630"
    ID_QUADRAT = "id_quadrat"

    # Rutas de entrada
    RUTA_PARCELAS = ruta_entradas / "infoVectorial.gpkg"
    RUTA_RASTERS_VOLUMEN = ruta_salidas / "con_sombras" / "07_fitovolumen"

    # Ruta de salida para el CSV
    RUTA_CSV_SALIDA = ruta_entradas / "biomasa" / "fitovolumen_completo.csv"
    RUTA_CSV_SALIDA.parent.mkdir(parents=True, exist_ok=True)

    # PROCESADO DE LA SERIE TEMPORAL
    print("\n--- Iniciando cálculo de Fitovolumen UAV para toda la serie temporal ---")

    # Cargar la capa de quadrats
    gdf_quadrats = gpd.read_file(RUTA_PARCELAS, layer=CAPA_QUADRATS)

    # Buscar todos los rasters de volumen
    archivos_tif = sorted(RUTA_RASTERS_VOLUMEN.glob("*_Volumen_cm3.tif"))
    if not archivos_tif:
        raise FileNotFoundError(f"No se encontraron rasters de volumen en: {RUTA_RASTERS_VOLUMEN}")

    lista_resultados = []

    for ruta_tif in archivos_tif:
        # Extraer fecha del nombre de archivo, ej: "20240516_Volumen_cm3.tif" -> "20240516"
        try:
            fecha_vuelo = ruta_tif.stem.split('_')[0]
            # Validar que la fecha tenga el formato correcto
            datetime.strptime(fecha_vuelo, '%Y%m%d').astimezone()
        except (IndexError, ValueError):
            print(f"  [!] Omitiendo archivo con nombre no estándar: {ruta_tif.name}")
            continue

        print(f"  Procesando fecha: {fecha_vuelo}...")
        df_fecha = calcular_fitovolumen_por_quadrat(ruta_tif, gdf_quadrats, ID_QUADRAT)
        df_fecha['fecha_vuelo'] = datetime.strptime(fecha_vuelo, '%Y%m%d').astimezone().strftime('%Y-%m-%d')
        lista_resultados.append(df_fecha)

    # Unir todos los dataframes y preparar el archivo final
    df_final = pd.concat(lista_resultados, ignore_index=True)
    # Conversión de unidades: cm³/m² -> m³/ha
    # 1. cm³ a m³ -> dividir por 1,000,000
    # 2. El resultado está por m² (tamaño del quadrat), para pasarlo a hectárea (10,000 m²) -> multiplicar por 10,000
    # Factor total: 10,000 / 1,000,000 = 0.01
    df_final['fitovol_uav'] = df_final['fitovol_uav_cm3'] / 100
    df_final = df_final[["id_quadrat", "fecha_vuelo", "fitovol_uav"]]

    # Guardar en CSV
    df_final.to_csv(RUTA_CSV_SALIDA, index=False, float_format='%.2f')
    print(f"\n[✓] Proceso completado. Archivo guardado en: {RUTA_CSV_SALIDA}")