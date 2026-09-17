import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from modulos.validacion_campo import genera_graficos

# Oculto advertencias menores de pandas/geopandas si es necesario para que no metan mucho ruido en la consola
warnings.filterwarnings("ignore", category=UserWarning)


def calcular_superficie_vegetacion(ruta_tif, gdf_parcelas, id_columna="id_parcela"):
    """
    Calcula el área de vegetación (píxeles = 1) para cada parcela en una imagen TIF,
    ignorando explícitamente los valores nodata.
    """
    resultados = []
    ruta_tif = Path(ruta_tif)

    with rasterio.open(ruta_tif) as src:
        nodata_value = src.nodata
        crs_raster = src.crs
        crs_vector = gdf_parcelas.crs

        if crs_raster != crs_vector:
            print(
                f"  [!] Reproyectando vector al vuelo para que coincida con {ruta_tif.name}..."
            )
            gdf_parcelas = gdf_parcelas.to_crs(crs_raster)

        res_x, res_y = src.res
        area_pixel = res_x * res_y

        for _, row in gdf_parcelas.iterrows():
            geom = row["geometry"]
            id_parc = row[id_columna]

            try:
                out_image, _out_transform = mask(src, [geom], crop=True)

                if nodata_value is None:
                    # hago un fallback si no hay nodata definido
                    conteo_veg = np.sum(out_image == 1)
                    conteo_no_veg = np.sum(out_image == 0)
                    conteo_total = conteo_veg + conteo_no_veg
                else:
                    # Recuento robusto ignorando nodata
                    conteo_veg = np.sum(out_image == 1)
                    conteo_total = np.sum(out_image != nodata_value)

                cobertura_quadrat_valor_absoluto = conteo_veg * area_pixel

                if conteo_total > 0:
                    porcentaje_cobertura = (conteo_veg / conteo_total) * 100.0
                else:
                    porcentaje_cobertura = 0.0

                resultados.append(
                    {
                        id_columna: id_parc,
                        "cobertura_otsu": porcentaje_cobertura,
                        "cobertura_quadrat_valor_absoluto": cobertura_quadrat_valor_absoluto,
                    }
                )
            except ValueError:
                resultados.append(
                    {
                        id_columna: id_parc,
                        "cobertura_otsu": np.nan,
                        "cobertura_quadrat_valor_absoluto": np.nan,
                    }
                )

    return pd.DataFrame(resultados)


def procesar_serie_temporal(
    dir_tif,
    dir_csv,
    ruta_gpkg,
    capa_gpkg=None,
    id_columna="id_parcela",
    col_area_csv="area_vegetacion",
    fecha_inicio=None,
    fecha_fin=None,
):
    """
    Procesa un rango de fechas, cruza con los CSV y devuelve un DataFrame unificado.
    Las fechas pueden ser objetos `date` o strings "AAAA-MM-DD".
    """
    try:
        fecha_inicio_date = pd.to_datetime(fecha_inicio).date() if fecha_inicio else None
        fecha_fin_date = pd.to_datetime(fecha_fin).date() if fecha_fin else None
    except Exception as e:
        print(f"\nFecha inválida: {e}\n")
        return pd.DataFrame()

    ruta_gpkg = Path(ruta_gpkg)
    dir_tif = Path(dir_tif)
    dir_csv = Path(dir_csv)

    if capa_gpkg:
        gdf_parcelas = gpd.read_file(ruta_gpkg, layer=capa_gpkg)
    else:
        gdf_parcelas = gpd.read_file(ruta_gpkg)

    archivos_tif = sorted(dir_tif.glob("*.tif"))
    if not archivos_tif:
        raise FileNotFoundError("No se han encontrado archivos .tif.")

    # Filtrado por rango de fechas si se especifica
    if fecha_inicio_date or fecha_fin_date:
        archivos_filtrados = []
        for ruta in archivos_tif:
            try:
                fecha_str = ruta.stem.split("_")[0]
                fecha_archivo = datetime.strptime(fecha_str, "%Y%m%d").date()  # noqa: DTZ007

                if fecha_inicio_date and fecha_archivo < fecha_inicio_date:
                    continue
                if fecha_fin_date and fecha_archivo > fecha_fin_date:
                    continue
                archivos_filtrados.append(ruta)
            except (ValueError, IndexError):
                print(f"  [!] Se omite el archivo con nombre no estándar: {ruta.name}")
                continue

        archivos_tif = archivos_filtrados
        print(
            f"Procesando {len(archivos_tif)} archivos entre {fecha_inicio_date or 'el inicio'} y {fecha_fin_date or 'el final'}."
        )

    if not archivos_tif:
        print("No se han encontrado archivos TIF en el rango de fechas especificado.")
        return pd.DataFrame()

    with rasterio.open(archivos_tif[0]) as src_tmp:
        crs_base_raster = src_tmp.crs

    if gdf_parcelas.crs != crs_base_raster:
        print(f"Reproyectando capa vectorial globalmente a {crs_base_raster}...")
        gdf_parcelas = gdf_parcelas.to_crs(crs_base_raster)

    lista_comparaciones = []

    for ruta_tif in archivos_tif:
        nombre_base = ruta_tif.stem
        fecha_str = nombre_base.split("_")[0]
        ruta_csv = dir_csv / f"{fecha_str}.csv"

        if not ruta_csv.exists():
            continue

        df_calculado = calcular_superficie_vegetacion(
            ruta_tif, gdf_parcelas, id_columna
        )
        df_csv = pd.read_csv(ruta_csv)

        if id_columna not in df_csv.columns or col_area_csv not in df_csv.columns:
            continue

        df_comparacion = pd.merge(
            df_calculado, df_csv[[id_columna, col_area_csv]], on=id_columna, how="inner"
        )
        df_comparacion = df_comparacion.rename(
            columns={col_area_csv: "cobertura_campo"}
        )
        df_comparacion["Fecha_ID"] = nombre_base

        lista_comparaciones.append(df_comparacion)
        print(f"Procesado: {nombre_base} ({len(df_comparacion)} parcelas)")

    if lista_comparaciones:
        return pd.concat(lista_comparaciones, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    FECHA_INICIO = "2024-04-01"  # "AAAA-MM-DD"
    FECHA_FIN = None
    INDICE = "ndvi"
    DATOS_BD = "02_rec_total_desn_mant_quadrat"
    TIPO_OTSU = "otsu_2c"

    RUTA_SALIDAS = Path("salidas")
    RUTA_ENTRADAS = Path("entradas")

    ruta_clasificaciones = (
        RUTA_SALIDAS / "segmentaciones" / f"clasificaciones_{TIPO_OTSU}" / INDICE
    )
    ruta_parcelas = RUTA_ENTRADAS / "infoVectorial.gpkg"

    capa_quadrats = "daliasQuadrats_32630"
    id_quadrat = "id_quadrat"

    datos_campo = {
        "01_usada_compas": "rec_veg_verde",
        "02_rec_total_desn_mant_quadrat": "cob_total",
        "03_rec_vegetal_quadrat": "rec_veg_ajustada",
    }
    if DATOS_BD not in datos_campo:
        raise ValueError(f"Opciones válidas: {list(datos_campo.keys())}")

    ruta_datos_campo = RUTA_ENTRADAS / "cobertura_campo" / DATOS_BD
    ruta_graficos = (
        RUTA_SALIDAS
        / "graficos_cobertura"
        / "00_sin_tratamiento"
        / DATOS_BD
        / INDICE
        / TIPO_OTSU
    )
    columna_cobertura_campo = datos_campo[DATOS_BD]

    try:
        df_resumen = procesar_serie_temporal(
            dir_tif=ruta_clasificaciones,
            dir_csv=ruta_datos_campo,
            ruta_gpkg=ruta_parcelas,
            capa_gpkg=capa_quadrats,
            id_columna=id_quadrat,
            col_area_csv=columna_cobertura_campo,
            fecha_inicio=FECHA_INICIO,
            fecha_fin=FECHA_FIN,
        )

        if not df_resumen.empty:
            df_adaptado = df_resumen.rename(
                columns={
                    "cobertura_campo": "valor_campo",
                    "cobertura_otsu": "valor_calculado",
                    "Fecha_ID": "fecha_vuelo"
                }
            ).copy()
            df_adaptado["indice"] = INDICE

            ruta_graficos.mkdir(parents=True, exist_ok=True)
            
            figuras, _ = genera_graficos(
                df_adaptado,
                etiqueta_eje_y="Cobertura vegetal calculada [%]",
                etiqueta_eje_x="Cobertura vegetal medida en campo [%]",
                titulo_base="Comparación global de cobertura vegetal",
                modo="unificado",
            )
            
            if INDICE in figuras:
                fig = figuras[INDICE]
                ruta_grafico = ruta_graficos / "grafico_global.png"
                fig.savefig(ruta_grafico, dpi=300, bbox_inches="tight")
                plt.close(fig)

    except Exception as e:
        print(f"\nHa habido un error: {e!s}")
