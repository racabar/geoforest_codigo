"""
Módulo para procesar series temporales y generar gráficos de validación
comparando los datos extraídos del dron frente a los datos reales tomados en campo.
"""
import math
import warnings
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm
from rasterio.mask import mask



warnings.filterwarnings("ignore", category=UserWarning)

def extraer_estadisticas_zonal(ruta_tif, gdf_parcelas, id_columna="id_parcela", tipo_metrica="cobertura"):
    """
    Extrae estadísticas zonales para polígonos desde un raster.
    tipo_metrica: 
      - "cobertura": calculo el porcentaje de píxeles con valor 1.
      - "suma": sumo el valor de todos los píxeles (útil para volumen).
    """
    resultados = []
    ruta_tif = Path(ruta_tif)

    with rasterio.open(ruta_tif) as src:
        nodata_value = src.nodata
        crs_raster = src.crs
        crs_vector = gdf_parcelas.crs

        if crs_raster != crs_vector:
            print(f"  El CRS de {ruta_tif.name} es diferente")
            print(f"      Ráster: {crs_raster} | Vector: {crs_vector}. Reproyectando vector al vuelo...")
            gdf_parcelas = gdf_parcelas.to_crs(crs_raster)

        for _, row in gdf_parcelas.iterrows():
            geom = row["geometry"]
            id_parc = row[id_columna]

            try:
                out_image, _out_transform = mask(src, [geom], crop=True)

                if nodata_value is None:
                    if tipo_metrica == "cobertura":
                        print(f"  El raster {ruta_tif.name} no tiene configurados los valores nulos, los cálculos de cobertura pueden estar mal")
                        conteo_veg = np.sum(out_image == 1)
                        conteo_no_veg = np.sum(out_image == 0)
                        conteo_total = conteo_veg + conteo_no_veg
                    else:
                        conteo_total = out_image.size
                else:
                    conteo_total = np.sum(out_image != nodata_value)

                if conteo_total > 0:
                    if tipo_metrica == "cobertura":
                        conteo_veg = np.sum(out_image == 1)
                        valor_calculado = (conteo_veg / conteo_total) * 100.0
                    elif tipo_metrica == "suma":
                        if nodata_value is not None:
                            pixeles_validos = out_image[out_image != nodata_value]
                        else:
                            pixeles_validos = out_image
                        valor_calculado = np.sum(pixeles_validos)
                    else:
                        valor_calculado = np.nan
                else:
                    valor_calculado = 0.0

                resultados.append({
                    id_columna: id_parc,
                    "valor_calculado": valor_calculado
                })
            except ValueError:
                # Polígono fuera del raster u otro error
                resultados.append({
                    id_columna: id_parc,
                    "valor_calculado": np.nan
                })

    return pd.DataFrame(resultados)


def procesar_serie_temporal_generico(dir_tif, dir_csv, ruta_gpkg, capa_gpkg, id_columna,
                                     col_campo_csv, tipo_metrica="cobertura", fecha_inicio=None, fecha_fin=None):
    """
    Procesa una serie de rasters, extrayendo datos para los polígonos y cruzándolos con los CSV.
    """
    try:
        fecha_inicio_date = pd.to_datetime(fecha_inicio).date() if fecha_inicio else None
        fecha_fin_date = pd.to_datetime(fecha_fin).date() if fecha_fin else None
    except Exception as e:
        print(f"\nERROR de fecha: {e}\n")
        return pd.DataFrame()

    ruta_gpkg = Path(ruta_gpkg)
    dir_tif = Path(dir_tif)
    dir_csv = Path(dir_csv)

    mensaje_capa = f" (Capa: {capa_gpkg})" if capa_gpkg else " (Capa por defecto)"
    print(f"Cargando capa vectorial desde: {ruta_gpkg}{mensaje_capa}")

    if capa_gpkg:
        gdf_parcelas = gpd.read_file(ruta_gpkg, layer=capa_gpkg)
    else:
        gdf_parcelas = gpd.read_file(ruta_gpkg)

    archivos_tif = sorted(dir_tif.glob("*.tif"))
    if not archivos_tif:
        raise FileNotFoundError("No se han encontrado archivos .tif en el directorio especificado")

    if fecha_inicio_date or fecha_fin_date:
        archivos_filtrados = []
        for ruta in archivos_tif:
            try:
                fecha_str = ruta.stem.split('_')[0]
                try:
                    fecha_archivo = datetime.strptime(fecha_str, '%Y%m%d').astimezone().date()  # Uso astimezone() para que coja la zona horaria local y Ruff no dé avisos
                except ValueError:
                    try:
                        fecha_archivo = datetime.strptime(fecha_str, '%y%m%d').astimezone().date()  # Uso astimezone() para que coja la zona horaria local y Ruff no dé avisos
                    except ValueError:
                        print(f"  Omitiendo archivo con nombre de fecha no estándar: {ruta.name}")
                        continue

                if fecha_inicio_date and fecha_archivo < fecha_inicio_date:
                    continue
                if fecha_fin_date and fecha_archivo > fecha_fin_date:
                    continue
                archivos_filtrados.append(ruta)
            except (ValueError, IndexError):
                print(f"  Omitiendo archivo con nombre no estándar: {ruta.name}")
                continue
        
        archivos_tif = archivos_filtrados
        print(f"Procesando {len(archivos_tif)} archivos entre {fecha_inicio_date or 'el inicio'} y {fecha_fin_date or 'el final'}.")

    if not archivos_tif:
        print("No se han encontrado archivos .tif en el rango de fechas especificado")
        return pd.DataFrame()

    lista_comparaciones = []

    for ruta_tif in archivos_tif:
        nombre_base = ruta_tif.stem
        partes_nombre = nombre_base.split("_")
        fecha_str = partes_nombre[0]
        # El sufijo será el resto del nombre (ej. 'ndvi_otsu_2c' o 'Volumen')
        indice_str = "_".join(partes_nombre[1:]) if len(partes_nombre) > 1 else "desconocido"
        
        # Intento convertir fechas cortas (230614) a largas (20230614) para buscar el CSV
        try:
            if len(fecha_str) == 6:
                fecha_csv_str = datetime.strptime(fecha_str, '%y%m%d').astimezone().strftime('%Y%m%d')  # Uso astimezone() para que coja la zona horaria local y Ruff no dé avisos
            else:
                fecha_csv_str = fecha_str
        except ValueError:
            fecha_csv_str = fecha_str

        ruta_csv = dir_csv / f"{fecha_csv_str}.csv"

        print(f"\nProcesando imagen: {nombre_base}")
        print(f"  -> Buscando datos de campo en: {ruta_csv.name}")

        if not ruta_csv.exists():
            print(f"  No se ha encontrado el .csv correspondiente: {ruta_csv}. Se salta la validación para esta imagen.")
            continue

        df_calculado = extraer_estadisticas_zonal(ruta_tif, gdf_parcelas, id_columna, tipo_metrica)
        df_csv = pd.read_csv(ruta_csv)

        if id_columna not in df_csv.columns or col_campo_csv not in df_csv.columns:
            print(f"  La tabla {fecha_csv_str}.csv no tiene las columnas necesarias ('{id_columna}', '{col_campo_csv}').")
            continue

        df_comparacion = pd.merge(df_calculado, df_csv[[id_columna, col_campo_csv]], on=id_columna, how="inner")
        df_comparacion = df_comparacion.rename(columns={col_campo_csv: "valor_campo"})
        df_comparacion["diferencia"] = df_comparacion["valor_calculado"] - df_comparacion["valor_campo"]
        df_comparacion["error_relativo"] = np.where(
            df_comparacion["valor_campo"] > 0,
            (df_comparacion["diferencia"] / df_comparacion["valor_campo"]) * 100,
            0
        )
        df_comparacion["fecha_vuelo"] = fecha_str
        df_comparacion["indice"] = indice_str
        df_comparacion["nombre_imagen"] = nombre_base

        lista_comparaciones.append(df_comparacion)
        print(f"  -> {len(df_comparacion)} parcelas comparadas exitosamente.")

    if lista_comparaciones:
        df_final = pd.concat(lista_comparaciones, ignore_index=True)
        return df_final
    else:
        return pd.DataFrame()


def genera_graficos(df_resultados, etiqueta_eje_y="Calculado", etiqueta_eje_x="Campo",
                    titulo_base="Comparación", modo="desglosado", fecha_inicio=None, fecha_fin=None,
                    filtra_outliers=False, colores_por_fecha=True, dibuja_intervalo_confianza=False, unidades_rmse="%"):
    """
    Genera gráficos de dispersión 1:1 para los resultados de la comparación.
    modo: 
      - "desglosado": Creo una matriz de subplots, uno por fecha de vuelo.
      - "unificado": Creo un único gráfico global con todas las observaciones juntas.
    Devuelvo un diccionario con las figuras generadas y un DataFrame con las métricas.
    """
    if df_resultados.empty:
        print("No hay datos para visualizar.")
        return {}, pd.DataFrame()

    # FILTRADO OPCIONAL POR FECHAS
    if fecha_inicio or fecha_fin:
        try:
            f_inicio = pd.to_datetime(fecha_inicio).date() if fecha_inicio else None
            f_fin = pd.to_datetime(fecha_fin).date() if fecha_fin else None

            def parsear_fecha(f_str):
                try:
                    if len(f_str) == 6:
                        return datetime.strptime(f_str, '%y%m%d').astimezone().date()  # Uso astimezone() para que coja la zona horaria local y Ruff no dé avisos
                    else:
                        return datetime.strptime(f_str, '%Y%m%d').astimezone().date()  # Uso astimezone() para que coja la zona horaria local y Ruff no dé avisos
                except ValueError:
                    return None
            
            fechas_reales = df_resultados["fecha_vuelo"].astype(str).apply(parsear_fecha)
            mascara = pd.Series(True, index=df_resultados.index)
            
            if f_inicio:
                mascara = mascara & (fechas_reales >= f_inicio)
            if f_fin:
                mascara = mascara & (fechas_reales <= f_fin)
                
            df_resultados = df_resultados[mascara].copy()

            if df_resultados.empty:
                print("No hay datos que cumplan el filtro de fechas para visualizar.")
                return {}, pd.DataFrame()
        except Exception as e:
            print(f"  Error aplicando el filtro de fechas en gráficos: {e}")

    indices = df_resultados["indice"].unique()
    metricas_por_fecha = []
    metricas_globales = []
    figuras_generadas = {}

    for indice in indices:
        print(f"\nGenerando gráficos (modo {modo}) para: {indice.upper()}...")
        df_indice = df_resultados[df_resultados["indice"] == indice].copy()

        # FILTRO DE VALORES ATÍPICOS IQR
        if filtra_outliers:
            diferencia = df_indice["valor_calculado"] - df_indice["valor_campo"]
            Q1 = diferencia.quantile(0.25)
            Q3 = diferencia.quantile(0.75)
            IQR = Q3 - Q1
            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR
            
            mascara_iqr = (diferencia >= limite_inf) & (diferencia <= limite_sup)
            n_outliers = (~mascara_iqr).sum()
            df_indice = df_indice[mascara_iqr]
            print(f"  Filtro IQR activado: se han descartado {n_outliers} valores atípicos.")
            
            if df_indice.empty:
                print(f"  No quedan datos tras aplicar el filtro IQR para {indice.upper()}. Omitiendo...")
                continue

        max_global = float(max(df_indice["valor_campo"].max(), df_indice["valor_calculado"].max()))
        if "cobertura" in titulo_base.lower():
            limite_eje = float(max(100.0, max_global * 1.05))
        else:
            limite_eje = float(max_global * 1.05)


        # MODO UNIFICADO (Global)
        if modo == "unificado":
            fig, eje = plt.subplots(figsize=(8, 8))
            y_true_global = df_indice["valor_campo"]
            y_pred_global = df_indice["valor_calculado"]

            if len(df_indice) > 1:
                # matriz_corr = np.corrcoef(y_true_global, y_pred_global)
                # Uso statsmodels para una regresión más completa
                x_const = sm.add_constant(y_true_global)
                modelo = sm.OLS(y_pred_global, x_const).fit()
                
                b, m = modelo.params
                r2_global = modelo.rsquared
                rmse_global = np.sqrt(modelo.mse_resid)
                p_valor_global = modelo.f_pvalue

                print(f"  Ecuación de la recta global: y = {m:.4f}x + {b:.4f}")
                print(f"  p-valor global: {p_valor_global:.4e}")
            else:
                modelo = None
                r2_global, rmse_global, m, b = np.nan, np.nan, np.nan, np.nan
                p_valor_global = np.nan

            metricas_globales.append({
                "indice": indice, "fecha_vuelo": "Todas", 
                "R2": r2_global, "RMSE": rmse_global, "p_valor": p_valor_global, "N_muestras": len(df_indice)
            })

            eje.scatter(y_true_global, y_pred_global, alpha=0.5, c="forestgreen", edgecolors="white", s=60)
            eje.plot([0, limite_eje], [0, limite_eje], "r--", label="Línea 1:1", linewidth=1.5)

            if len(df_indice) > 1:
                x_vals = np.array([0, limite_eje])
                y_vals = m * x_vals + b
                eje.plot(
                    x_vals,
                    y_vals,
                    "b-",
                    label=f"Tendencia global ($y={m:.2f}x{b:+.2f}$)",
                    linewidth=1.5,
                )
                
                if dibuja_intervalo_confianza and modelo:
                    x_sorted = np.sort(y_true_global)
                    intervalo_confianza = modelo.get_prediction(sm.add_constant(x_sorted)).conf_int(alpha=0.05)
                    eje.fill_between(x_sorted,
                                     intervalo_confianza[:, 0],
                                     intervalo_confianza[:, 1],
                                     color='b', alpha=0.15, label='IC 95%')

            eje.set_xlim(0, limite_eje)
            eje.set_ylim(0, limite_eje)
            eje.set_aspect("equal", adjustable="box")

            texto_metricas = f"$R^2$ global: {r2_global:.3f}\nRMSE global: {rmse_global:.2f} {unidades_rmse}\np-valor: {p_valor_global:.2e}"
            props = {
                "boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9, "edgecolor": "lightgray"
            }
            eje.text(
                0.05,
                0.95,
                texto_metricas,
                transform=eje.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )

            tamano_nombre_ejes = 14
            # eje.set_title(f"{titulo_base} Global - {indice.upper()}", fontsize=14, pad=15)
            eje.set_xlabel(etiqueta_eje_x, fontsize=tamano_nombre_ejes)
            eje.set_ylabel(etiqueta_eje_y, fontsize=tamano_nombre_ejes)
            eje.tick_params(axis="both", which="major", labelsize=14)
            eje.grid(True, linestyle="--", alpha=0.5)
            eje.legend(loc="lower right", bbox_to_anchor=(1, 0.025), fontsize=12)

            fig.tight_layout()
            figuras_generadas[indice] = fig


        # MODO DESGLOSADO (Subplots por Fecha)
        elif modo == "desglosado":
            fechas = df_indice["fecha_vuelo"].unique()
            n_fechas = len(fechas)

            if n_fechas == 3:
                # 3 gráficos: 1 centrado arriba y 2 abajo
                fig_comb = plt.figure(figsize=(12, 12))
                gs = fig_comb.add_gridspec(2, 4)
                ax1 = fig_comb.add_subplot(gs[0, 1:3])
                ax2 = fig_comb.add_subplot(gs[1, 0:2])
                ax3 = fig_comb.add_subplot(gs[1, 2:4])
                axes_comb = [ax1, ax2, ax3]
            else:
                cols = min(3, n_fechas) if n_fechas > 0 else 1
                rows = math.ceil(n_fechas / cols) if n_fechas > 0 else 1
                fig_comb, axes_comb_mat = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), squeeze=False)
                axes_comb = axes_comb_mat.flatten()

            for i, fecha in enumerate(fechas):
                df_fecha = df_indice[df_indice["fecha_vuelo"] == fecha]
                y_true = df_fecha["valor_campo"]
                y_pred = df_fecha["valor_calculado"]

                if len(df_fecha) > 1:
                    # Uso statsmodels, que permite más configuración de la regresión
                    x_const = sm.add_constant(y_true)
                    modelo = sm.OLS(y_pred, x_const).fit()
                    
                    b, m = modelo.params
                    r2 = modelo.rsquared
                    rmse = np.sqrt(modelo.mse_resid)
                    p_valor = modelo.f_pvalue

                    print(f"  [{fecha}] p-valor: {p_valor:.4e}")
                else:
                    modelo = None
                    r2, rmse, m, b = np.nan, np.nan, np.nan, np.nan
                    p_valor = np.nan

                metricas_por_fecha.append({"indice": indice, "fecha_vuelo": fecha, "R2": r2, "RMSE": rmse, "p_valor": p_valor})

                eje = axes_comb[i]
                eje.scatter(y_true, y_pred, alpha=0.7, c="forestgreen", edgecolors="white", s=60)
                eje.plot([0, limite_eje], [0, limite_eje], "r--", label="Línea 1:1", linewidth=1.5)

                if len(df_fecha) > 1:
                    x_vals = np.array([0, limite_eje])
                    y_vals = m * x_vals + b
                    eje.plot(x_vals, y_vals, "b-", label=f"Tendencia ($y={m:.2f}x{b:+.2f}$)", linewidth=1.5)

                    if dibuja_intervalo_confianza and modelo:
                        x_sorted = np.sort(y_true)
                        intervalo_confianza = modelo.get_prediction(sm.add_constant(x_sorted)).conf_int(alpha=0.05)
                        eje.fill_between(x_sorted,
                                         intervalo_confianza[:, 0],
                                         intervalo_confianza[:, 1],
                                         color='b', alpha=0.15, label='IC 95%')

                eje.set_xlim(0, limite_eje)
                eje.set_ylim(0, limite_eje)
                eje.set_aspect("equal", adjustable="box")

                texto_metricas = f"$R^2$: {r2:.2f}\nRMSE: {rmse:.2f} {unidades_rmse}\np-valor: {p_valor:.2e}"
                props = {"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9, "edgecolor": "lightgray"}
                eje.text(0.05, 0.95, texto_metricas, transform=eje.transAxes, fontsize=14, verticalalignment="top", bbox=props)

                try:
                    if len(str(fecha)) == 6:
                        fecha_formateada = datetime.strptime(str(fecha), '%y%m%d').astimezone().strftime('%d-%m-%Y')  # Uso astimezone() para que coja la zona horaria local y Ruff no dé avisos
                    else:
                        fecha_formateada = datetime.strptime(str(fecha), '%Y%m%d').astimezone().strftime('%d-%m-%Y')  # Uso astimezone() para que coja la zona horaria local y Ruff no dé avisos
                except ValueError:
                    fecha_formateada = fecha

                tamano_nombre_ejes = 14
                # eje.set_title(f"{titulo_base} - {indice.upper()}", fecha_formateada, fontsize=16, pad=10)
                eje.set_xlabel(etiqueta_eje_x, fontsize=tamano_nombre_ejes)
                eje.set_ylabel(etiqueta_eje_y, fontsize=tamano_nombre_ejes)
                eje.tick_params(axis="both", which="major", labelsize=14)
                eje.grid(True, linestyle="--", alpha=0.5)
                eje.legend(loc="lower right", fontsize=8)

            for j in range(n_fechas, len(axes_comb)):
                fig_comb.delaxes(axes_comb[j])

            # Ajusto los márgenes de los subplots para evitar solapamientos
            fig_comb.tight_layout(pad=2.0)
            
            figuras_generadas[indice] = fig_comb

    # Preparo DataFrame de métricas según el modo seleccionado
    df_metricas = pd.DataFrame()
    if modo == "unificado" and metricas_globales:
        df_metricas = pd.DataFrame(metricas_globales)
    elif modo == "desglosado" and metricas_por_fecha:
        df_metricas = pd.DataFrame(metricas_por_fecha)
        df_metricas = df_metricas[["indice", "fecha_vuelo", "R2", "RMSE", "p_valor"]]

    return figuras_generadas, df_metricas
