from pathlib import Path

import matplotlib.pyplot as plt

from modulos.validacion_campo import genera_graficos, procesar_serie_temporal_generico

if __name__ == "__main__":
    FECHA_INICIO = "2024-04-01"  # "AAAA-MM-DD"
    FECHA_FIN = None
    INDICE = "ndvi"
    DATOS_BD = "02_rec_total_desn_mant_quadrat"
    TIPO_OTSU = "otsu_2c"
    capa_quadrats = "daliasQuadrats_32630"
    id_quadrat = "id_quadrat"

    # Definición de directorios raíz
    RUTA_ENTRADAS = Path("entradas")
    RUTA_SALIDAS = Path("salidas")

    # Validaciones de parámetros
    indices_validos = {"ndvi", "tvi2"}
    if INDICE not in indices_validos:
        raise ValueError(f"Opciones de índice válidas: {list(indices_validos)}")

    datos_campo_validos = {
        "01_usada_compas": "rec_veg_verde",
        "02_rec_total_desn_mant_quadrat": "cob_total",
        "03_rec_vegetal_quadrat": "rec_veg_ajustada"
    }
    if DATOS_BD not in datos_campo_validos:
        raise ValueError(f"Opciones de datos de campo válidas: {list(datos_campo_validos.keys())}")

    columna_cobertura_campo = datos_campo_validos[DATOS_BD]

    # Rutas específicas de entrada
    ruta_clasificaciones = RUTA_SALIDAS / "segmentaciones" / f"clasificaciones_{TIPO_OTSU}" / INDICE
    ruta_datos_campo = RUTA_ENTRADAS / "cobertura_campo" / DATOS_BD
    ruta_parcelas = RUTA_ENTRADAS / "infoVectorial.gpkg"

    # Rutas específicas de salida
    ruta_graficos = RUTA_SALIDAS / "graficos_cobertura" / "00_sin_tratamiento" / DATOS_BD / INDICE / TIPO_OTSU

    # VALIDACIÓN CON DATOS DE CAMPO
    try:
        df_resumen = procesar_serie_temporal_generico(
            dir_tif=ruta_clasificaciones,
            dir_csv=ruta_datos_campo,
            ruta_gpkg=ruta_parcelas,
            capa_gpkg=capa_quadrats,
            id_columna=id_quadrat.lower(),
            col_campo_csv=columna_cobertura_campo,
            tipo_metrica="cobertura",
            fecha_inicio=FECHA_INICIO,
            fecha_fin=FECHA_FIN
        )

        if not df_resumen.empty:
            figuras_generadas, df_metricas_graficos = genera_graficos(
                df_resultados=df_resumen, 
                # dir_salida=ruta_graficos,
                etiqueta_eje_y="Cobertura calculada [%]",
                etiqueta_eje_x="Cobertura campo [%]",
                titulo_base="Validación cobertura vegetal"
            )

            for indice_grafico, figura in figuras_generadas.items():
                nombre_archivo_figura = f"validacion_cobertura_{DATOS_BD}_{indice_grafico}.png"
                ruta_salida_figura = ruta_graficos / nombre_archivo_figura
                figura.savefig(ruta_salida_figura, dpi=300, bbox_inches='tight')
                print(f"[OK] Gráfico guardado en: {ruta_salida_figura}")
                plt.close(figura)
            
            if not df_metricas_graficos.empty:
                print("\n--- Resumen de Métricas ---")
                print(df_metricas_graficos.to_string(index=False))
                
                ruta_csv_metricas = ruta_graficos / f"df_resumen_cobertura_{DATOS_BD}.csv"
                df_resumen.to_csv(ruta_csv_metricas, index=False)
                print(f"\n[OK] DataFrame resumen guardado en: {ruta_csv_metricas}")

    except Exception as e:
        print(f"\nHa habido un error: {e!s}")
