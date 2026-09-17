from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modulos.validacion_campo import genera_graficos, procesar_serie_temporal_generico

if __name__ == "__main__":
    # Parámetros generales
    CAPA_QUADRATS = "daliasQuadrats_32630"
    ID_QUADRAT = "id_quadrat"
    COLUMNA_CAMPO_VOL = "fitovol"

    # Directorios raíz
    RUTA_ENTRADAS = Path("entradas")
    RUTA_SALIDAS = Path("salidas")

    # Rutas de entrada
    RUTA_DATOS_CAMPO_VOLUMEN = RUTA_ENTRADAS / "fitovolumen"
    RUTA_PARCELAS = RUTA_ENTRADAS / "infoVectorial.gpkg"
    
    # La ruta de salida de los rasters ahora es la de entrada para la validación
    RUTA_RASTER_VOLUMEN = RUTA_SALIDAS / "modelos_digitales" / "fitovolumen"
    
    # Rutas de salida para gráficos
    RUTA_GRAFICOS = RUTA_SALIDAS / "graficos_fitovolumen"

    # Ruta de salida de la tabla de resumen
    RUTA_CSV_RESUMEN = RUTA_GRAFICOS / "df_resumen_fitovolumen.csv"

    # Validación con datos de campo
    print("\n--- Iniciando Validación de Fitovolumen con Datos de Campo (Desglosado) ---")
    try:
        df_resumen_vol = procesar_serie_temporal_generico(
            dir_tif=RUTA_RASTER_VOLUMEN,
            dir_csv=RUTA_DATOS_CAMPO_VOLUMEN,
            ruta_gpkg=RUTA_PARCELAS,
            capa_gpkg=CAPA_QUADRATS,
            id_columna=ID_QUADRAT.lower(),
            col_campo_csv=COLUMNA_CAMPO_VOL,
            tipo_metrica="suma"
        )

        if not df_resumen_vol.empty:
            # homogeneización a m³/ha y simplificación den columnas
            # El dron viene en m³ por píxel (1m²). Para pasar a hectárea, multiplico por 10000
            df_resumen_vol['valor_calculado'] = df_resumen_vol['valor_calculado'] * 10000
            
            # Recalculo las estadísticas de error usando las columnas limpias
            df_resumen_vol["diferencia"] = df_resumen_vol["valor_calculado"] - df_resumen_vol["valor_campo"]
            df_resumen_vol["error_relativo"] = np.where(
                df_resumen_vol["valor_campo"] > 0,
                (df_resumen_vol["diferencia"] / df_resumen_vol["valor_campo"]) * 100,
                0
            )

            # Parámetros de configuración del gráfico
            modo_grafico = "unificado"         # Puede ser "unificado" o "desglosado" por fecha
            colores_por_fecha=False            # Colorea los puntos por fecha
            filtra_outliers=False              # Activa / desactiva el filtro de outliers
            fecha_inicio="2024-04-01"          # Fecha de inicio para representar datos
            dibuja_intervalo_confianza=True    # Dibuja el intervalo de confianza en la regresión

            # Configuración del gráfico
            figuras, df_metricas = genera_graficos(
                df_resultados=df_resumen_vol,
                etiqueta_eje_y="Fitovolumen Calculado [$m^3/ha$]",
                etiqueta_eje_x="Fitovolumen Campo [$m^3/ha$]",
                titulo_base="Comparación Fitovolumen",
                modo=modo_grafico,
                filtra_outliers=filtra_outliers,
                fecha_inicio=fecha_inicio,
                colores_por_fecha=colores_por_fecha,
                dibuja_intervalo_confianza=dibuja_intervalo_confianza,
                unidades_rmse="$m^3/ha$"
            )
            
            # si no existe la carpeta la creo
            RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
            
            for fig in figuras.values():
                ruta_combinado = RUTA_GRAFICOS / f"00_resumen_fitovolumen_{modo_grafico}.png"
                fig.savefig(ruta_combinado, dpi=300, bbox_inches="tight")
                plt.close(fig)  # Esto libera memoria de la figura
                print(f"  Gráfico guardado en: {ruta_combinado}")

            if not df_metricas.empty:
                ruta_metricas_csv = RUTA_GRAFICOS / "metricas_r2_rmse.csv"
                df_metricas.to_csv(ruta_metricas_csv, index=False)
                print(f"  Tabla de métricas guardada en: {ruta_metricas_csv}")
            
            df_resumen_vol.to_csv(RUTA_CSV_RESUMEN, index=False)
            print(f"\nTAbla resumen guardada en: {RUTA_CSV_RESUMEN}")
        else:
            print("\nNo se ha generado la tabla de resumen. Comprobar que las rutas de los rasters y los csv están bien")

    except Exception as e:
        print(f"\nHa habido un fallo de validación: {e!s}")
