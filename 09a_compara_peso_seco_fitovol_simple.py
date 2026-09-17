import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from modulos.valida_supuestos_estadisticos import (
    graficar_diagnosticos,
    validar_supuestos_ols,
)


def compara_peso_seco_fitovol(
    ruta_csv,
    biomasa,
    fitovolumen,
    metodo_outliers="none",
    transformacion_log=False,
    guardar_figura=True,
    mostrar_figura=False,
    mostrar_outliers_grafico=True,
    dir_salida=Path("salidas") / "graficos_fitovolumen_biomasa",
):

    # Cargo los datos
    df = pd.read_csv(ruta_csv)
    
    # Conversión de unidades
    # Convertir peso_seco (g/m²) a biomasa seca (kg/ha)
    # Factor: / 1000 (g a kg) * 10000 (m² a ha) = * 10
    # Para tener el cálculo en kg/ha hay que descomentar esta línea
    # df[biomasa] = df[biomasa] * 10
    
    n_removidas_nan = len(df) - len(df[["id_quadrat", biomasa, fitovolumen]].dropna())
    if n_removidas_nan > 0:
        print(f"Se han eliminado {n_removidas_nan} muestras con valores nulos (nan)")
        
    # Llamada al nuevo módulo unificado
    from modulos.regresiones import ajustar_modelo_regresion
    resultados_reg = ajustar_modelo_regresion(
        df=df,
        col_biomasa=biomasa,
        cols_predictoras=[fitovolumen],
        metodo_outliers=metodo_outliers,
        transformacion_log=transformacion_log
    )
    
    # Descomprimo el diccionario qeu devuelve la función con los resultados de la regresión
    modelo = resultados_reg["modelo"]
    rmse = resultados_reg["rmse"]
    mae = resultados_reg["mae"]
    n_muestras = resultados_reg["n_muestras_validas"]
    n_outliers = resultados_reg["n_outliers"]
    df_inliers = resultados_reg["datos_inliers"]
    df_outliers = resultados_reg["datos_outliers"]
    diagnosticos = resultados_reg["diagnosticos"]
    predicciones = resultados_reg["predicciones"]
    
    # Preparo datos para el gráfico
    x = df_inliers[fitovolumen].values  # x es el fitovolumen
    y = df_inliers[biomasa].values      # y es el peso seco
    
    df_limpio = pd.concat([df_inliers, df_outliers]) # Reconstruyo df_limpio para los ejes del plot
    
    # coeficientes
    intercepto = modelo.params.iloc[0]
    pendiente = modelo.params.iloc[1]

    # métricas adicionales
    r2 = modelo.rsquared
    r2_ajustado = modelo.rsquared_adj
    p_valor = modelo.f_pvalue

    # Estadísticas del modelo
    resumen = modelo.summary()

    # Configuro los textos según la transformación
    if transformacion_log:
        str_ecuacion = f"ln(biomasa) = {pendiente:.6f} * ln(fitovol) + {intercepto:.6f}"
        str_rmse = f"{rmse:.6f} (log g/m²)"
        str_mae = f"{mae:.6f} (log g/m²)"
    else:
        str_ecuacion = f"biomasa_kg_ha = {pendiente:.6f} * fitovol + {intercepto:.6f}"
        str_rmse = f"{rmse:.6f} g/m²"
        str_mae = f"{mae:.6f} g/m²"

    # Validación de supuestos estadísticos
    diagnosticos = validar_supuestos_ols(modelo)

    # Imprimo resultados en la consola
    print("\n" + "=" * 70)

    print("\nVALIDACIÓN DE SUPUESTOS ESTADÍSTICOS:")
    print(
        f"  Normalidad (Shapiro-Wilk)    : {diagnosticos['normalidad']['interpretacion']} (p={diagnosticos['normalidad']['p_valor']:.4f})"
    )
    print(
        f"  Homocedasticidad (B-Pagan)   : {diagnosticos['homocedasticidad']['interpretacion']} (p={diagnosticos['homocedasticidad']['p_valor']:.4f})"
    )
    print(
        f"  Independencia (D-Watson)     : {diagnosticos['independencia']['interpretacion']} (stat={diagnosticos['independencia']['estadistico']:.2f})"
    )

    print("\nRESULTADOS DE LA REGRESIÓN LINEAL")
    print("=" * 70)
    print(f"Filtro de outliers: {metodo_outliers}")
    print(f"Transformación Logarítmica: {'Sí' if transformacion_log else 'No'}")
    print("\nMétricas:")
    print(f"  R²: {r2:.6f}")
    print(f"  R² Ajustado: {r2_ajustado:.6f}")
    print(f"  RMSE: {str_rmse}")
    print(f"  MAE: {str_mae}")
    print(f"  p-valor: {p_valor:.4e}")
    print(f"  N muestras: {n_muestras} (Outliers removidos: {n_outliers})")
    print("\nResumen del modelo:")
    print(resumen)
    print("\nEcuación de la recta (Modelo Predictivo):")
    print(f"  {str_ecuacion}")
    print("\n" + "=" * 70)

    # CREACI´NO DEL GRÁFICO
    _, ax = plt.subplots(figsize=(12, 10))

    # Scatter plot de datos válidos
    ax.scatter(
        x,
        y,
        alpha=0.7,
        s=60,
        color="forestgreen",
        edgecolors="white",
        linewidth=0.5,
        label="Datos válidos",
    )

    # Scatter plot de outliers (si los hay y quiero mostrarlos)
    if not df_outliers.empty and mostrar_outliers_grafico:
        x_out = df_outliers[fitovolumen].values
        y_out = df_outliers[biomasa].values
        ax.scatter(
            x_out,
            y_out,
            alpha=0.6,
            s=50,
            color="crimson",
            marker="X",
            label="Valores atípicos",
        )

    # Línea de regresión
    # Aumento un poco el rango de X para abarcar tanto los datos válidos como los atípicos
    x_min, x_max = df_limpio[fitovolumen].min(), df_limpio[fitovolumen].max()
    x_plot = np.linspace(x_min, x_max, 100)
    y_regresion = pendiente * x_plot + intercepto

    ax.plot(
        x_plot,
        y_regresion,
        color="b",
        linestyle="-",
        linewidth=2,
        label="Recta de regresión",
    )

    # Intervalo de confianza al 95%
    intervalo_confianza = modelo.get_prediction(sm.add_constant(x_plot)).conf_int(
        alpha=0.05
    )
    ax.fill_between(
        x_plot,
        intervalo_confianza[:, 0],
        intervalo_confianza[:, 1],
        color="b",
        alpha=0.15,
        label="IC 95%",
    )

    # Etiquetas y título
    if transformacion_log:
        ax.set_xlabel("Ln fitovolumen UAV", fontsize=16)
        ax.set_ylabel("Ln biomasa seca", fontsize=16)
    else:
        ax.set_xlabel("Fitovolumen UAV (m³/ha)", fontsize=16)
        ax.set_ylabel("Biomasa seca (g/m²)", fontsize=16)

    # titulo = "Regresión Lineal: Fitovolumen vs Biomasa Seca"
    # if transformacion_log:
    #     titulo += "\n(Escala Logarítmica)"
    # if metodo_outliers != "none":
    #     titulo += f"\n(Filtro IQR: {metodo_outliers})"
    # ax.set_title(titulo, fontsize=16, fontweight="bold", pad=20)

    # Texto con las métricas en la esquina del gráfico
    unidad_rmse = "(log g/m²)" if transformacion_log else "g/m²"
    texto_metricas = f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f} {unidad_rmse}\np-valor = {p_valor:.2e}"

    props = {
        "boxstyle": "round, pad=0.5", "facecolor": "white", "alpha": 0.8, "edgecolor": "black"
    }
    ax.text(
        0.05,
        0.95,
        texto_metricas,
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=props,
    )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=14, facecolor="white", edgecolor="black", framealpha=1.0)

    plt.tight_layout()

    # Guardo la figura
    if guardar_figura:
        # sufijo_filtro = f"_{metodo_outliers}" if metodo_outliers != "none" else ""
        sufijo_log = "_log" if transformacion_log else "_linear"
        nombre_base = f"regresion{sufijo_log}_{fitovolumen}_peso_seco"

        dir_salida = Path(dir_salida)
        dir_salida.mkdir(parents=True, exist_ok=True)

        # Gráfico resultado
        ruta_figura = dir_salida / f"{nombre_base}.png"
        plt.savefig(ruta_figura, dpi=300, bbox_inches="tight")
        print(f"\n[✓] Figura guardada en: {ruta_figura}")

        # Gráficos de la validación de supuetos estadísticos
        ruta_diag = dir_salida / f"{nombre_base}_diagnostico.png"
        graficar_diagnosticos(modelo, ruta_salida=ruta_diag, mostrar_figura=False)
        print(f"[✓] Panel de diagnóstico guardado en: {ruta_diag}")

        # Guardo los resultados de la regresión en un archivo de texto para tenerlos a mano
        ruta_resultados_txt = ruta_figura.with_name(f"{nombre_base}_resultados.txt")

        texto_resultados = (
            f"{'=' * 70}\n"
            f"RESULTADOS DE LA REGRESIÓN LINEAL\n"
            f"{'=' * 70}\n\n"
            f"Filtro de outliers aplicado: {metodo_outliers}\n"
            f"Transformación Logarítmica: {'Sí' if transformacion_log else 'No'}\n"
            f"Outliers excluidos: {n_outliers}\n\n"
            f"Ecuación del modelo predictivo:\n"
            f"  {str_ecuacion}\n\n"
            f"Métricas:\n"
            f"  R²: {r2:.6f}\n"
            f"  R² Ajustado: {r2_ajustado:.6f}\n"
            f"  RMSE: {str_rmse}\n"
            f"  MAE: {str_mae}\n"
            f"  p-valor: {p_valor:.4e}\n"
            f"  N muestras finales: {n_muestras}\n\n"
            f"--- DIAGNÓSTICO DE SUPUESTOS OLS ---\n"
            f"  Normalidad (Shapiro-Wilk)  : {diagnosticos['normalidad']['interpretacion']} (p-valor: {diagnosticos['normalidad']['p_valor']:.4f})\n"
            f"  Homocedasticidad (B-Pagan) : {diagnosticos['homocedasticidad']['interpretacion']} (p-valor: {diagnosticos['homocedasticidad']['p_valor']:.4f})\n"
            f"  Independencia (D-Watson)   : {diagnosticos['independencia']['interpretacion']} (Estadístico: {diagnosticos['independencia']['estadistico']:.4f})\n\n"
            f"Resumen del modelo (Statsmodels):\n"
            f"{resumen}\n"
            f"{'=' * 70}\n"
        )
        ruta_resultados_txt.write_text(texto_resultados, encoding="utf-8")
        print(f"[✓] Resultados de la regresión guardados en: {ruta_resultados_txt}")

    # Saco la figura
    if mostrar_figura:
        plt.show()
    else:
        plt.close()

    # Guardo las métricas en un diccionario
    resultados = {
        "ecuacion": f"y = {pendiente:.6f}x + {intercepto:.6f}",
        "pendiente": pendiente,
        "intercepto": intercepto,
        "r2": r2,
        "r2_ajustado": r2_ajustado,
        "rmse": rmse,
        "mae": mae,
        "p_valor": p_valor,
        "n_muestras_validas": n_muestras,
        "n_outliers": n_outliers,
        "modelo": modelo,
        "datos_inliers": df_inliers,
        "datos_outliers": df_outliers,
        "predicciones": predicciones,
        "diagnosticos": diagnosticos,
    }

    return resultados


if __name__ == "__main__":
    RUTA_ENTRADAS = Path("entradas")
    RUTA_SALIDAS = Path("salidas")

    RUTA_CSV = RUTA_ENTRADAS / "biomasa" / "biomasa.csv"

    # Escenario A:
    #   peso_seco - fitovol_uav
    #   Método outliers = none
    #   Transformación logarítmica = False
    # Escenario B:
    #   peso_seco - fitovol_uav_ndvi
    #   Método outliers = none
    #   Transformación logarítmica = False
    # Escenario C:
    #   peso_seco - fitovol_uav
    #   Método outliers = none
    #   Transformación logarítmica = True
    # Escenario D: 
    #   peso_seco - fitovol_uav_ndvi
    #   Método outliers = none
    #   Transformación logarítmica = True
    # Escenarios E y F: script 12b_compara_peso_seco_fitovol_ndvi.py
    COLUMNA_BIOMASA = "peso_seco"
    COLUMNA_FITOVOLUMEN = "fitovol_uav_ndvi"

    # Las opciones son
    #   - "univariado": se aplica a los valores de cada variable
    #   - "residuos": se aplica a los residuos después de copmarar
    #   - "none": no se filtran los outliers
    # Si se pasa una lista con varias opciones calcula las tres ["none", "univariado", "residuos"]
    METODOS_OUTLIERS = ["none"]
    TRANSFORMACION_LOGARITMICA = True  # Interruptor para cambiar al modelo logarítmico
    
    RUTA_GUARDADO = RUTA_SALIDAS / "graficos_fitovolumen_biomasa" / METODOS_OUTLIERS[0]

    # Validar que el archivo existe
    if not RUTA_CSV.exists():
        print(f"Archivo no encontrado: {RUTA_CSV}")
        sys.exit(1)

    for metodo in METODOS_OUTLIERS:
        print(
            f"\n--- Iniciando análisis con método: {metodo.upper()} (Logarítmico: {TRANSFORMACION_LOGARITMICA}) ---"
        )
        try:
            resultados = compara_peso_seco_fitovol(
                ruta_csv=RUTA_CSV,
                biomasa=COLUMNA_BIOMASA,
                fitovolumen=COLUMNA_FITOVOLUMEN,
                metodo_outliers=metodo,                         # Método de cálculo de los valores atípicos
                transformacion_log=TRANSFORMACION_LOGARITMICA,  # Opción logarítmica
                guardar_figura=True,
                mostrar_figura=False,                           # Si lo dejo en False guarda las 3 del tirón
                mostrar_outliers_grafico=True,                  # Muestra o no los outliers en el gráfico
                dir_salida=RUTA_GUARDADO,
            )

            print(f" Proceso completado para el método de outliers: {metodo}")

        except Exception as e:
            print(f"\nHa habido un problema con el método {metodo}: {e!s}")
