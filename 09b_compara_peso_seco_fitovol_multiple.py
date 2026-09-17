from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from modulos.valida_supuestos_estadisticos import graficar_diagnosticos


def regresion_multiple_biomasa(
    ruta_csv,
    var1,
    var2,
    var3,
    metodo_outliers="residuos",
    transformacion_log=True,
    guardar_figura=True,
    mostrar_figura=False,
    dir_salida=Path("salidas") / "graficos_multiple",
):

    print(f"\nRegresión Múltiple: {var1} ~ {var2} + {var3}")

    # Cargo los datos
    df = pd.read_csv(ruta_csv)

    # Llamada al nuevo módulo unificado
    from modulos.regresiones import ajustar_modelo_regresion
    resultados_reg = ajustar_modelo_regresion(
        df=df,
        col_biomasa=var1,
        cols_predictoras=[var2, var3],
        metodo_outliers=metodo_outliers,
        transformacion_log=transformacion_log
    )
    
    # Descomprimo el diccionario qeu devuelve la función con los resultados de la regresión
    modelo = resultados_reg["modelo"]
    rmse = resultados_reg["rmse"]
    mae = resultados_reg["mae"]
    df_inliers = resultados_reg["datos_inliers"]
    df_outliers = resultados_reg["datos_outliers"]
    diagnosticos = resultados_reg["diagnosticos"]
    vif_data = resultados_reg["vif_data"]
    predicciones = resultados_reg["predicciones"]
    
    # Preparo datos para el resto del script (gráficos, etc)
    X = df_inliers[[var2, var3]]
    y = df_inliers[var1]
    
    p_valor = modelo.f_pvalue

    # Imprimmo los resultados
    unidad = "(log g/m²)" if transformacion_log else "g/m²"
    ecuacion = f"{var1} = {modelo.params.iloc[0]:.4f} + {modelo.params.iloc[1]:.4f}*{var2} + {modelo.params.iloc[2]:.4f}*{var3}"

    resultados_texto = []
    resultados_texto.append("=" * 70)
    resultados_texto.append("RESULTADOS REGRESIÓN MÚLTIPLE (OLS)")
    resultados_texto.append("=" * 70)
    resultados_texto.append(f"Ecuación: {ecuacion}")
    resultados_texto.append("\nMétricas Predictivas:")
    resultados_texto.append(f"  R² Ajustado : {modelo.rsquared_adj:.4f}")
    resultados_texto.append(f"  RMSE        : {rmse:.4f} {unidad}")
    resultados_texto.append(f"  MAE         : {mae:.4f} {unidad}")

    resultados_texto.append("\nAnálisis de Multicolinealidad (VIF):")
    for idx, row in vif_data.iterrows():
        if row["Variable"] != "const":
            alerta = " (¡Alerta! Alto solapamiento)" if row["VIF"] > 5 else " (Normal)"
            resultados_texto.append(
                f"  {row['Variable']:<15}: {row['VIF']:.2f}{alerta}"
            )

    resultados_texto.append("\nDiagnóstico de Supuestos:")
    resultados_texto.append(
        f"  Normalidad (S-W) : {diagnosticos['normalidad']['interpretacion']} (p={diagnosticos['normalidad']['p_valor']:.4f})"
    )
    resultados_texto.append(
        f"  Homocedasticidad : {diagnosticos['homocedasticidad']['interpretacion']} (p={diagnosticos['homocedasticidad']['p_valor']:.4f})"
    )
    resultados_texto.append(
        f"  Independencia    : {diagnosticos['independencia']['interpretacion']} (stat={diagnosticos['independencia']['estadistico']:.2f})"
    )
    resultados_texto.append("=" * 70)
    resultados_texto.append(
        str(modelo.summary().tables[1])
    )  # Imprimo solo la tabla de coeficientes
    resultados_texto.append("=" * 70)

    texto_final = "\n".join(resultados_texto)
    print("\n" + texto_final)

    # Gráficos
    # Gráfico 3D del plano de regresión
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Puntos reales
    ax.scatter(
        X[var2], X[var3], y, c="forestgreen", marker="o", s=50, alpha=0.8, label="Datos"
    )
    if not df_outliers.empty:
        ax.scatter(
            df_outliers[var2],
            df_outliers[var3],
            df_outliers[var1],
            c="crimson",
            marker="X",
            s=50,
            label="Outliers",
        )

    # Creo la malla (meshgrid) para el plano de predicción
    x1_rango = np.linspace(X[var2].min(), X[var2].max(), 20)
    x2_rango = np.linspace(X[var3].min(), X[var3].max(), 20)
    x1_malla, x2_malla = np.meshgrid(x1_rango, x2_rango)

    # Calculo Z (biomasa) para el plano
    z_plano = (
        modelo.params.iloc[0]
        + modelo.params.iloc[1] * x1_malla
        + modelo.params.iloc[2] * x2_malla
    )

    ax.plot_surface(
        x1_malla, x2_malla, z_plano, color="dodgerblue", alpha=0.3, edgecolor="none"
    )

    label_fito = "Ln Fitovolumen" if transformacion_log else "Fitovolumen"
    label_bio = "Ln Biomasa" if transformacion_log else "Biomasa"
    ax.set_xlabel(f"\n{label_fito}", fontsize=12)
    ax.set_ylabel("\nNDVI Medio", fontsize=12)
    ax.set_zlabel(f"\n{label_bio}", fontsize=12)
    # ax.set_title("Plano de Regresión Lineal Múltiple", fontsize=15, pad=20)

    # Métricas predictivas en la esquina superior izquierda
    texto_metricas = (
        # f"$R^2$ = {modelo.rsquared:.2f} ($R^2_{{adj}}$ = {modelo.rsquared_adj:.2f})\n"
        f"$R^2$ = {modelo.rsquared:.2f}\nRMSE = {rmse:.2f} {unidad}"
    )
    props = {
        "boxstyle": "round,pad=0.5",
        "facecolor": "white",
        "alpha": 0.85,
        "edgecolor": "gray",
    }
    ax.text2D(
        0.02,
        0.95,
        texto_metricas,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    ax.legend(loc="upper right")

    # Guardo las salidas
    if guardar_figura:
        carpeta_salida = Path(dir_salida)
        carpeta_salida.mkdir(parents=True, exist_ok=True)

        sufijo_log = "_log" if transformacion_log else "_linear"
        nombre_base = f"mlr{sufijo_log}_biomasa"

        # Guardo el gráfico 3D
        fig.savefig(
            carpeta_salida / f"{nombre_base}_3D.png", dpi=300, bbox_inches="tight"
        )

        # Guardo los diagnósticos de supuestos estadísticos
        graficar_diagnosticos(
            modelo, ruta_salida=carpeta_salida / f"{nombre_base}_diagnostico.png"
        )

        # Guardo el gráfico de reales vs predichos (2D desde el 3D)
        fig2, ax2 = plt.subplots(figsize=(12, 10))

        # Scatter plot de datos usados
        ax2.scatter(
            y,
            predicciones,
            alpha=0.7,
            s=60,
            color="forestgreen",
            edgecolors="white",
            linewidth=0.5,
            label="Datos utilizados",
        )

        # Aquí se predicen y dibujan outliers si existen
        if not df_outliers.empty:
            X_out = df_outliers[[var2, var3]]
            X_out_const = sm.add_constant(X_out, has_constant="add")
            predicciones_outliers = modelo.predict(X_out_const)
            y_out = df_outliers[var1]
            ax2.scatter(
                y_out,
                predicciones_outliers,
                alpha=0.6,
                s=50,
                color="crimson",
                marker="X",
                label="Valores atípicos",
            )

        # Línea de referencia (1:1)
        min_val = min(y.min(), predicciones.min())
        max_val = max(y.max(), predicciones.max())
        ax2.plot(
            [min_val, max_val],
            [min_val, max_val],
            "b-",
            lw=2,
            label="Recta 1:1 (Predicción Perfecta)",
        )

        ax2.set_xlabel(f"Valor Real ({label_bio})", fontsize=16)
        ax2.set_ylabel(f"Valor Predicho ({label_bio})", fontsize=16)

        # Recuadro de métricas
        texto_metricas_2d = f"$R^2$ = {modelo.rsquared:.2f}\nRMSE = {rmse:.2f} {unidad}\np-valor = {p_valor:.2e}"
        props_2d = {
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "black",
        }
        ax2.text(
            0.05,
            0.95,
            texto_metricas_2d,
            transform=ax2.transAxes,
            fontsize=16,
            verticalalignment="top",
            bbox=props_2d,
        )

        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(
            loc="lower right",
            fontsize=14,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
        )

        fig2.tight_layout()
        fig2.savefig(
            carpeta_salida / f"{nombre_base}_reales_vs_predichos.png",
            dpi=300,
            bbox_inches="tight",
        )

        # Guardo los resultados en un archivo de texto
        ruta_txt = carpeta_salida / f"{nombre_base}_resultados.txt"
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(texto_final + "\n")

        print(
            f"\n[✓] Gráficos, diagnósticos y resultados guardados en: {carpeta_salida}"
        )

    if mostrar_figura:
        plt.show()
    else:
        plt.close("all")

    return modelo


if __name__ == "__main__":
    RUTA_ENTRADAS = Path("entradas")
    RUTA_SALIDAS = Path("salidas")

    RUTA_CSV = RUTA_ENTRADAS / "biomasa" / "biomasa.csv"

    # Escenario E:
    #   peso_seco - fitovol_uav - ndvi_media
    #   Método outliers = none
    #   transformación logarítmica = True
    # Escenario F:
    #   peso_seco - fitovol_uav - ndvi_media
    #   Método outliers = residuos
    #   transformación logarítmica = True
    VAR_RESPUESTA = "peso_seco"
    VAR_PREDICTORA_1 = "fitovol_uav"
    VAR_PREDICTORA_2 = "ndvi_media"

    # Las opciones para los residuos son
    #   - "univariado": se aplica a los valores de cada variable
    #   - "residuos": se aplica a los residuos después de copmarar
    #   - "none": no se filtran los outliers
    METODO_OUTLIERS = "residuos"
    TRANSFORMACION_LOGARITMICA = True  # Indica si se aplica una transformación logarítmica o no

    RUTA_GUARDADO = (
        RUTA_SALIDAS / "graficos_fitovolumen_biomasa_multiple" / METODO_OUTLIERS
    )

    # Ejecuto la regresión si la tabla con los datos existe
    if RUTA_CSV.exists():
        try:
            modelo_mlr = regresion_multiple_biomasa(
                ruta_csv=RUTA_CSV,
                var1=VAR_RESPUESTA,
                var2=VAR_PREDICTORA_1,
                var3=VAR_PREDICTORA_2,
                metodo_outliers=METODO_OUTLIERS,
                transformacion_log=TRANSFORMACION_LOGARITMICA,
                guardar_figura=True,
                mostrar_figura=False,
                dir_salida=RUTA_GUARDADO,
            )
        except (ValueError, KeyError, FileNotFoundError) as e:
            print(f"Error en la ejecución: {e}")
    else:
        print(f"Falta el archivo en: {RUTA_CSV}")
