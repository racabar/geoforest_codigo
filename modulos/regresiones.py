"""
Módulo centralizado para la preparación de datos (nulos, transformaciones lógicas)
y el ajuste de modelos de regresión lineal (simple y múltiple) mediante mínimos cuadrados ordinarios (OLS).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

from modulos.filtro_outliers import calcular_mascara_iqr
from modulos.valida_supuestos_estadisticos import validar_supuestos_ols


def ajustar_modelo_regresion(
    df,
    col_biomasa,
    cols_predictoras,
    metodo_outliers="none",
    transformacion_log=False,
):
    """
    Prepara los datos, aplica transformación logarítmica, filtra outliers y
    ajusta un modelo de regresión lineal (simple o múltiple)
    """
    
    # Valido que están las columnas que deben en el csv
    # Si no están lanzo un error
    columnas_req = [col_biomasa] + cols_predictoras
    for col in columnas_req:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: '{col}'")

    # Limpio los datos nulos y creo un nuevo dataframe para trabajar con él
    df_limpio = df[["id_quadrat"] + columnas_req].dropna().copy()

    # Si elijo una transformación logarítmica
    if transformacion_log:
        # Aquí filtro valores <= 0 para que no den errores matemáticos con el logaritmo
        # NOTA: Solo aplico logaritmo a biomasa y a la primera variable predictora (fitovolumen)
        # El NDVI (u otros índices en el futuro) normalmente no se transforma al poder tener valores negativos
        # y ser un índice acotado, por lo qeu no tiene sentido aplicar logaritmo a un índice que ya está normalizado entre -1 y 1
        col_fitovolumen = cols_predictoras[0]
        df_limpio = df_limpio[
            (df_limpio[col_biomasa] > 0) & (df_limpio[col_fitovolumen] > 0)
        ].copy()
        
        df_limpio[col_biomasa] = np.log(df_limpio[col_biomasa])
        df_limpio[col_fitovolumen] = np.log(df_limpio[col_fitovolumen])

    n_inicial = len(df_limpio)
    
    if n_inicial < len(cols_predictoras) + 2:
        raise ValueError("No hay suficientes muestras para realizar el análisis")

    # Aplico el filtro de outliers que seleccione
    print(f"Filtro de outliers: {metodo_outliers}")

    if metodo_outliers == "univariado":
        # Filtro IQR independiente
        mascara_final = calcular_mascara_iqr(df_limpio[col_biomasa])
        for col in cols_predictoras:
            mascara_final = mascara_final & calcular_mascara_iqr(df_limpio[col])

    elif metodo_outliers == "residuos":
        # Filtro IQR basado en los residuos del modelo inicial
        X_prev = sm.add_constant(df_limpio[cols_predictoras])
        y_prev = df_limpio[col_biomasa]
        
        # Ajusto el modelo previo para sacar residuos
        modelo_previo = sm.OLS(y_prev, X_prev).fit()
        residuos = pd.Series(modelo_previo.resid, index=df_limpio.index)

        # Calculo la máscara sobre los residuos
        mascara_final = calcular_mascara_iqr(residuos)

    elif metodo_outliers == "none" or metodo_outliers is None:
        # Sin filtro de outliers
        mascara_final = pd.Series([True] * len(df_limpio), index=df_limpio.index)

    else:
        raise ValueError("Las opciones para metodo_outliers son 'none', 'univariado' o 'residuos'")

    # SEparo los datos válidos de los atípicos
    df_inliers = df_limpio[mascara_final]
    df_outliers = df_limpio[~mascara_final]

    n_muestras = len(df_inliers)
    n_outliers = len(df_outliers)

    if n_outliers > 0:
        print(f"Se han filtrado {n_outliers} outliers usando el método '{metodo_outliers}'")

    print(f"Muestras finales para utilizar en la regresión: {n_muestras}")

    # Preparo datos para regresión (sólo con los inliers)
    X = df_inliers[cols_predictoras]
    y = df_inliers[col_biomasa]

    # Regresión lineal (usando statsmodels)
    X_const = sm.add_constant(X)
    modelo = sm.OLS(y, X_const).fit()

    # predicciones
    predicciones = modelo.predict(X_const)

    # métricas
    rmse = np.sqrt(mean_squared_error(y, predicciones))
    mae = np.mean(np.abs(y - predicciones))
    
    # Análisis de Multicolinealidad (VIF) si hay más de 1 predictor
    vif_data = None
    if len(cols_predictoras) > 1:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

    # Validación de supuestos estadísticos
    diagnosticos = validar_supuestos_ols(modelo)

    # Guardo las métricas en un diccionario
    resultados = {
        "modelo": modelo,
        "rmse": rmse,
        "mae": mae,
        "n_muestras_validas": n_muestras,
        "n_outliers": n_outliers,
        "datos_inliers": df_inliers,
        "datos_outliers": df_outliers,
        "predicciones": predicciones,
        "diagnosticos": diagnosticos,
        "vif_data": vif_data
    }

    return resultados
