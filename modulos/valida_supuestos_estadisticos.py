"""
Módulo para evaluar y graficar los supuestos estadísticos de un modelo OLS 
(Normalidad de Shapiro-Wilk, Homocedasticidad de Breusch-Pagan e Independencia de Durbin-Watson).
"""
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
from scipy import stats
from statsmodels.stats.stattools import durbin_watson


def validar_supuestos_ols(modelo):
    """
    Evalúa los supuestos principales de una regresión lineal OLS:
      - Normalidad de los residuos (Shapiro-Wilk)
      - Homocedasticidad (Breusch-Pagan)
      - Independencia/Ausencia de Autocorrelación (Durbin-Watson)
    """
    residuos = modelo.resid
    exog = modelo.model.exog

    # Normalidad: Test de Shapiro-Wilk
    # H0: La distribución de los datos es normal
    stat_sw, p_sw = stats.shapiro(residuos)
    interp_sw = "Se cumple (Aceptada)" if p_sw > 0.05 else "NO se cumple (Alerta de asimetría/colas pesadas)"

    # Homocedasticidad: Test de Breusch-Pagan
    # H0: La varianza de los residuos es constante
    _, p_bp, _, _ = smd.het_breuschpagan(residuos, exog)
    interp_bp = "Se cumple (Varianza constante)" if p_bp > 0.05 else "NO se cumple (Alerta de heterocedasticidad)"

    # Independencia: Estadístico de Durbin-Watson
    # Rango [0, 4]. Cerca de 2 indica ausencia de autocorrelación de primer orden.
    dw_stat = durbin_watson(residuos)
    if 1.5 <= dw_stat <= 2.5:
        interp_dw = "Se cumple (Sin autocorrelación fuerte)"
    else:
        interp_dw = "Alerta (Posible autocorrelación en los residuos)"

    return {
        "normalidad": {
            "estadistico": stat_sw,
            "p_valor": p_sw,
            "interpretacion": interp_sw
        },
        "homocedasticidad": {
            "p_valor": p_bp,
            "interpretacion": interp_bp
        },
        "independencia": {
            "estadistico": dw_stat,
            "interpretacion": interp_dw
        }
    }


def graficar_diagnosticos(modelo, ruta_salida=None, mostrar_figura=False):
    """
    Genera un panel gráfico con dos plots clásicos para diagnosticar un modelo OLS.
    
    Args:
        modelo: Un objeto de resultados de regresión OLS ajustado por statsmodels.
        ruta_salida (Path o str): Ruta donde guardar la figura. Si es None, no se guarda.
        mostrar_figura (bool): Si es True, muestra la figura interactiva.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    residuos = modelo.resid
    predichos = modelo.fittedvalues

    # Plot 1: Q-Q Plot (Evalúa Normalidad)
    sm.qqplot(residuos, line='45', fit=True, ax=ax[0])
    ax[0].set_title("Q-Q plot de los residuos\n(Evaluación de normalidad)", fontsize=16, pad=15)
    ax[0].grid(True, linestyle='--', alpha=0.6)
    ax[0].set_xlabel("Cuantiles teóricos", fontsize=14)
    ax[0].set_ylabel("Cuantiles de los residuos", fontsize=14)
    
    # Plot 2: Residuos vs Valores Predichos (Evalúa Homocedasticidad y Linealidad)
    # Busco que los puntos formen una nube aleatoria sin patrón de embudo
    ax[1].scatter(predichos, residuos, alpha=0.7, color='steelblue', edgecolors='white', s=50)
    ax[1].axhline(0, color='crimson', linestyle='--', linewidth=2)
    ax[1].set_xlabel("Valores predichos (biomasa)", fontsize=16)
    ax[1].set_ylabel("Residuos", fontsize=14)
    ax[1].set_title("Residuos vs. Predichos\n(Evaluación de homocedasticidad)", fontsize=16, pad=15)
    ax[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if ruta_salida:
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    
    if mostrar_figura:
        plt.show()
    else:
        plt.close(fig)
