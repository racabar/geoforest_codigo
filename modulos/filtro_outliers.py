"""
Módulo para la detección y filtrado de valores atípicos (outliers) en conjuntos de datos
utilizando el método del Rango Intercuartílico (IQR)
"""

def calcular_mascara_iqr(serie, multiplicador=1.5):
    """Calcula la máscara booleana para filtrar outliers usando el rango intercuartílico"""
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - multiplicador * iqr
    limite_superior = q3 + multiplicador * iqr
    
    return (serie >= limite_inferior) & (serie <= limite_superior)