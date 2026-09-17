from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns


def crea_stack_indices(diccionario_indices):
    stack_temp = []
    meta = None

    # Primero leemos todas las imágenes
    for name, filepath in diccionario_indices.items():
        with rasterio.open(filepath) as src:
            if meta is None:
                meta = src.meta.copy()
            data = src.read(1).astype(np.float32)
            stack_temp.append((name, data))
            print(f"Cargado '{name}' con dimensiones: {data.shape}")

    if not stack_temp:
        raise ValueError("El diccionario de índices está vacío.")

    # Obtengo el tamaño mínimo común por si hay alguna imagen que tiene alguna fila o columna de más
    min_rows = min(data.shape[0] for _, data in stack_temp)
    min_cols = min(data.shape[1] for _, data in stack_temp)

    print(
        f"-> Recortando todas las imágenes a las dimensiones mínimas comunes: {min_rows}x{min_cols}"
    )

    # Recorto todas las imágenes a ese tamaño mínimo
    stack = [data[:min_rows, :min_cols] for _, data in stack_temp]

    # Actualizo los metadatos para reflejar el nuevo tamaño
    meta.update({"height": min_rows, "width": min_cols})

    return np.array(stack), meta


if __name__ == "__main__":
    ruta_base = Path(Path(__file__).resolve().parent)
    ruta_indices = ruta_base / "salidas" / "indices"

    FECHAS = ["240516", "250123", "250523"]

    for fecha in FECHAS:
        indices = {
            "ndvi": ruta_indices / f"{fecha}_ndvi.tif",
            "ndre": ruta_indices / f"{fecha}_ndre.tif",
            "osavi": ruta_indices / f"{fecha}_osavi.tif",
            "msavi": ruta_indices / f"{fecha}_msavi.tif",
            "mcari2": ruta_indices / f"{fecha}_mcari2.tif",
            "tvi2": ruta_indices / f"{fecha}_tvi2.tif",
        }

        valor_nodata = -9999.0

        stack_data, metadata = crea_stack_indices(indices)
        n_features, rows, cols = stack_data.shape

        print(f"Dimensiones de la imagen: {rows}x{cols} píxeles. Índices: {n_features}")

        # Controlo los valores nulos (-9999 y nan)
        valid_mask = (stack_data != valor_nodata) & (~np.isnan(stack_data))
        valid_mask = valid_mask.all(axis=0)

        X_valid = stack_data[:, valid_mask].T

        if X_valid.shape[0] == 0:
            raise ValueError("Todos los píxeles son NoData ¡¡Algo pasa!!")

        # Cojo una muestra aleatoria de 100,000 píxeles para un cálculo estadístico rápido
        # y así no trabajar con la imagen entera, que puede tardar demasiado
        np.random.seed(42)
        n_muestras_corr = min(100000, X_valid.shape[0])
        indices_muestra_corr = np.random.choice(
            X_valid.shape[0], size=n_muestras_corr, replace=False
        )
        X_muestra_corr = X_valid[indices_muestra_corr, :]

        # Creo un dataframe de pandas para enseñar la tabla con la correlación
        nombres_indices = list(indices.keys())
        df_corr = pd.DataFrame(X_muestra_corr, columns=nombres_indices)
        matriz_corr = df_corr.corr().round(3)  # Pearson por defecto

        matriz_corr.to_csv(
            ruta_indices
            / "matrices_correlacion"
            / f"20{fecha}_matriz_correlacion_indices.csv",
            index=False,
        )

        print("Matriz de correlación de Pearson (Valores > 0.90 indican alta redundancia):")
        print(matriz_corr.to_string())

        plt.figure(figsize=(10, 8))
        # Creo y configuro el mapa de calor con los valores, paleta de colores y formato de 2 decimales
        # y lo guardo en una variable para poder configurar la leyenda
        ax = sns.heatmap(
            matriz_corr,
            annot=True,              # Muestra las etiquetas de los valores en cada celda
            annot_kws={"size": 14},  # Tamaño de la fuente de las etiquetas ineriores
            cmap="coolwarm",         # Mapa de color
            vmin=-1,
            vmax=1,
            fmt=".2f",
            linewidths=0.5,
        )
        # configuración de las etiquetas de los ejes
        tamano_etiquetas_ejes = 12
        plt.xticks(fontsize=tamano_etiquetas_ejes)
        plt.yticks(fontsize=tamano_etiquetas_ejes, rotation=90) # rotation=90 para que el eje Y se lea en vertical

        # Configuración de las etiquetas de la leyenda
        cbar = ax.collections[0].colorbar  # Obtengo la barra de color de la leyenda
        # Alineo las etiquetas a la derecha
        for label in cbar.ax.get_yticklabels():
            label.set_horizontalalignment("right")
        # Añado un margen para que el texto no pise la barra de colores y configuro el tamaño de fuente
        cbar.ax.tick_params(
            axis="y",
            pad=50,
            labelsize=14
            ) 

        # Configuración del título
        plt.title(
            f"{fecha[4:6]}-{fecha[2:4]}-20{fecha[0:2]}",
            fontsize=16)
        plt.tight_layout()

        # Guardo el gráfico como imagen en el mismo directorio de salidas
        imagen_salida = (
            ruta_indices
            / "matrices_correlacion"
            / f"20{fecha}_matriz_correlacion_indices.png"
        )
        plt.savefig(imagen_salida, dpi=300)
        print(f"\nGráfica de correlación guardada en: {imagen_salida}")

        # Enseño la ventana con el gráfico en el ide
        plt.show()
        plt.close()
