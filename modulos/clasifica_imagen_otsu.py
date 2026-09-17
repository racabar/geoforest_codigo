"""
Módulo para la binarización y clasificación automática de imágenes (ej. separar
vegetación de suelo desnudo) utilizando el método de umbralización de Otsu.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap

# pyrefly: ignore [missing-import]
from skimage.filters import threshold_multiotsu


def clasifica_imagen_otsu(ruta_entrada, ruta_salida, banda, clases_otsu=2, visualizar=False):
    ruta_entrada = Path(ruta_entrada)
    ruta_salida = Path(ruta_salida)

    if not ruta_entrada.exists():
        print(f"No existe el archivo de entrada: {ruta_entrada}")
        return None

    # Creo el directorio de salida si no existe
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    try:
        with rasterio.open(ruta_entrada) as src:
            # Leo la banda donde está la información y me quedo con los metadatos
            data = src.read(banda)
            meta = src.profile.copy()
            nodata_val = src.nodata

            # Creo una máscara de datos válidos para calcular Otsu ignorando valores nulos
            if nodata_val is not None:
                mascara_validos = (data != nodata_val) & (~np.isnan(data))
            else:
                mascara_validos = ~np.isnan(data)

            # Me quedo con los píxeles válidos para calcular el umbral
            datos_validos = data[mascara_validos]

            if datos_validos.size == 0:
                print(f"La imagen {ruta_entrada.name} no tiene datos válidos para calcular Otsu.")
                return None

            # Calculo los umbrales de Otsu
            # Devuelvo n_clases - 1 umbrales con threshold_multiotsu
            umbrales = threshold_multiotsu(datos_validos, classes=clases_otsu)

            print(f"Umbrales de Otsu calculados para {ruta_entrada.name}: {umbrales}")

            # Clasifico usando np.digitize
            # Devuelvo los índices de los bins a los que pertenece cada valor con np.digitize
            # Si solo son dos umbrales, asigno 0 por debajo del umbral y 1 por encima con np.digitize
            imagen_clasificada = np.digitize(data, bins=umbrales).astype(np.uint8)

            # Controlo los valores nulos si hay, asignándoles el valor -9999 (255 si casteamos a uint8)
            # Ya que la imagen_clasificada es uint8, asigno 255 como nodata, es más estándar para este tipo.
            VALOR_NODATA = 255
            
            mask_nodata = np.isnan(data)
            if nodata_val is not None:
                mask_nodata = mask_nodata | (data == nodata_val)

            if mask_nodata.any():
                imagen_clasificada[mask_nodata] = VALOR_NODATA

            # Actualizo los metadatos de la imagen de salida
            meta.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw',
                nodata=VALOR_NODATA  # Defino el nodata explícitamente a 255
            )

            # Formateo los umbrales para la descripción
            umbrales_str = "_".join([f"{u:.3f}" for u in umbrales])

            # Guardo la imagen reclasificada asignándole los metadatos de la original
            with rasterio.open(ruta_salida, 'w', **meta) as salida:
                salida.write(imagen_clasificada, 1)
                salida.set_band_description(1, f"Clasificacion_Otsu_Umbrales_{umbrales_str}")

            print(f"Clasificación Otsu guardada en: {ruta_salida}")

            # Visualizo (Opcional y con downsampling para evitar OutOfMemory)
            if visualizar:
                plt.figure(figsize=(10, 8))
                
                # Hago un downsampling (diezmado) saltando de 10 en 10 píxeles 
                # para evitar saturar la memoria RAM en imágenes de dron muy pesadas.
                img_reducida = imagen_clasificada[::10, ::10]

                # Enmascaro NoData para visualización
                datos_plot = np.ma.masked_equal(img_reducida, VALOR_NODATA)

                # Si son 2 clases (quemado/no quemado) muestro solo dos colores (negro y blanco)
                if clases_otsu == 2:
                    # Asumo que la clase alta (1) es quemado
                    datos_masked = np.ma.masked_where(img_reducida != 1, img_reducida)
                    cmap = ListedColormap(['black'])
                    plt.imshow(datos_masked, cmap=cmap, interpolation='nearest')
                    plt.title(f"Área Quemada (Clase 1) - Otsu: {ruta_salida.name}")
                else:
                    # Si son más clases muestro usando una rampa de color (Viridis)
                    plt.imshow(datos_plot, cmap='viridis', interpolation='nearest')
                    plt.colorbar(ticks=range(clases_otsu), label='Clase')
                    plt.title(f"Clasificación Multi-Otsu ({clases_otsu} clases): {ruta_salida.name}")

                plt.axis('off')
                plt.show()
            
            return umbrales

    except Exception as e:
        print(f"Error en clasificación Otsu para {ruta_entrada.name}: {e}")
        return None


if __name__ == "__main__":
    # Cojo la ruta absoluta, que con la relativa estoy teniendo problemas
    ruta_base = Path(__file__).resolve().parent.parent
    ruta_indices = ruta_base / "entradas" / "indices"

    indice = "tvi2"

    # Me quedo solo con los archivos del índice que me interesa
    archivos = list(ruta_indices.glob(f"*{indice}.tif"))

    for archivo in archivos:
        # Cojo la fecha del índice, que son los 6 primeros caracteres del archivo
        fecha = archivo.name[:6]

        ruta_salida = ruta_base / f"salidas/segmentaciones/20260312_otsu_ndvi_tvi2/20{fecha}_{indice}_otsu_3c.tif"

        print(f"Procesando: {archivo.name}...")

        clasifica_imagen_otsu(
            archivo,
            ruta_salida,
            1,
            3,
            visualizar=False # Por defecto lo apago para procesamiento en lote
        )