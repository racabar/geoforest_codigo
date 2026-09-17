from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from modulos.clasifica_imagen_otsu import clasifica_imagen_otsu


def reclasificar_otsu_a_binario(ruta_entrada_3c, ruta_salida_binaria):
    try:
        with rasterio.open(ruta_entrada_3c) as src:
            meta = src.profile.copy()
            data = src.read(1)
            nodata_original = src.nodata

        # Reclasifico la imagen
        # np.where(condicion, valor_si_verdadero, valor_si_falso)
        # Todo lo que sea estrictamente mayor a 0 (clases 1 y 2) pasará a ser 1.
        data_reclasificada = np.where(data > 0, 1, 0).astype(np.uint8)
        
        # Mantengo el nodata original en la reclasificación
        if nodata_original is not None:
            data_reclasificada[data == nodata_original] = nodata_original

        # Guardo la imagen reclasificada con los metadatos
        with rasterio.open(ruta_salida_binaria, 'w', **meta) as dst:
            dst.write(data_reclasificada, 1)
            dst.set_band_description(1, "Otsu_Binario_Reclasificado")

        print(f"  -> Reclasificación binaria guardada en: {ruta_salida_binaria.name}")

    except Exception as e:
        print(f"Error al reclasificar la imagen {ruta_entrada_3c.name}: {e}")


def procesar_indices_otsu(ruta_entrada_indices, ruta_salida_otsu, nombre_indice, num_clases, banda):
    # Me quedo con los archivos que coincidan con el índice que me interesa
    archivos = list(ruta_entrada_indices.glob(f"*_{nombre_indice}.tif"))

    if not archivos:
        print(f"No hay archivos del índice '{nombre_indice}.tif' en {ruta_entrada_indices}")
        return [], []

    rutas_generadas_3c = []
    datos_umbrales = []

    # Itero sobre cada archivo encontrado
    for archivo in archivos:
        # Cojo la fecha (6 primeros caracteres del nombre del archivo)
        fecha = archivo.name[:6]

        # Creo el nombre y ruta de salida para la imagen de Otsu
        nombre_salida_3c = f"20{fecha}_{nombre_indice}_otsu_{num_clases}c.tif"
        ruta_salida_3c = ruta_salida_otsu / nombre_salida_3c

        print(f"\nProcesando Otsu ({num_clases} clases) para: {archivo.name}...")

        # Llamo a la función del módulo para clasificar con Otsu
        umbrales = clasifica_imagen_otsu(
            ruta_entrada=archivo,
            ruta_salida=ruta_salida_3c,
            banda=banda,
            clases_otsu=num_clases
        )

        # Si el archivo se creó bien, lo guardo en la lista de retorno
        if ruta_salida_3c.exists():
            rutas_generadas_3c.append(ruta_salida_3c)
            # Solo si devuelve umbrales (para clases >= 2 hay al menos 1 umbral)
            if umbrales is not None and len(umbrales) >= 1:
                # Modificado para soportar 2 o 3 clases y extraer la lista completa
                datos_dict = {'indice': archivo.name}
                for i, u in enumerate(umbrales):
                    datos_dict[f'otsu{i+1}'] = u
                datos_umbrales.append(datos_dict)

    print("\nProcesamiento Otsu finalizado.")
    return rutas_generadas_3c, datos_umbrales


if __name__ == "__main__":
    RUTA_BASE = Path(__file__).resolve().parent
    RUTA_INDICES = RUTA_BASE / "salidas" / "indices"
    RUTA_CLASIFICACIONES = RUTA_BASE / "salidas" / "segmentaciones"

    indice = "tvi2"
    clases_otsu = 3
    banda_procesamiento = 1

    ruta_salida_otsu_final = RUTA_CLASIFICACIONES / f"clasificaciones_otsu_{clases_otsu}c" / indice
    # Me aseguro de que existe el directorio y si no existe lo creo
    ruta_salida_otsu_final.mkdir(parents=True, exist_ok=True)

    # Clasifico los índices con Otsu
    archivos_procesados_3c, datos_umbrales = procesar_indices_otsu(
        ruta_entrada_indices=RUTA_INDICES,
        ruta_salida_otsu=ruta_salida_otsu_final,
        nombre_indice=indice,
        num_clases=clases_otsu,
        banda=banda_procesamiento
    )
    
    # Guardo los umbrales en un CSV
    if datos_umbrales:
        df_umbrales = pd.DataFrame(datos_umbrales)
        ruta_csv_umbrales = ruta_salida_otsu_final / f"umbrales_otsu_{indice}.csv"
        df_umbrales.to_csv(ruta_csv_umbrales, index=False)
        print(f"\nTabla de umbrales guardada en: {ruta_csv_umbrales}")
    else:
        print("\nNo se ha podido guardar la tabla de umbrales porque no se han generado datos de Otsu.")

    # Reclasifico los resultados de Otsu en 2 clases
    if archivos_procesados_3c and clases_otsu != 2:
        print("\nIniciando fase de reclasificación binaria...")
        for ruta_3c in archivos_procesados_3c:
            # Nombre del archivo reclasificado
            nombre_binario = ruta_3c.name.replace(f"_{clases_otsu}c.tif", f"_{clases_otsu}c_binario.tif")
            ruta_binaria = ruta_salida_otsu_final / nombre_binario
            ruta_binaria.parent.mkdir(parents=True, exist_ok=True)

            reclasificar_otsu_a_binario(
                ruta_entrada_3c=ruta_3c,
                ruta_salida_binaria=ruta_binaria
            )

        print("\nFlujo de trabajo completo finalizado.")
