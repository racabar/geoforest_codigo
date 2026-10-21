"""
Script para calcular el fitovolumen por píxel (cm³) a partir de la serie temporal
de Modelos Digitales de Altura de Copa (CHM) mediante Dask y rioxarray.

Aplica un umbral de altura mínima para excluir el suelo/ruido y exporta los rásteres
de volumen resultantes.
"""

from pathlib import Path

import dask  # para optimizar la carga y procesado de los rasters en memoria
import dask.base
import rioxarray


def procesar_fitovolumen(input_dir_path, output_dir_path, altura_minima_m):
    """
    Calcula el fitovolumen por píxel para una serie temporal de CHMs en cm³.

    Args:
        input_dir_path: Ruta al directorio con los CHM (ej. './mosaicos')
        output_dir_path: Ruta donde se guardarán los rasters de volumen
        altura_minima_m: Umbral mínimo de altura para considerar como dosel (en metros)
    """
    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Buscar todos los archivos CHM en el directorio
    chm_files = sorted(input_dir.glob("*_chm.tif"))

    if not chm_files:
        print("No se encontraron archivos '*_chm.tif' en el directorio especificado.")
        return

    print(f"Se detectaron {len(chm_files)} archivos CHM. Iniciando procesamiento...")

    for file_path in chm_files:
        print(f"Procesando: {file_path.name}...")

        # 1. Leer el CHM usando Dask. Aumentamos chunks a 4096x4096 para aprovechar los 32GB de RAM
        chm = rioxarray.open_rasterio(
            file_path, masked=True, chunks={"x": 4096, "y": 4096}
        )

        # 2. Control de Calidad: Verificar si el CRS es proyectado (métrico)
        if chm.rio.crs and chm.rio.crs.is_geographic:
            print(
                f"[ADVERTENCIA CRÍTICA] {file_path.name} tiene un CRS geográfico. "
                f"El área del píxel estará en grados². Debes reproyectar a UTM primero."
            )

        # 3. Extraer la resolución espacial y calcular el área del píxel en m²
        res_x, res_y = chm.rio.resolution()
        pixel_area_m2 = abs(
            res_x * res_y
        )  # abs() por si res_y es negativa (origen Top-Left)

        # 4. Aplicar máscara de altura mínima para excluir suelo/ruido y valores negativos
        # Asignamos 0 de volumen a lo que esté por debajo del umbral o sea NoData original
        chm_dosel = chm.where(chm >= altura_minima_m, 0)

        # CÁLCULO DEL FITOVOLUMEN (ÁLGEBRA DE MAPAS):
        #   - Fórmula base: (Altura_m * Área_m²) * 1.000.000
        #   - Unidad final: Centímetros cúbicos (cm³) por píxel
        #   - Esto es para:
        #       1) Evitar errores de precisión de coma flotante (Float32) al sumar valores muy pequeños
        #       2) Mejorar la legibilidad de leyendas y simbología en software GIS.
        volumen_pixel_m3 = chm_dosel * pixel_area_m2
        volumen_pixel_cm3 = volumen_pixel_m3 * 1000000.0

        # 6. Preparar metadatos para la exportación
        # Copiamos atributos espaciales del original
        volumen_pixel_cm3.rio.write_crs(chm.rio.crs, inplace=True)
        volumen_pixel_cm3.name = "fitovolumen_cm3"

        # 7. Guardar el nuevo raster de forma optimizada
        output_filename = file_path.stem.replace("_chm", "_Volumen_cm3") + ".tif"
        output_filepath = output_dir / output_filename

        # Exportamos usando ventanas/bloques para mantener bajo el uso de RAM
        volumen_pixel_cm3.rio.to_raster(
            output_filepath,
            compress="lzw",
            tiled=True,
            blockxsize=1024,
            blockysize=1024,
            windowed=True,
            nodata=-9999,
        )

        # Calcular estadísticas globales desencadenando el cómputo distribuido
        volumen_total = float(volumen_pixel_cm3.sum().compute().values)
        print(
            f"  -> Guardado: {output_filename} | Volumen Total Estimado: {volumen_total:,.2f} cm³"
        )


if __name__ == "__main__":
    # FORZADO DE MOTOR DASK:
    #   - scheduler="threads": Obliga a usar el motor multihilo local para evitar errores SSL en Windows.
    dask.config.set(scheduler="threads")

    # PARCHE ANTI-SSL (Bloquear módulo distributed):
    #   - _DISTRIBUTED_AVAILABLE = False: Impide que Dask intente importar 'distributed' internamente.
    #   - Esto evita que 'tornado' y 'ssl' lean el almacén de certificados de Windows corrupto y devuelva un error
    dask.base._DISTRIBUTED_AVAILABLE = False

    # Parámetros generales
    altura_minima = 0.05

    # Directorios raíz
    ruta_entradas = Path("entradas")
    ruta_salidas = Path("salidas")

    # Rutas de entrada / salida
    RUTA_CHM_MOSAICOS = ruta_salidas / "con_sombras" / "06_chm_mosaicos"
    RUTA_RASTERS_VOLUMEN = ruta_salidas / "con_sombras" / "07_fitovolumen"

    # Cálculo del fitovolumen
    procesar_fitovolumen(RUTA_CHM_MOSAICOS, RUTA_RASTERS_VOLUMEN, altura_minima)
    print("\nFitovolumen calculado")
