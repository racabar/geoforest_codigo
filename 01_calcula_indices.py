from datetime import datetime
from pathlib import Path

from modulos.calcula_indices import calcula_indices

hora_inicio = datetime.now().astimezone()

ruta_base = Path(Path(__file__).resolve().parent)

ruta_bandas = ruta_base / "entradas" / "bandas"
ruta_indices = ruta_base / "entradas" / "indices"

indices_a_calcular = ["TriVI"]


def procesa_indices(ruta_bandas, ruta_indices, indices):
    if not ruta_bandas.exists():
        print(f"No se encuentra el directorio de bandas en {ruta_bandas}")
        return

    ruta_indices.mkdir(parents=True, exist_ok=True)

    for ruta_fecha in ruta_bandas.iterdir():
        if not ruta_fecha.is_dir():
            continue

        fecha = ruta_fecha.name

        for ruta_bloque in ruta_fecha.iterdir():
            if not ruta_bloque.is_dir():
                continue

            bloque = ruta_bloque.name
            print(f"\n--- Procesando vuelo de fecha: {fecha}, bloque: {bloque} ---")

            # Formateo el prefijo del nombre de archivo (230614_b1_4)
            bloque_formateado = bloque.lower().replace("-", "_")

            try:
                # Llamo al módulo de cálculo de índices
                calcula_indices(
                    ruta_entrada=str(ruta_bloque),
                    ruta_salida=str(ruta_indices),
                    lista_indices=indices,
                    crs_salida="EPSG:32630"
                )

                # Renombro los rasters calculados "al vuelo" en el mismo directorio
                for indice in indices:
                    archivo_original = ruta_indices / f"{indice}.tif"

                    if archivo_original.exists():
                        # Nombre final del archivo del índice
                        nombre_final = f"{fecha}_{bloque_formateado}_{indice.lower()}.tif"
                        archivo_renombrado = ruta_indices / nombre_final

                        # Controlo si el archivo ya existe y lo elimino antes de renombrar
                        if archivo_renombrado.exists():
                            archivo_renombrado.unlink()

                        archivo_original.rename(archivo_renombrado)
                        print(f"Renombrado a: {nombre_final}")
                    else:
                        print(f"No existe el archivo generado para {indice}")

            except Exception as e:
                print(f"Fallo al procesar {fecha}/{bloque}: {e}")


if __name__ == "__main__":
    procesa_indices(ruta_bandas, ruta_indices, indices_a_calcular)
    print("\nTiempo de procesado total: ", datetime.now().astimezone() - hora_inicio)
