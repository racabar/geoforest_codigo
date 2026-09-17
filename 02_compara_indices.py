from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns

hora_inicio = datetime.now().astimezone()
print(hora_inicio)


FECHA_INICIO = "2025-05-23"  # "AAAA-MM-DD" o "" 2024-05-16 2025-01-23 2025-05-23
FECHA_FIN = "2025-05-23"
INDICES_PERMITIDOS = ["NDVI", "NDRE", "OSAVI", "MSAVI", "MCARI2"] # Lista de índices: NDVI, TVI2, NDRE, GNDVI, OSAVI, MSAVI, MCARI2, TVI2
# INDICES_PERMITIDOS = ["TVI2"]


def load_indices_sample(ruta_base, indices_dict, n_samples=100000):
    all_data = []

    # Convierto el string a un objeto Path
    directory = Path(ruta_base)

    for name, filename in indices_dict.items():
        file_path = directory / filename

        if not file_path.exists():
            print(f"No está el archivo {file_path}")
            continue

        with rasterio.open(file_path) as src:
            # Leo la banda 1 y la aplano
            band = src.read(1).flatten()
            # Elimino nan, infinitos y nodata
            mask = np.isfinite(band) & (band != src.nodata)
            valid_data = band[mask]

            # Hago un muestreo aleatorio si hay más de n_samples, sino uso todos los datos válidos
            if len(valid_data) > n_samples:
                sample = np.random.choice(valid_data, n_samples, replace=False)
            else:
                sample = valid_data

            all_data.append(pd.DataFrame({"Valor": sample, "Índice": name}))

    return pd.concat(all_data, ignore_index=True)


ruta_indices = Path("salidas/indices")

mapeo_indices = {
    "2023-06-14 NDVI": "230614_ndvi.tif",
    "2023-10-10 NDVI": "231010_ndvi.tif",
    "2024-05-16 NDVI": "240516_ndvi.tif",
    "2025-01-23 NDVI": "250123_ndvi.tif",
    "2025-05-23 NDVI": "250523_ndvi.tif",
    "2023-06-14 TVI2": "230614_tvi2.tif",
    "2023-10-10 TVI2": "231010_tvi2.tif",
    "2024-05-16 TVI2": "240516_tvi2.tif",
    "2025-01-23 TVI2": "250123_tvi2.tif",
    "2025-05-23 TVI2": "250523_tvi2.tif",
    "2023-06-14 NDRE": "230614_ndre.tif",
    "2023-10-10 NDRE": "231010_ndre.tif",
    "2024-05-16 NDRE": "240516_ndre.tif",
    "2025-01-23 NDRE": "250123_ndre.tif",
    "2025-05-23 NDRE": "250523_ndre.tif",
    "2023-06-14 GNDVI": "230614_gndvi.tif",
    "2023-10-10 GNDVI": "231010_gndvi.tif",
    "2024-05-16 GNDVI": "240516_gndvi.tif",
    "2025-01-23 GNDVI": "250123_gndvi.tif",
    "2025-05-23 GNDVI": "250523_gndvi.tif",
    "2023-06-14 OSAVI": "230614_osavi.tif",
    "2023-10-10 OSAVI": "231010_osavi.tif",
    "2024-05-16 OSAVI": "240516_osavi.tif",
    "2025-01-23 OSAVI": "250123_osavi.tif",
    "2025-05-23 OSAVI": "250523_osavi.tif",
    "2023-06-14 MSAVI": "230614_msavi.tif",
    "2023-10-10 MSAVI": "231010_msavi.tif",
    "2024-05-16 MSAVI": "240516_msavi.tif",
    "2025-01-23 MSAVI": "250123_msavi.tif",
    "2025-05-23 MSAVI": "250523_msavi.tif",
    "2023-06-14 MCARI2": "230614_mcari2.tif",
    "2023-10-10 MCARI2": "231010_mcari2.tif",
    "2024-05-16 MCARI2": "240516_mcari2.tif",
    "2025-01-23 MCARI2": "250123_mcari2.tif",
    "2025-05-23 MCARI2": "250523_mcari2.tif",
}

# Filtro el diccionario mapeo_indices por fecha e índice
mapeo_indices_filtrado = {}
for key, value in mapeo_indices.items():
    partes = key.split()
    fecha_str = partes[0]
    indice_str = partes[-1] if len(partes) > 1 else None
    
    # Compruebo el filtro de índice
    if INDICES_PERMITIDOS is not None and indice_str not in INDICES_PERMITIDOS:
        continue

    try:
        fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d").astimezone()
        incluir = True
        if FECHA_INICIO:
            inicio_dt = datetime.strptime(FECHA_INICIO, "%Y-%m-%d").astimezone()
            if fecha_dt < inicio_dt:
                incluir = False
        if FECHA_FIN:
            fin_dt = datetime.strptime(FECHA_FIN, "%Y-%m-%d").astimezone()
            if fecha_dt > fin_dt:
                incluir = False
                
        if incluir:
            mapeo_indices_filtrado[key] = value
    except ValueError:
        # Si no se puede parsear la fecha, lo incluyo por defecto
        mapeo_indices_filtrado[key] = value

mapeo_indices = mapeo_indices_filtrado

if not mapeo_indices:
    raise ValueError("No hay índices que cumplan con los filtros de fecha especificados.")

# Cargo los datos
df = load_indices_sample(ruta_indices, mapeo_indices)

# Extraigo el tipo de índice (última palabra) para usarlo en la coloración
# Ejemplo: "2023-06-14 NDVI" -> "NDVI"
df["Tipo"] = df["Índice"].apply(lambda x: x.split()[-1])

# RIDGE PLOT
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Cuadrícula de subplots (uno por índice)
# Uso hue="Tipo" para que cada índice tenga un color distinto
g = sns.FacetGrid(df, row="Índice", hue="Tipo", aspect=9, height=1.2, palette="viridis")

# Dibujo las densidades (KDE)
g.map(sns.kdeplot, "Valor", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "Valor", clip_on=False, color="w", lw=2, bw_adjust=.5)

# Línea base para cada gráfico
g.map(plt.axhline, y=0, lw=2, clip_on=False)


# Etiquetas
def label(x, color, label):
    ax = plt.gca()
    # Cojo el título del subplot (Índice = 2023-06-14 NDVI, por ejemplo)
    title = ax.get_title()
    # Me quedo solo con la parte derecha del nombre del índice si existe
    if " = " in title:
        row_text = title.split(" = ")[-1]
    else:
        row_text = title
        
    ax.text(0, .2, row_text, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "Valor")

# Esto es para que parezcan histogramas apilados
g.figure.subplots_adjust(hspace=-.25)
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

plt.xlabel("Valor del Índice")
plt.suptitle("Comparativa de índices", fontsize=16)

# Guardado del gráfico
if mapeo_indices:
    # Uso la primera fecha que ha encontrado para el nombre del archivo
    first_key = next(iter(mapeo_indices.keys()))
    date_str = first_key.split(' ')[0]
    # output_filename = f"comparacion_indices_{date_str}.png"
    output_filename = "comparacion_indices_230614.png"
    output_path = ruta_indices / "histogramas" / output_filename
    
    # Se crea carpeta si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Se guarda antes de show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {output_path}")

# Muestro el gráfico
plt.show()

print(f"\nTiempo de procesado total: {datetime.now().astimezone() - hora_inicio}")
