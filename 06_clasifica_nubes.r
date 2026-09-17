library(lidR)
library(terra)
library(tools)

dir_lidar <- "entradas/lidar"
dir_clasificaciones <- "salidas/segmentaciones/clasificaciones_otsu_2c/ndvi"
dir_salida <- "entradas/lidar/clasificados"

if (!dir.exists(dir_salida)) {
  dir.create(dir_salida, recursive = TRUE)
}

# FUNCIÓN DE PROCESAMIENTO POR BLOQUE (CHUNK)
# Esta función se ejecutará para cada cuadrícula independiente
procesar_chunk <- function(cluster, raster_ndvi) {
  # Leo solo la porción de puntos actual
  las <- readLAS(cluster)

  # Compruebo si el bloque está vacío
  if (is.empty(las)) {
    return(NULL)
  }

  # Limpio la clasificación que ya tenga la nube de puntos
  las@data$Classification <- 1L

  # Extraigo los datos del ráster
  las <- merge_spatial(las, raster_ndvi, attribute = "clase_otsu")

  # Hago la reclasificación según la clase de Otsu y la clasificación ASPRS
  las@data$Classification[las@data$clase_otsu == 0] <- 2L
  las@data$Classification[las@data$clase_otsu == 1] <- 4L
  las@data$Classification[las@data$clase_otsu == 255] <- 1L
  las@data$clase_otsu <- NULL

  # Se devuelve la nube procesada
  return(las)
}


archivos_laz <- list.files(dir_lidar, pattern = "\\.laz$", full.names = TRUE)

for (ruta_laz in archivos_laz) {
  nombre_archivo <- tools::file_path_sans_ext(basename(ruta_laz))
  fecha_vuelo <- regmatches(nombre_archivo, regexpr("^[0-9]+", nombre_archivo))
  nombre_tif <- paste0(fecha_vuelo, "_ndvi_otsu_2c.tif")
  ruta_tif <- file.path(dir_clasificaciones, nombre_tif)

  if (!file.exists(ruta_tif)) {
    next
  }

  cat("Procesando en Mosaicos:", nombre_archivo, "\n")

  raster_ndvi <- rast(ruta_tif)

  # Creo un catálogo en lugar de leer el archivo completo
  ctg <- readLAScatalog(ruta_laz)

  # Configuro el motor para optimizar el uso de la RAM
  # Divido el trabajo en celdas de 200x200 metros
  opt_chunk_size(ctg) <- 200
  # No se necesita un buffer para los solapes al ser la clasificación un proceso 1 a 1 en vertical
  opt_chunk_buffer(ctg) <- 0
  # Esto es para ver una barra de progreso en consola (funciona regular...)
  opt_progress(ctg) <- TRUE

  # Configuro el guardado automático en SUBCARPETAS por vuelo
  dir_salida_vuelo <- file.path(dir_salida, nombre_archivo)
  if (!dir.exists(dir_salida_vuelo)) {
    dir.create(dir_salida_vuelo, recursive = TRUE)
  }

  # Activo la compresión laz
  opt_laz_compression(ctg) <- TRUE

  # Defino la plantilla (SIN extensión, lidR añade .laz automáticamente)
  opt_output_files(ctg) <- file.path(
    dir_salida_vuelo,
    paste0(nombre_archivo, "_class_tile_{XLEFT}_{YBOTTOM}")
  )

  # Ejecuto la función de procesado por bloque
  catalog_apply(ctg, procesar_chunk, raster_ndvi = raster_ndvi)

  # Esto fuerza la limpieza de basura en memoria para la siguiente iteración
  rm(ctg, raster_ndvi)
  gc()
}

cat("Procesado finalizado.\n")
