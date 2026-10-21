# Script para normalizar alturas de nubes LiDAR clasificadas y generar Modelos
# Digitales de Altura de Copa (CHM) a 5 cm de resolución.
#
# Procesa las nubes por bloques en paralelo con lidR, genera los CHMs individuales
# y los consolida en mosaicos ráster georreferenciados.

library(lidR)
library(terra)
library(future)

# Configuración de la paralelización de la carga eficientemente para usar 4 núcleos de CPU de los 12 disponibles y 32 GB de ram
# Si uso 4 núcleos de la CPU se consume toda la RAM y recibo un mensaje de bad allocation
plan(multisession, workers = 4)

dir_clasificados <- "salidas/sin_sombras/05_lidar_clasificados"
dir_salida_base <- "salidas/sin_sombras"

# 5 cm de resolución del chm
resolucion_raster <- 0.05

# Carpeta de salida para los bloques
dir_chm <- file.path(dir_salida_base, "06.1_chm_bloques")
if (!dir.exists(dir_chm)) {
  dir.create(dir_chm, recursive = TRUE)
}

# Carpeta de salida para los mosaicos finales
dir_mosaicos <- file.path(dir_salida_base, "06.2_chm_mosaicos")
if (!dir.exists(dir_mosaicos)) {
  dir.create(dir_mosaicos, recursive = TRUE)
}

# Función para normalizar en 3D y generar el CHM en memoria
procesar_chm_chunk <- function(chunk) {
  las <- readLAS(chunk)

  # Medida de seguridad si el bloque está vacío
  if (is.empty(las)) {
    return(NULL)
  }

  # Compruebo si hay suficientes puntos de suelo para normalizar el bloque (mínimo 3 para un plano)
  if (sum(las@data$Classification == 2L) < 3) {
    return(NULL)
  }

  # CApturo de errores seguro fallos de memoria
  chm_chunk <- tryCatch(
    {
      # En lugar de normalizar usando 'tin()' directamente sobre los millones de puntos vectoriales
      # (lo cual obliga a R a triangular billones de combinaciones en C++ para cada punto de vegetación),
      # primero rasterizo un MDT temporal rápido a resolución de 20 cm.
      # El relieve del suelo es suave y no varía a nivel de milímetros, por lo que 20 cm es una cifra aceptable
      mdt_temp <- rasterize_terrain(las, res = 0.20, algorithm = tin())

      # Normalización 3D por lookup de píxel (Complejidad O(N) infinitamente más rápida)
      # R solo lee la celda de terreno correspondiente debajo de cada punto de vegetación.
      las_norm <- normalize_height(las, mdt_temp)

      # Limpieza de ruido por variaciones pequeñas de suelo en el plano normalizado (Z = 0)
      las_norm@data$Z[las_norm@data$Classification == 2L] <- 0
      las_norm@data$Z[las_norm@data$Z < 0] <- 0

      # rasterizo la copa del pastizal sobre la nube normalizada a la resolución final (5 cm)
      rasterize_canopy(
        las_norm,
        res = resolucion_raster,
        algorithm = p2r(subcircle = 0.04)
      )
    },
    error = function(e) {
      message("   Ha habido un error al procesar la imagen: ", e$message)
      return(NULL)
    }
  )

  # Esto fuerza la limpieza de basura en memoria para la siguiente iteración
  gc()
  return(chm_chunk)
}

# EJECUCIÓN DEL FLUJO DE TRABAJO
carpetas_vuelos <- list.dirs(
  dir_clasificados,
  full.names = TRUE,
  recursive = FALSE
)

# Filtramos para excluir la carpeta 'con_sombras' de ejecuciones anteriores
carpetas_vuelos <- carpetas_vuelos[basename(carpetas_vuelos) != "con_sombras"]

# esto es solo para comprobar que va todo bien mientras se ejecuta
cat("Se han detectado", length(carpetas_vuelos), "vuelos clasificados.\n\n")

for (ruta_vuelo in carpetas_vuelos) {
  nombre_vuelo <- basename(ruta_vuelo)

  if (length(list.files(ruta_vuelo, pattern = "\\.laz$")) == 0) {
    next
  }

  tiempo_inicio <- Sys.time()

  cat("PROCESANDO VUELO (Método Normalización Híbrida):", nombre_vuelo, "\n")

  # Cargo el catálogo de teselas
  ctg <- readLAScatalog(ruta_vuelo)

  # Optimizaciones de entrada / salida para optimizar el uso de ram y acelerar la ejecución
  # xyzc: lee en memoria coordenadas (X,Y,Z) y código de clasificación.
  opt_select(ctg) <- "xyzc"
  # filter: ignora los puntos con clase 1 (ruido/no clasificados) o bordes sin interés
  # Solo cargo suelo (2) y cegetación (4). Así se reduce el uso de la ram
  opt_filter(ctg) <- "-keep_class 2 4"

  # Configuración para el portátil (32 GB ram, 6 núcleos CPU)
  # Esto reduce el tamaño de bloque a 100m, para que sean lo suficientemente pequeños, así la triangulación
  # local del MDT temporal es bastante más rápida y consume bastante poco (unos 500 MB por worker)
  opt_chunk_size(ctg) <- 100
  opt_chunk_buffer(ctg) <- 15 # Buffer de 15 metros para evitar costuras
  opt_progress(ctg) <- TRUE
  opt_output_files(ctg) <- ""

  # GENERO EL CHM (normalizado en 3D)
  cat("   -> Normalizando nube y generando CHM de", nombre_vuelo, "\n")
  lista_chm <- catalog_apply(ctg, procesar_chm_chunk)

  # Filtro usando inherits() de forma segura para conservar solo los SpatRaster válidos
  lista_chm <- Filter(function(x) inherits(x, "SpatRaster"), lista_chm)

  # VAlidoi que la colección no esté vacía antes de hacer la fusión
  if (length(lista_chm) == 0) {
    warning("   TODAS LAS PORCIONES FALLARON PARA EL VUELO: ", nombre_vuelo, "")
    next
  }

  # Guardo la imagen en el disco
  cat("   -> Fusionando y exportando CHM de alta resolución (5 cm)...\n")
  ruta_salida_chm <- file.path(dir_chm, paste0(nombre_vuelo, "_CHM.tif"))

  # Convierto la lista de rásters en una colección nativa de terra
  coleccion_chm <- sprc(lista_chm)

  # Fusiono guardando directamente en el disco duro para evitar saturar la ram
  terra::merge(
    coleccion_chm,
    filename = ruta_salida_chm,
    overwrite = TRUE,
    wopt = list(gdal = c("COMPRESS=LZW"))
  )

  # Controles míos para saber cuánto dura la ejecución y si hay errores de memoria
  tiempo_fin <- Sys.time()
  duracion_total <- difftime(tiempo_fin, tiempo_inicio, units = "secs")

  # Formateo el tiempo en minutos y segundos legibles
  minutos <- floor(as.numeric(duracion_total) / 60)
  segundos <- round(as.numeric(duracion_total) %% 60, 1)

  cat(sprintf(
    "   -> Tiempo de procesado para %s: %d min %s seg\n",
    nombre_vuelo,
    minutos,
    segundos
  ))

  # libero la ram antes del próximo vuelo
  rm(ctg, lista_chm, coleccion_chm)
  gc(reset = TRUE)
}

cat("Modelos CHM de bloques creados\n")

# CREO MOSAICOS POR FECHA

cat("\nIniciando fase de mosaicado por fechas...\n")

# Buscamos todos los CHM individuales generados
archivos_chm <- list.files(dir_chm, pattern = "\\.tif$", full.names = TRUE)

if (length(archivos_chm) > 0) {
  # Extraemos las fechas únicas (primeros 8 caracteres del nombre del archivo)
  fechas <- unique(substr(basename(archivos_chm), 1, 8))
  
  for (fecha in fechas) {
    cat("   -> Creando mosaico para la fecha:", fecha, "\n")
    
    # Seleccionamos los archivos que corresponden a esta fecha
    archivos_fecha <- archivos_chm[grepl(paste0("^", fecha), basename(archivos_chm))]
    ruta_salida_mosaico <- file.path(dir_mosaicos, paste0(fecha, "_chm.tif"))
    
    # Fusionamos (incluso si solo hay 1 archivo, merge lo guardará con el nuevo nombre y compresión)
    lista_rasters <- lapply(archivos_fecha, rast)
    coleccion_mosaico <- sprc(lista_rasters)
    
    terra::merge(
      coleccion_mosaico,
      filename = ruta_salida_mosaico,
      overwrite = TRUE,
      wopt = list(gdal = c("COMPRESS=LZW"))
    )
    
    # Limpiamos RAM
    rm(lista_rasters, coleccion_mosaico)
    gc(reset = TRUE)
  }
  cat("\nMosaicos CHM finalizados y guardados en:", dir_mosaicos, "\n")
} else {
  cat("\nNo se encontraron archivos CHM en la carpeta de bloques para fusionar.\n")
}
