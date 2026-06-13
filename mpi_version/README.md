# Proyecto Final: Ordenamiento Paralelo con MPI

Este módulo forma parte del proyecto de **Computación Paralela y Distribuida**, enfocado en la implementación y comparación de algoritmos de ordenamiento (QuickSort y MergeSort) utilizando la interfaz de paso de mensajes (**MPI**).

## Estructura del Proyecto

```text
mpi_version/
├── src/
│   ├── main.py            # Punto de entrada para pruebas individuales
│   ├── merge_sort.py      # Implementación de MergeSort (Secuencial y MPI)
│   ├── quick_sort.py      # Implementación de QuickSort (Secuencial y MPI)
│   ├── generate_plots.py  # Script para automatizar pruebas y generar gráficas
│   └── utils.py           # Utilidades de medición y generación de datos
├── results/               # Carpeta donde se guardan las gráficas generadas
├── Dockerfile             # Configuración del entorno portable
└── requirements.txt       # Dependencias de Python (mpi4py, numpy, matplotlib)
```

## Requisitos

- **Docker** instalado en el sistema.
- Alternativamente (para ejecución local sin Docker):
  - Python 3.10+
  - Una implementación de MPI (OpenMPI o MPICH)
  - Librerías: `pip install -r requirements.txt`

## Ejecución con Docker (Recomendado)

### 1. Construir la imagen
Desde la raíz de la carpeta `mpi_version`:
```bash
docker build -t mpi-sorting .
```

### 2. Ejecutar un Benchmark individual
Para ordenar un arreglo de 100,000 elementos con 4 procesos:
```bash
docker run --rm mpi-sorting 100000
```

### 3. Generar gráficas para el informe
Para ejecutar todas las pruebas y extraer las gráficas a tu máquina local:
```bash
# En Windows (PowerShell):
docker run --rm -v ${PWD}/results:/app/results mpi-sorting python3 src/generate_plots.py

# En Linux/macOS:
docker run --rm -v $(pwd)/results:/app/results mpi-sorting python3 src/generate_plots.py
```

## Análisis de Métricas
El proyecto calcula automáticamente:
- **Tiempo de Ejecución:** Comparativa entre la versión de un solo núcleo vs varios núcleos.
- **Speedup:** $S = \frac{T_{secuencial}}{T_{paralelo}}$
- **Eficiencia:** $E = \frac{S}{P}$ (donde $P$ es el número de procesos).
