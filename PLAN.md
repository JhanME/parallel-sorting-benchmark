# Plan de Proyecto: Comparación de Algoritmos de Ordenamiento (CPU vs GPU)

Este documento detalla la estrategia de implementación, estructura de archivos y metodología para el proyecto final de Computación Paralela y Distribuida.

## 1. Estructura del Proyecto

```text
paralela-proyecto/
├── src/
│   ├── common/             # Utilidades comunes (generación de datos, validación)
│   ├── sequential/         # Versiones secuenciales (Compañeros)
│   │   ├── quicksort.py
│   │   └── mergesort.py
│   ├── mpi/                # Versiones distribuidas con MPI (Compañeros)
│   │   ├── quicksort_mpi.py
│   │   └── mergesort_mpi.py
│   └── cuda/               # Versiones en GPU con Numba CUDA (Tu parte)
│       ├── quicksort_cuda.py
│       └── mergesort_cuda.py
├── scripts/
│   ├── generate_data.py    # Generador de arreglos aleatorios
│   └── run_experiments.py  # Automatización de pruebas y toma de tiempos
├── results/                # Tablas y gráficos generados
├── docker/
│   └── Dockerfile
├── requirements.txt
└── README.md               # Guía de ejecución
```

## 2. Estrategia de Implementación (GPU - Numba CUDA)

### QuickSort en GPU
El QuickSort es un reto en GPU debido a su naturaleza recursiva.
- **Enfoque:** Implementar un particionamiento paralelo o utilizar una versión iterativa con una pila (stack) gestionada en memoria global/compartida.
- **Optimización:** Para arreglos pequeños resultantes de la partición, se puede cambiar a un ordenamiento por inserción dentro del kernel para evitar la sobrecarga de lanzamiento de hilos.

### MergeSort en GPU
Es más natural para el paralelismo masivo.
- **Enfoque:** Implementación iterativa (bottom-up).
- **Fases:** 
    1. Ordenar bloques pequeños en memoria compartida.
    2. Realizar mezclas (merges) sucesivas aumentando el tamaño del bloque hasta completar el arreglo.

## 3. Metodología Experimental

1. **Tamaños de entrada:** $10^3, 10^4, 10^5, 10^6, 10^7$ elementos.
2. **Métricas:**
   - **Tiempo de ejecución ($T$):** Promedio de 5 ejecuciones por caso.
   - **Speedup ($S$):** $S = T_{seq} / T_{par}$
   - **Eficiencia ($E$):** $E = S / p$ (donde $p$ es el número de núcleos/hilos).
3. **Validación:** Comprobar que el arreglo resultante está efectivamente ordenado (`all(a[i] <= a[i+1])`).

## 4. Dockerización

- **Base Image:** `nvidia/cuda:12.x-base-ubuntu22.04` (o similar con soporte para Python).
- **Dependencias:** 
  - Python 3.x
  - Numba (para CUDA)
  - mpi4py (para las versiones MPI de tus compañeros)
  - NumPy, Matplotlib, Pandas (para análisis y gráficos)

## 5. Próximos Pasos Inmediatos

1. Definir los utilitarios comunes (`src/common/utils.py`) para asegurar que todos usen el mismo formato de datos y cronómetro.
2. Implementar un esqueleto de `run_experiments.py`.
3. Comenzar con `src/cuda/mergesort_cuda.py` ya que es más estable en GPU.
