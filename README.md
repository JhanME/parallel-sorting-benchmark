# Proyecto Final: Comparación de Algoritmos de Ordenamiento (CPU vs GPU)

Este proyecto compara el rendimiento de **QuickSort** y **MergeSort** en tres modalidades: Secuencial, Distribuida (MPI) y Acelerada por GPU (CUDA).

## Requisitos

- Python 3.x
- CUDA Toolkit (para las versiones de GPU)
- Numba
- NumPy, Pandas, Matplotlib
- MPI (OpenMPI o MPICH) y mpi4py

## Estructura

- `src/sequential/`: Versiones secuenciales.
- `src/mpi/`: Versiones distribuidas con MPI.
- `src/cuda/`: Versiones aceleradas con Numba CUDA.
- `scripts/`: Scripts para ejecutar experimentos y generar datos.
- `docker/`: Configuración para despliegue en contenedores.

## Ejecución

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar experimentos:**
   ```bash
   python scripts/run_experiments.py
   ```

3. **Ejecutar con Docker:**
   ```bash
   docker build -t tu-usuario/proyecto-paralela -f docker/Dockerfile .
   docker run --gpus all tu-usuario/proyecto-paralela
   ```

## Publicación en Docker Hub

1. Inicia sesión: `docker login`
2. Sube la imagen: `docker push tu-usuario/proyecto-paralela`

---
**Curso:** Computación Paralela y Distribuida
**Integrantes:** [Nombres de los integrantes]
