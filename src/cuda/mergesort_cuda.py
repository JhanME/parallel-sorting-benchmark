import numpy as np
from numba import cuda
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.utils import get_timer, check_sorted, generate_random_array


# ─────────────────────────────────────────────────────────────
# Kernel: Fusiona pares de sub-arreglos ordenados en paralelo
# Cada hilo se encarga de una fusión independiente.
# ─────────────────────────────────────────────────────────────
@cuda.jit
def merge_kernel(arr, temp_arr, width, n):
    """
    Kernel de fusión para MergeSort bottom-up en GPU.
    
    Cada hilo procesa un par de sub-arreglos de tamaño 'width':
      - arr[start .. mid-1]  (izquierda, ya ordenada)
      - arr[mid   .. end-1]  (derecha,   ya ordenada)
    El resultado fusionado se escribe en temp_arr[start .. end-1].
    
    Al final de cada pasada, los punteros se intercambian (pointer swap)
    para evitar copias innecesarias entre GPU y CPU.
    """
    idx = cuda.grid(1)

    # Cada hilo procesa el segmento que le corresponde
    start = idx * 2 * width

    if start >= n:
        return  # hilo fuera de rango

    mid = min(start + width, n)
    end = min(start + 2 * width, n)

    # Copiar la mitad izquierda si no hay mitad derecha
    if mid >= end:
        for x in range(start, end):
            temp_arr[x] = arr[x]
        return

    # Fusión clásica de dos sub-arreglos ordenados
    i = start   # índice en la mitad izquierda
    j = mid     # índice en la mitad derecha
    k = start   # índice de escritura en temp_arr

    while i < mid and j < end:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            i += 1
        else:
            temp_arr[k] = arr[j]
            j += 1
        k += 1

    # Vaciar elementos restantes del lado izquierdo
    while i < mid:
        temp_arr[k] = arr[i]
        i += 1
        k += 1

    # Vaciar elementos restantes del lado derecho
    while j < end:
        temp_arr[k] = arr[j]
        j += 1
        k += 1


def mergesort_gpu(arr: np.ndarray) -> np.ndarray:
    """
    MergeSort bottom-up acelerado por GPU con Numba CUDA.

    Complejidad: O(n log n)  —  todas las fusiones de cada nivel son paralelas.

    Fases:
      width=1:  fusiona pares individuales → bloques de 2
      width=2:  fusiona pares de 2        → bloques de 4
      width=4:  fusiona pares de 4        → bloques de 8
      …
      width=n/2: fusión final             → arreglo completo ordenado

    Optimizaciones implementadas:
      - Pointer swap entre d_arr1 y d_arr2: evita copias Device→Device.
      - cuda.synchronize() explícito tras cada kernel para garantizar
        consistencia antes del siguiente nivel de fusión.
      - threads_per_block=256 es un buen equilibrio entre ocupancia y
        recursos de registros en hardware CUDA moderno.
    """
    n = len(arr)
    if n <= 1:
        return arr.copy()

    threads_per_block = 256

    # Transferir datos a la GPU (ping-pong buffers)
    d_arr1 = cuda.to_device(arr.copy())
    d_arr2 = cuda.to_device(np.empty_like(arr))

    width = 1
    while width < n:
        # Número de fusiones independientes en este nivel
        num_merges = math.ceil(n / (2 * width))
        blocks_per_grid = math.ceil(num_merges / threads_per_block)

        # Lanzar kernel: d_arr1 es entrada, d_arr2 es salida
        merge_kernel[blocks_per_grid, threads_per_block](d_arr1, d_arr2, width, n)
        cuda.synchronize()  # Esperar a que todos los hilos terminen este nivel

        # Pointer swap: no copia datos, solo intercambia referencias en Python
        d_arr1, d_arr2 = d_arr2, d_arr1

        width *= 2

    # Tras el último swap, el resultado final está en d_arr1
    return d_arr1.copy_to_host()


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000

    print(f"MergeSort GPU (Numba CUDA) — tamaño: {size:,}")
    data = generate_random_array(size)

    start = get_timer()
    sorted_data = mergesort_gpu(data)
    elapsed = get_timer() - start

    print(f"Resultado : {'CORRECTO ✓' if check_sorted(sorted_data) else 'INCORRECTO ✗'}")
    print(f"Time: {elapsed:.6f}s")
