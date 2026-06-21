import numpy as np
from numba import cuda
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common.utils import get_timer, check_sorted, generate_random_array

# ─────────────────────────────────────────────────────────────
# Kernel 1: Ordena bloques pequeños in-place con Insertion Sort
# Se usa en la base de la recursión (subarreglos <= THRESHOLD)
# ─────────────────────────────────────────────────────────────
@cuda.jit
def insertion_sort_kernel(arr, starts, ends, n_segments):
    """
    Cada hilo ordena un segmento pequeño con Insertion Sort.
    'starts' y 'ends' son arreglos con los límites de cada segmento.
    """
    tid = cuda.grid(1)
    if tid >= n_segments:
        return

    lo = starts[tid]
    hi = ends[tid]   # inclusivo

    # Insertion Sort sobre arr[lo..hi]
    for i in range(lo + 1, hi + 1):
        key = arr[i]
        j = i - 1
        while j >= lo and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# ─────────────────────────────────────────────────────────────
# Kernel 2: Particionamiento paralelo (un bloque por segmento)
# Aplica el esquema de Lomuto en GPU usando atomic adds.
# ─────────────────────────────────────────────────────────────
@cuda.jit
def partition_kernel(arr, temp, start, end, pivot, left_count, right_count):
    """
    Clasifica arr[start..end-1] respecto al pivote.
    Elementos < pivot  → temp[0 .. left_count-1]
    Elementos >= pivot → temp[left_count .. ]   (el pivot se insertará en el host)
    Usa atomic.add para asignar posiciones de forma segura entre hilos.
    """
    idx = cuda.grid(1)
    n = end - start

    if idx < n:
        val = arr[start + idx]
        if val < pivot:
            pos = cuda.atomic.add(left_count, 0, 1)
            temp[pos] = val
        elif val > pivot:
            pos = cuda.atomic.add(right_count, 0, 1)
            # Guardamos en la parte derecha del temp (a partir de n)
            temp[n + pos] = val
        # val == pivot: se maneja en el host como el pivote central


def _gpu_partition(d_arr, d_temp, start, end):
    """
    Particiona d_arr[start:end] respecto al pivote (d_arr[start]).
    Devuelve el índice final del pivote dentro del arreglo global.
    """
    n = end - start
    if n <= 0:
        return start

    pivot = d_arr[start:start+1].copy_to_host()[0]

    d_left_count  = cuda.to_device(np.zeros(1, dtype=np.int32))
    d_right_count = cuda.to_device(np.zeros(1, dtype=np.int32))

    threads = 256
    blocks  = math.ceil(n / threads)
    partition_kernel[blocks, threads](d_arr, d_temp, start, end,
                                       pivot, d_left_count, d_right_count)
    cuda.synchronize()

    left_count  = int(d_left_count.copy_to_host()[0])
    right_count = int(d_right_count.copy_to_host()[0])

    # Reconstruir en el arreglo original:
    # [izquierda | pivot | derecha]
    pivot_pos = start + left_count

    # Copiar izquierda desde temp[0:left_count]
    if left_count > 0:
        d_arr[start:pivot_pos] = d_temp[0:left_count]

    # Poner el pivote
    d_arr[pivot_pos:pivot_pos+1] = cuda.to_device(np.array([pivot], dtype=d_arr.dtype))

    # Copiar derecha desde temp[n:n+right_count]
    if right_count > 0:
        d_arr[pivot_pos+1:pivot_pos+1+right_count] = d_temp[n:n+right_count]

    return pivot_pos


# ─────────────────────────────────────────────────────────────
# Función principal: QuickSort híbrido en GPU
# - Usa particionamiento paralelo en GPU para segmentos grandes
# - Cae a Insertion Sort en GPU para segmentos pequeños (≤ THRESHOLD)
# ─────────────────────────────────────────────────────────────
THRESHOLD = 32   # segmentos <= THRESHOLD se ordenan con Insertion Sort en GPU

def quicksort_gpu(arr: np.ndarray) -> np.ndarray:
    """
    QuickSort híbrido acelerado por GPU.
    
    Estrategia:
      - Itera con una pila de segmentos pendientes (evita recursión).
      - Segmentos grandes: particiona en GPU con partition_kernel.
      - Segmentos pequeños (≤ THRESHOLD): acumula y ordena en batch
        con insertion_sort_kernel para maximizar el uso de GPU.
    
    Nota académica: QuickSort no es el algoritmo más natural para GPU
    (la dependencia de datos del pivote limita el paralelismo masivo).
    MergeSort escala mejor. Sin embargo, este híbrido demuestra técnicas
    reales de paralelismo: atomic adds y lanzamiento de kernels en lotes.
    """
    n = len(arr)
    if n <= 1:
        return arr.copy()

    # Mover datos a GPU
    d_arr  = cuda.to_device(arr.copy())
    d_temp = cuda.to_device(np.zeros(n * 2, dtype=arr.dtype))  # buffer auxiliar

    # Pila de rangos [start, end) pendientes
    stack = [(0, n)]

    # Acumulador de segmentos pequeños para ordenar en batch
    small_starts = []
    small_ends   = []

    while stack:
        start, end = stack.pop()
        seg_size = end - start

        if seg_size <= 1:
            continue

        if seg_size <= THRESHOLD:
            # Guardar para ordenar en batch al final
            small_starts.append(start)
            small_ends.append(end - 1)  # inclusivo
            continue

        # Particionar en GPU
        pivot_pos = _gpu_partition(d_arr, d_temp, start, end)

        # Empujar sub-segmentos al stack
        if pivot_pos > start:
            stack.append((start, pivot_pos))
        if pivot_pos + 1 < end:
            stack.append((pivot_pos + 1, end))

    # Ordenar todos los segmentos pequeños en un solo lanzamiento de kernel
    if small_starts:
        n_small = len(small_starts)
        d_starts = cuda.to_device(np.array(small_starts, dtype=np.int32))
        d_ends   = cuda.to_device(np.array(small_ends,   dtype=np.int32))

        threads = 256
        blocks  = math.ceil(n_small / threads)
        insertion_sort_kernel[blocks, threads](d_arr, d_starts, d_ends, n_small)
        cuda.synchronize()

    return d_arr.copy_to_host()


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000

    print(f"QuickSort GPU (Híbrido CUDA) — tamaño: {size:,}")
    data = generate_random_array(size)

    start = get_timer()
    sorted_data = quicksort_gpu(data)
    elapsed = get_timer() - start

    print(f"Resultado : {'CORRECTO ✓' if check_sorted(sorted_data) else 'INCORRECTO ✗'}")
    print(f"Time: {elapsed:.6f}s")
