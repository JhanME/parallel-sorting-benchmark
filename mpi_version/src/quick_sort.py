import numpy as np

def sequential_quick_sort(arr):
    """Implementación secuencial de Quick Sort."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return sequential_quick_sort(left) + middle + sequential_quick_sort(right)

def mpi_quick_sort(arr, comm):
    """Implementación paralela de Quick Sort con MPI."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Distribuir el trabajo inicial (Scatter)
    n = len(arr) if rank == 0 else None
    n = comm.bcast(n, root=0)
    
    local_size = n // size
    local_arr = np.empty(local_size, dtype=np.int32)
    comm.Scatter(arr, local_arr, root=0)

    # 2. Ordenar localmente
    # Nota: Convertimos a lista para el quicksort secuencial simple y luego a numpy
    local_sorted = np.array(sequential_quick_sort(local_arr.tolist()), dtype=np.int32)

    # 3. Mezclar resultados (Gather y Merge final para simplicidad en comparación)
    # En una implementación más avanzada usaríamos pivotes paralelos, 
    # pero para el modelo de comparación de tareas, el merge de resultados ordenados es efectivo.
    all_sorted = comm.gather(local_sorted, root=0)

    if rank == 0:
        # Fusión final de los arreglos ordenados
        final_arr = all_sorted[0]
        for i in range(1, len(all_sorted)):
            final_arr = merge_arrays(final_arr, all_sorted[i])
        return final_arr
    return None

def merge_arrays(left, right):
    """Auxiliar para fusionar arreglos ordenados en el proceso raíz."""
    result = np.empty(len(left) + len(right), dtype=left.dtype)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result[k] = left[i]
            i += 1
        else:
            result[k] = right[j]
            j += 1
        k += 1
    result[k:] = left[i:] if i < len(left) else right[j:]
    return result
