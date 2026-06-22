from mpi4py import MPI
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

# Para compatibilidad con el resto del benchmark de la estructura final
def quicksort_mpi(data=None):
    comm = MPI.COMM_WORLD
    # Convertir a int32 si es necesario, ya que la implementación del usuario usa np.int32
    if comm.Get_rank() == 0:
        if data is None:
            data = np.empty(0, dtype=np.int32)
        else:
            data = np.asarray(data, dtype=np.int32)
    return mpi_quick_sort(data, comm)

if __name__ == "__main__":
    from src.common.utils import check_sorted, generate_random_array, get_timer

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    data = generate_random_array(size_arg) if rank == 0 else None

    comm.Barrier()
    start = get_timer()
    sorted_data = quicksort_mpi(data)
    comm.Barrier()
    elapsed = get_timer() - start

    if rank == 0:
        print(f"Time: {elapsed:.6f}s")
        if not check_sorted(sorted_data):
            print("Error: Arreglo no ordenado correctamente.")
