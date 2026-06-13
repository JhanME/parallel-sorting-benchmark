import numpy as np
from mpi4py import MPI
from .utils import Timer

def merge(left, right):
    """Fusiona dos arreglos ordenados."""
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

def sequential_merge_sort(arr):
    """Implementación secuencial de Merge Sort."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = sequential_merge_sort(arr[:mid])
    right = sequential_merge_sort(arr[mid:])
    return merge(left, right)

def mpi_merge_sort(arr, comm):
    """Implementación paralela de Merge Sort con MPI."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 1. Distribuir el trabajo (Scatter)
    n = len(arr) if rank == 0 else None
    n = comm.bcast(n, root=0)
    
    local_size = n // size
    local_arr = np.empty(local_size, dtype=np.int32)
    
    comm.Scatter(arr, local_arr, root=0)
    
    # 2. Ordenamiento local
    local_arr = sequential_merge_sort(local_arr)
    
    # 3. Fusión iterativa (Estructura de árbol)
    step = 1
    while step < size:
        if rank % (2 * step) == 0:
            if rank + step < size:
                neighbor_arr = comm.recv(source=rank + step)
                local_arr = merge(local_arr, neighbor_arr)
        else:
            neighbor = rank - step
            comm.send(local_arr, dest=neighbor)
            break
        step *= 2
        
    return local_arr if rank == 0 else None
