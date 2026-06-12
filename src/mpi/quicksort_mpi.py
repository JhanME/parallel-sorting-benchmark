from mpi4py import MPI
import numpy as np
import sys
import os

# Agregar la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.sequential.quicksort import quicksort_sequential

def quicksort_mpi(data=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Distribuir datos
    n = None
    if rank == 0:
        n = len(data)
    n = comm.bcast(n, root=0)

    local_n = n // size
    local_data = np.empty(local_n, dtype=np.float32)

    comm.Scatter(data, local_data, root=0)

    # 2. Ordenar localmente usando el QuickSort secuencial que implementamos
    local_data = quicksort_sequential(local_data)

    # 3. Reunir y mezclar (GATHER + SORT en Rank 0 es la forma más simple de MPI QuickSort)
    # Aunque no es lo más "distribuido", es común para fines académicos básicos.
    # Una versión más avanzada sería Hypercube QuickSort.
    
    all_sorted = None
    if rank == 0:
        all_sorted = np.empty(n, dtype=np.float32)
    
    comm.Gather(local_data, all_sorted, root=0)
    
    if rank == 0:
        # El Gather nos da [pedazo1_ordenado, pedazo2_ordenado, ...]
        # Solo falta mezclar estos pedazos. Para simplicidad, usamos np.sort
        # que es muy eficiente en el Merge final.
        all_sorted.sort()
        return all_sorted
    return None

if __name__ == "__main__":
    from src.common.utils import generate_random_array, check_sorted, get_timer
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    size_arg = 1000
    if len(sys.argv) > 1:
        size_arg = int(sys.argv[1])
    
    data = None
    if rank == 0:
        data = generate_random_array(size_arg)
    
    start = get_timer()
    sorted_data = quicksort_mpi(data)
    end = get_timer()
    
    if rank == 0:
        is_sorted = check_sorted(sorted_data)
        # Solo imprimir el tiempo para que el script de experimentos lo capture
        print(f"Time: {end - start:.6f}s")
        if not is_sorted:
            print("Error: Arreglo no ordenado correctamente.")
