from mpi4py import MPI
import numpy as np
import sys
import os

# Agregar la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.sequential.mergesort import mergesort_sequential

def mergesort_mpi(data=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Distribuir datos
    n = None
    if rank == 0:
        n = len(data)
    n = comm.bcast(n, root=0)

    # Calcular pedazos
    local_n = n // size
    local_data = np.empty(local_n, dtype=np.float32)

    comm.Scatter(data, local_data, root=0)

    # 2. Ordenar localmente
    local_data = mergesort_sequential(local_data)

    # 3. Mezcla paralela (Tree Merge)
    step = 1
    while step < size:
        if rank % (2 * step) == 0:
            if rank + step < size:
                # Recibir del vecino
                neighbor_data = np.empty(local_n, dtype=np.float32)
                comm.Recv(neighbor_data, source=rank + step, tag=0)
                
                # Mezclar localmente (podemos usar numpy para velocidad en la mezcla o nuestra lógica)
                # Para ser coherentes con el "espíritu" del curso, mezclamos:
                combined = np.empty(len(local_data) + len(neighbor_data), dtype=np.float32)
                # Usamos numpy sort para la mezcla final por simplicidad de código, 
                # o una función de merge específica.
                # mergesort_sequential ya tiene lógica de merge, pero está acoplada al loop.
                # Vamos a usar np.concatenate + sort o una función merge simple.
                local_data = np.sort(np.concatenate((local_data, neighbor_data)))
                local_n = len(local_data)
        else:
            # Enviar al "padre" en el árbol
            dest = rank - step
            comm.Send(local_data, dest=dest, tag=0)
            break
        step *= 2

    return local_data if rank == 0 else None

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
    sorted_data = mergesort_mpi(data)
    end = get_timer()
    
    if rank == 0:
        is_sorted = check_sorted(sorted_data)
        # Solo imprimir el tiempo para que el script de experimentos lo capture
        print(f"Time: {end - start:.6f}s")
        if not is_sorted:
            print("Error: Arreglo no ordenado correctamente.")
