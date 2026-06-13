import sys
import os
from mpi4py import MPI
import numpy as np

# Ajustar path para importar módulos locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import generate_random_array, is_sorted, Timer
from merge_sort import mpi_merge_sort, sequential_merge_sort
from quick_sort import mpi_quick_sort, sequential_quick_sort

def run_benchmark(n):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data = None
    if rank == 0:
        print(f"\n--- Iniciando Prueba: N={n}, Procesos={size} ---")
        data = generate_random_array(n)

    # --- MERGE SORT ---
    comm.Barrier()
    with Timer() as t_mpi_merge:
        result_merge = mpi_merge_sort(data, comm)
    
    if rank == 0:
        with Timer() as t_seq_merge:
            sequential_merge_sort(data.copy())
        
        print(f"MergeSort MPI: {t_mpi_merge.duration:.4f}s")
        print(f"MergeSort Seq: {t_seq_merge.duration:.4f}s")
        print(f"Speedup Merge: {t_seq_merge.duration / t_mpi_merge.duration:.2f}x")
        print(f"Correcto: {is_sorted(result_merge)}")

    # --- QUICK SORT ---
    comm.Barrier()
    with Timer() as t_mpi_quick:
        result_quick = mpi_quick_sort(data, comm)

    if rank == 0:
        with Timer() as t_seq_quick:
            sequential_quick_sort(data.copy().tolist())
        
        print(f"QuickSort MPI: {t_mpi_quick.duration:.4f}s")
        print(f"QuickSort Seq: {t_seq_quick.duration:.4f}s")
        print(f"Speedup Quick: {t_seq_quick.duration / t_mpi_quick.duration:.2f}x")
        print(f"Correcto: {is_sorted(result_quick)}")

if __name__ == "__main__":
    N = 100000 # Tamaño por defecto
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    run_benchmark(N)
