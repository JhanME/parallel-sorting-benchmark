from mpi4py import MPI
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.sequential.mergesort import mergesort_sequential


def _split_counts(n: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.full(workers, n // workers, dtype=np.int32)
    counts[: n % workers] += 1
    displacements = np.zeros(workers, dtype=np.int32)
    displacements[1:] = np.cumsum(counts[:-1])
    return counts, displacements


def _merge_sorted(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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

    if i < len(left):
        result[k:] = left[i:]
    elif j < len(right):
        result[k:] = right[j:]

    return result


def mergesort_mpi(data=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if data is None:
            data = np.empty(0, dtype=np.float32)
        data = np.asarray(data, dtype=np.float32)
        n = len(data)
    else:
        data = None
        n = None

    n = comm.bcast(n, root=0)
    counts, displacements = _split_counts(n, size)
    local_data = np.empty(counts[rank], dtype=np.float32)

    comm.Scatterv([data, counts, displacements, MPI.FLOAT], local_data, root=0)

    local_data = mergesort_sequential(local_data)

    step = 1
    while step < size:
        if rank % (2 * step) == 0:
            if rank + step < size:
                neighbor_data = comm.recv(source=rank + step, tag=step)
                local_data = _merge_sorted(local_data, neighbor_data)
        else:
            comm.send(local_data, dest=rank - step, tag=step)
            break
        step *= 2

    return local_data if rank == 0 else None


if __name__ == "__main__":
    from src.common.utils import check_sorted, generate_random_array, get_timer

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    data = generate_random_array(size_arg) if rank == 0 else None

    comm.Barrier()
    start = get_timer()
    sorted_data = mergesort_mpi(data)
    comm.Barrier()
    elapsed = get_timer() - start

    if rank == 0:
        print(f"Time: {elapsed:.6f}s")
        if not check_sorted(sorted_data):
            print("Error: Arreglo no ordenado correctamente.")
