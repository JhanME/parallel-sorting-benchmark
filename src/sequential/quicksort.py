import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _partition_lomuto(arr: np.ndarray, low: int, high: int) -> int:
    # Elige arr[high] como pivote y reordena el subarreglo
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quicksort_sequential(arr: np.ndarray) -> np.ndarray:
    # QuickSort iterativo usando pila explícita — O(n log n) promedio
    n = len(arr)
    if n <= 1:
        return arr

    stack = [(0, n - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            pivot_idx = _partition_lomuto(arr, low, high)
            stack.append((low, pivot_idx - 1))   # subarreglo izquierdo
            stack.append((pivot_idx + 1, high))  # subarreglo derecho

    return arr


if __name__ == "__main__":
    from src.common.utils import generate_random_array, check_sorted, get_timer

    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

    print(f"QuickSort Secuencial — tamaño: {size:,}")
    data = generate_random_array(size)

    start = get_timer()
    sorted_data = quicksort_sequential(data.copy())
    elapsed = get_timer() - start

    print(f"Resultado : {'CORRECTO ✓' if check_sorted(sorted_data) else 'INCORRECTO ✗'}")
    print(f"Time: {elapsed:.6f}s")
