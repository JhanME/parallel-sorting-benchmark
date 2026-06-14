import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _merge(arr: np.ndarray, left: int, mid: int, right: int) -> None:
    # Fusiona arr[left:mid] y arr[mid:right] en su lugar
    l_part = arr[left:mid].copy()
    r_part = arr[mid:right].copy()

    l_idx, r_idx, k = 0, 0, left
    while l_idx < len(l_part) and r_idx < len(r_part):
        if l_part[l_idx] <= r_part[r_idx]:
            arr[k] = l_part[l_idx]; l_idx += 1
        else:
            arr[k] = r_part[r_idx]; r_idx += 1
        k += 1

    # Vaciar lo que sobró en cada lado
    while l_idx < len(l_part):
        arr[k] = l_part[l_idx]; l_idx += 1; k += 1
    while r_idx < len(r_part):
        arr[k] = r_part[r_idx]; r_idx += 1; k += 1


def mergesort_sequential(arr: np.ndarray) -> np.ndarray:
    # MergeSort iterativo bottom-up — O(n log n) en todos los casos
    n = len(arr)
    if n <= 1:
        return arr

    # Fusiona subarreglos de tamaño creciente: 1 → 2 → 4 → 8 …
    width = 1
    while width < n:
        for i in range(0, n, 2 * width):
            left  = i
            mid   = min(i + width,     n)
            right = min(i + 2 * width, n)
            if mid < right:
                _merge(arr, left, mid, right)
        width *= 2

    return arr


if __name__ == "__main__":
    from src.common.utils import generate_random_array, check_sorted, get_timer

    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

    print(f"MergeSort Secuencial — tamaño: {size:,}")
    data = generate_random_array(size)

    start = get_timer()
    sorted_data = mergesort_sequential(data.copy())
    elapsed = get_timer() - start

    print(f"Resultado : {'CORRECTO ✓' if check_sorted(sorted_data) else 'INCORRECTO ✗'}")
    print(f"Time: {elapsed:.6f}s")
