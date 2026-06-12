import numpy as np

def mergesort_sequential(arr):
    """
    Implementación secuencial de MergeSort (Bottom-up / Iterativo).
    """
    n = len(arr)
    width = 1
    while width < n:
        for i in range(0, n, 2 * width):
            left = i
            mid = min(i + width, n)
            right = min(i + 2 * width, n)
            
            # Merge logic
            if mid < right:
                l_part = arr[left:mid].copy()
                r_part = arr[mid:right].copy()
                
                l_idx, r_idx = 0, 0
                k = left
                
                while l_idx < len(l_part) and r_idx < len(r_part):
                    if l_part[l_idx] <= r_part[r_idx]:
                        arr[k] = l_part[l_idx]
                        l_idx += 1
                    else:
                        arr[k] = r_part[r_idx]
                        r_idx += 1
                    k += 1
                
                while l_idx < len(l_part):
                    arr[k] = l_part[l_idx]
                    l_idx += 1
                    k += 1
                
                while r_idx < len(r_part):
                    arr[k] = r_part[r_idx]
                    r_idx += 1
                    k += 1
        width *= 2
    return arr

if __name__ == "__main__":
    from src.common.utils import generate_random_array, check_sorted
    data = generate_random_array(1000)
    sorted_data = mergesort_sequential(data.copy())
    print(f"MergeSort Sequential: {'Correct' if check_sorted(sorted_data) else 'Incorrect'}")
