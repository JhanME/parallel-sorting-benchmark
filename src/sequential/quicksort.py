import numpy as np

def quicksort_sequential(arr):
    """
    Implementación secuencial de QuickSort (Iterativo con stack).
    """
    n = len(arr)
    if n <= 1:
        return arr
    
    # Stack para simular recursión
    stack = [(0, n - 1)]
    
    while stack:
        low, high = stack.pop()
        if low < high:
            # Particionamiento (Hoare o Lomuto)
            # Usaremos Lomuto para simplicidad
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            p = i + 1
            
            # Push sub-arreglos al stack
            stack.append((low, p - 1))
            stack.append((p + 1, high))
    
    return arr

if __name__ == "__main__":
    from src.common.utils import generate_random_array, check_sorted
    data = generate_random_array(1000)
    sorted_data = quicksort_sequential(data.copy())
    print(f"QuickSort Sequential: {'Correct' if check_sorted(sorted_data) else 'Incorrect'}")
