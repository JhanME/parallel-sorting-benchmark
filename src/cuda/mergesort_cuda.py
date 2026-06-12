import numpy as np
from numba import cuda
import math
from src.common.utils import get_timer, check_sorted, generate_random_array

@cuda.jit
def merge_kernel(arr, temp_arr, width, n):
    """
    Kernel para fusionar sub-arreglos ordenados.
    """
    idx = cuda.grid(1)
    
    # Cada hilo se encarga de una fusión de dos sub-arreglos
    # El paso actual es 2 * width
    start = idx * 2 * width
    
    if start < n:
        mid = min(start + width, n)
        end = min(start + 2 * width, n)
        
        i = start
        j = mid
        k = start
        
        while i < mid and j < end:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                i += 1
            else:
                temp_arr[k] = arr[j]
                j += 1
            k += 1
            
        while i < mid:
            temp_arr[k] = arr[i]
            i += 1
            k += 1
            
        while j < end:
            temp_arr[k] = arr[j]
            j += 1
            k += 1

def mergesort_gpu(arr):
    n = len(arr)
    threads_per_block = 256
    
    # Mover datos a la GPU
    d_arr1 = cuda.to_device(arr)
    d_arr2 = cuda.to_device(np.zeros_like(arr))
    
    width = 1
    while width < n:
        num_merges = math.ceil(n / (2 * width))
        blocks_per_grid = math.ceil(num_merges / threads_per_block)
        
        # Lanzar el kernel: d_arr1 es entrada, d_arr2 es salida
        merge_kernel[blocks_per_grid, threads_per_block](d_arr1, d_arr2, width, n)
        
        # ¡IMPORTANTE! Intercambiar los arreglos (Pointer Swap)
        # Esto es mucho más rápido que d_arr1.copy_to_device(d_arr2)
        d_arr1, d_arr2 = d_arr2, d_arr1
        
        width *= 2
        
    # El resultado final está en d_arr1 (debido al último intercambio)
    return d_arr1.copy_to_host()

if __name__ == "__main__":
    # Prueba rápida
    size = 10000
    data = generate_random_array(size)
    
    print(f"Ordenando {size} elementos con MergeSort CUDA...")
    start = get_timer()
    sorted_data = mergesort_gpu(data)
    end = get_timer()
    
    if check_sorted(sorted_data):
        print(f"¡Éxito! Tiempo: {end - start:.4f}s")
    else:
        print("Error: El arreglo no está ordenado.")
