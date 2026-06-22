import numpy as np
import time

def check_sorted(arr):
    """Verifica si un arreglo está ordenado de forma ascendente."""
    return np.all(arr[:-1] <= arr[1:])

def get_timer():
    """Retorna el tiempo actual en segundos."""
    return time.perf_counter()

def generate_random_array(size, seed=42):
    """Genera un arreglo de números aleatorios (float32 por defecto para compatibilidad con GPU)."""
    np.random.seed(seed)
    return np.random.rand(size).astype(np.float32)
