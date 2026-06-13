import numpy as np
import time

def generate_random_array(n):
    """Genera un arreglo de n enteros aleatorios."""
    return np.random.randint(0, 1000000, size=n, dtype=np.int32)

def is_sorted(arr):
    """Verifica si un arreglo está ordenado."""
    return np.all(arr[:-1] <= arr[1:])

class Timer:
    """Clase simple para medir tiempos de ejecución."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start
