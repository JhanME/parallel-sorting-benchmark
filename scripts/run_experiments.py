import os
import sys
import pandas as pd
import numpy as np
import subprocess
import time
import shutil

# Agregar la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.utils import get_timer, generate_random_array, check_sorted
from src.sequential.mergesort import mergesort_sequential
from src.sequential.quicksort import quicksort_sequential

# Importar las funciones de GPU (si están disponibles localmente)
try:
    from src.cuda.mergesort_cuda import mergesort_gpu
    from src.cuda.quicksort_cuda import quicksort_gpu
    HAS_CUDA = True
except ImportError:
    print("Advertencia: No se pudo cargar módulos CUDA. Saltando pruebas de GPU.")
    HAS_CUDA = False

MPI_COMMANDS = [
    "mpiexec",
    r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe",
    "mpirun",
]

def run_mpi(script_path, size):
    # Ejecutar MPI y capturar el tiempo
    # Asumimos que el script imprime el tiempo en la última línea con "Time: X.XXXXs"
    # Intentar con mpiexec y mpirun
    found_mpi = False
    for mpi_cmd in MPI_COMMANDS:
        mpi_path = shutil.which(mpi_cmd) or (mpi_cmd if os.path.exists(mpi_cmd) else None)
        if mpi_path is None:
            print(f"Advertencia: no se encontro {mpi_cmd} en PATH.")
            continue

        found_mpi = True
        cmd = [mpi_path, "-n", "4", sys.executable, os.path.abspath(script_path), str(size)]
        print("Ejecutando MPI:", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "Time:" in line:
                        return float(line.split(":")[1].strip().replace("s", ""))
                print(f"Advertencia: {mpi_cmd} no devolvio una linea con Time:.")
                print(result.stdout.strip())
            else:
                print(f"Error ejecutando {mpi_cmd}:")
                print(result.stderr.strip() or result.stdout.strip())
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error ejecutando {mpi_cmd}: {e}")
    if not found_mpi:
        print("Error: no se encontro ningun ejecutable MPI. Verifica que mpiexec este en PATH.")
    return 0

def main():
    # Tamaños reducidos para pruebas rápidas si es necesario
    sizes = [1000, 10000, 100000, 1000000]
    results = []
    
    for size in sizes:
        print(f"\n--- Probando con tamaño: {size} ---")
        data = generate_random_array(size)
        
        # 1. Secuencial MergeSort
        start = get_timer()
        mergesort_sequential(data.copy())
        t_seq_merge = get_timer() - start
        
        # 2. Secuencial QuickSort
        start = get_timer()
        quicksort_sequential(data.copy())
        t_seq_quick = get_timer() - start
        
        print(f"Secuencial MergeSort: {t_seq_merge:.4f}s")
        print(f"Secuencial QuickSort: {t_seq_quick:.4f}s")

        # 3. MPI
        t_mpi_merge = run_mpi("src/mpi/mergesort_mpi.py", size)
        t_mpi_quick = run_mpi("src/mpi/quicksort_mpi.py", size)
        print(f"MPI MergeSort: {t_mpi_merge:.4f}s")
        print(f"MPI QuickSort: {t_mpi_quick:.4f}s")
        
        # 4. GPU (CUDA)
        t_cuda_merge = 0
        t_cuda_quick = 0
        if HAS_CUDA:
            try:
                # MergeSort GPU
                start = get_timer()
                mergesort_gpu(data)
                t_cuda_merge = get_timer() - start
                print(f"CUDA MergeSort: {t_cuda_merge:.4f}s")
                
                # QuickSort GPU
                start = get_timer()
                quicksort_gpu(data)
                t_cuda_quick = get_timer() - start
                print(f"CUDA QuickSort: {t_cuda_quick:.4f}s")
            except Exception as e:
                print(f"Error en GPU: {e}")

        results.append({
            "Size": size,
            "Seq_MergeSort": t_seq_merge,
            "Seq_QuickSort": t_seq_quick,
            "MPI_MergeSort": t_mpi_merge,
            "MPI_QuickSort": t_mpi_quick,
            "CUDA_MergeSort": t_cuda_merge,
            "CUDA_QuickSort": t_cuda_quick,
            "Speedup_Merge_CUDA": t_seq_merge / t_cuda_merge if t_cuda_merge > 0 else 0,
            "Speedup_Merge_MPI": t_seq_merge / t_mpi_merge if t_mpi_merge > 0 else 0
        })

    # Guardar resultados
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    csv_path = "results/experiments_results.csv"
    try:
        df.to_csv(csv_path, index=False)
        print(f"\nResultados guardados en {csv_path}")
    except PermissionError:
        timestamp = int(time.time())
        new_path = f"results/experiments_results_{timestamp}.csv"
        df.to_csv(new_path, index=False)
        print(f"\nAdvertencia: No se pudo escribir en {csv_path} (archivo abierto?).")
        print(f"Resultados guardados en {new_path}")
    except Exception as e:
        print(f"\nError al guardar resultados: {e}")

if __name__ == "__main__":
    main()
