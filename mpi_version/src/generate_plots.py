import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def run_mpi_test(n, num_procs):
    """Ejecuta el comando mpirun y captura la salida."""
    cmd = [
        "mpirun", "--allow-run-as-root", 
        "-n", str(num_procs), 
        "python3", "src/main.py", str(n)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        print(f"Error ejecutando prueba: {e}")
        return ""

def parse_results(output):
    """Extrae tiempos de la salida del script main.py."""
    results = {}
    # Buscar patrones como "MergeSort MPI: 0.1234s"
    merge_mpi = re.search(r"MergeSort MPI: ([\d.]+)s", output)
    merge_seq = re.search(r"MergeSort Seq: ([\d.]+)s", output)
    quick_mpi = re.search(r"QuickSort MPI: ([\d.]+)s", output)
    quick_seq = re.search(r"QuickSort Seq: ([\d.]+)s", output)

    if merge_mpi and merge_seq:
        results['merge_mpi'] = float(merge_mpi.group(1))
        results['merge_seq'] = float(merge_seq.group(1))
    if quick_mpi and quick_seq:
        results['quick_mpi'] = float(quick_mpi.group(1))
        results['quick_seq'] = float(quick_seq.group(1))
    
    return results

def generate_plots(sizes, data_merge, data_quick):
    """Genera gráficas comparativas."""
    os.makedirs("results", exist_ok=True)
    
    # 1. Gráfica de Tiempos
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, [d['seq'] for d in data_merge], 'o-', label='MergeSort Seq')
    plt.plot(sizes, [d['mpi'] for d in data_merge], 's-', label='MergeSort MPI')
    plt.plot(sizes, [d['seq'] for d in data_quick], 'v-', label='QuickSort Seq')
    plt.plot(sizes, [d['mpi'] for d in data_quick], 'x-', label='QuickSort MPI')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamaño del Arreglo (N)')
    plt.ylabel('Tiempo de Ejecución (s)')
    plt.title('Comparativa de Tiempos: Secuencial vs MPI')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('results/comparativa_tiempos.png')
    
    # 2. Gráfica de Speedup
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, [d['seq']/d['mpi'] for d in data_merge], 'o-', label='Speedup MergeSort')
    plt.plot(sizes, [d['seq']/d['mpi'] for d in data_quick], 's-', label='Speedup QuickSort')
    plt.xlabel('Tamaño del Arreglo (N)')
    plt.ylabel('Speedup (Seq/MPI)')
    plt.title('Speedup con MPI (4 Procesos)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/speedup_grafica.png')
    
    print("\nGráficas generadas en la carpeta 'results/'.")

def main():
    sizes = [10000, 50000, 100000, 500000]
    num_procs = 4
    
    data_merge = []
    data_quick = []
    
    print(f"Iniciando Benchmarking Automatizado...")
    
    for n in sizes:
        print(f"Probando N={n}...", end=" ", flush=True)
        output = run_mpi_test(n, num_procs)
        res = parse_results(output)
        
        if res:
            data_merge.append({'seq': res['merge_seq'], 'mpi': res['merge_mpi']})
            data_quick.append({'seq': res['quick_seq'], 'mpi': res['quick_mpi']})
            print("OK")
        else:
            print("Error al parsear")

    generate_plots(sizes, data_merge, data_quick)

if __name__ == "__main__":
    main()
