# Usar una imagen base de Python ligera
FROM python:3.10-slim

# Instalar dependencias del sistema para MPI
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requerimientos e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y crear carpeta de resultados
COPY src/ ./src/
RUN mkdir -p results

# Comando por defecto para ejecutar el benchmark
# Se usa mpirun con --allow-run-as-root para Docker y -n 4 procesos por defecto
ENTRYPOINT ["mpirun", "--allow-run-as-root", "-n", "4", "python", "src/main.py"]
CMD ["100000"]
