# Usar una imagen base de NVIDIA con CUDA y Python
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libopenmpi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configurar alias para python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requerimientos e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Comando por defecto (puede ser sobreescrito al ejecutar)
CMD ["python", "scripts/run_experiments.py"]
