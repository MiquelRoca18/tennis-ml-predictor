# Dockerfile para Tennis ML Predictor API
# ========================================
# Updated: 2026-01-07 12:31 - Force rebuild
# 
# Build optimizado multi-stage para producción
# 
# Build:
#   docker build -t tennis-ml-api .
# 
# Run:
#   docker run -p 8000:8000 -v $(pwd)/modelos:/app/modelos tennis-ml-api

# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Instalar solo dependencias runtime necesarias
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias de Python desde builder
COPY --from=builder /root/.local /root/.local

# Copiar código de la aplicación
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY scripts/ ./scripts/

# Copiar modelos entrenados
COPY modelos/ ./modelos/

# Crear directorios necesarios
RUN mkdir -p logs datos resultados

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 tennisml && \
    chown -R tennisml:tennisml /app && \
    # Copiar binarios de Python al directorio del usuario
    cp -r /root/.local /home/tennisml/.local && \
    chown -R tennisml:tennisml /home/tennisml/.local

# Cambiar a usuario no-root
USER tennisml

# Asegurar que los scripts de Python están en PATH
ENV PATH=/home/tennisml/.local/bin:$PATH

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

