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
FROM python:3.12-slim AS builder

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

# Instalar solo dependencias runtime necesarias (+ curl para descargar TML-Database en build)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias de Python desde builder
COPY --from=builder /root/.local /root/.local

# CACHE BUST: 2026-01-21-17:55 - Force rebuild from here
# Copiar código de la aplicación
COPY src/ ./src/
COPY scripts/ ./scripts/

# Sistema usa solo baseline ELO + mercado (no .pkl ni selected_features)

# Script de arranque
COPY start.sh ./start.sh
RUN chmod +x start.sh

# Crear directorios necesarios (resultados para backtesting; logs)
RUN mkdir -p logs datos/raw datos/processed resultados

# Descargar CSVs de TML-Database para ELO en producción (2025 + 2026; cada build tiene datos recientes)
RUN curl -sL "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2025.csv" -o datos/raw/2025.csv && \
    curl -sL "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2026.csv" -o datos/raw/2026.csv

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

# Exponer puerto (Railway usa $PORT dinámico)
EXPOSE 8000

# Comando por defecto
CMD ["./start.sh"]

