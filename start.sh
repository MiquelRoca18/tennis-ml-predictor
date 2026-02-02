#!/bin/bash
# Script de arranque para Railway - captura errores de inicio
set -e
cd /app
echo "Starting Tennis ML API on port ${PORT:-8000}..."
exec uvicorn src.api.api_v2:app --host 0.0.0.0 --port ${PORT:-8000}
