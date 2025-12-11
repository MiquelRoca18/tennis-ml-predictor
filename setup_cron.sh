#!/bin/bash
# Setup script para configurar cron jobs en Linux/Mac

echo "=================================================="
echo "🤖 CONFIGURACIÓN DE CRON JOBS - TENNIS ML"
echo "=================================================="
echo ""

# Obtener directorio del proyecto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

echo "📁 Directorio del proyecto: $PROJECT_DIR"
echo "🐍 Python del venv: $VENV_PYTHON"
echo ""

# Verificar que existe el venv
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: No se encontró el entorno virtual en $VENV_PYTHON"
    echo "   Por favor, crea el entorno virtual primero con: python -m venv venv"
    exit 1
fi

echo "✅ Entorno virtual encontrado"
echo ""

# Crear archivo temporal con los cron jobs
TEMP_CRON=$(mktemp)

# Obtener crontab actual (si existe)
crontab -l > "$TEMP_CRON" 2>/dev/null || true

# Añadir comentario de identificación
echo "" >> "$TEMP_CRON"
echo "# Tennis ML Predictor - Automated Tasks" >> "$TEMP_CRON"
echo "# Generated on $(date)" >> "$TEMP_CRON"

# 1. Actualización de datos (3:00 AM diaria)
echo "0 3 * * * cd $PROJECT_DIR && $VENV_PYTHON src/automation/data_updater.py >> logs/cron_data_update.log 2>&1" >> "$TEMP_CRON"

# 2. Reentrenamiento del modelo (4:00 AM, verificación diaria pero ejecuta según estrategia)
echo "0 4 * * * cd $PROJECT_DIR && $VENV_PYTHON src/automation/model_retrainer.py >> logs/cron_retrain.log 2>&1" >> "$TEMP_CRON"

# 3. Predicciones diarias (9:00 AM)
echo "0 9 * * * cd $PROJECT_DIR && $VENV_PYTHON src/automation/daily_predictor.py >> logs/cron_daily.log 2>&1" >> "$TEMP_CRON"

# 4. Monitoreo del sistema (12:00 PM)
echo "0 12 * * * cd $PROJECT_DIR && $VENV_PYTHON src/automation/monitoring.py >> logs/cron_monitoring.log 2>&1" >> "$TEMP_CRON"

echo ""
echo "📋 Cron jobs a configurar:"
echo "  - 03:00 AM: Actualización de datos"
echo "  - 04:00 AM: Verificación de reentrenamiento"
echo "  - 09:00 AM: Predicciones diarias"
echo "  - 12:00 PM: Monitoreo del sistema"
echo ""

# Preguntar confirmación
read -p "¿Deseas instalar estos cron jobs? (s/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Ss]$ ]]; then
    # Instalar crontab
    crontab "$TEMP_CRON"
    
    if [ $? -eq 0 ]; then
        echo "✅ Cron jobs instalados correctamente"
        echo ""
        echo "📋 Para ver los cron jobs instalados:"
        echo "   crontab -l"
        echo ""
        echo "📋 Para editar los cron jobs:"
        echo "   crontab -e"
        echo ""
        echo "📋 Para eliminar todos los cron jobs:"
        echo "   crontab -r"
        echo ""
        echo "📝 Los logs se guardarán en: $PROJECT_DIR/logs/"
    else
        echo "❌ Error instalando cron jobs"
        exit 1
    fi
else
    echo "❌ Instalación cancelada"
    echo ""
    echo "💡 Si quieres instalar manualmente, añade estas líneas a tu crontab (crontab -e):"
    echo ""
    cat "$TEMP_CRON" | grep "^0"
fi

# Limpiar archivo temporal
rm "$TEMP_CRON"

echo ""
echo "=================================================="
echo "✅ CONFIGURACIÓN COMPLETADA"
echo "=================================================="
