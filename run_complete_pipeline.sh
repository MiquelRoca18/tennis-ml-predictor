#!/bin/bash
# Pipeline Completo - Desde Cero hasta Modelo Final
# Este script ejecuta todo el proceso de principio a fin

set -e  # Detener si hay algún error

echo "========================================================================"
echo "🚀 PIPELINE COMPLETO - PREDICCIÓN DE TENIS"
echo "========================================================================"
echo ""

# Directorio del proyecto (usar directorio actual)
# cd se ejecutará desde donde se llame el script

echo "========================================================================"
echo "🧹 PASO 0: LIMPIEZA (Opcional - comentar si no quieres limpiar)"
echo "========================================================================"
echo ""
echo "Limpiando datos antiguos..."
rm -rf datos/raw/*.csv
rm -rf datos/processed/*.csv
echo "Limpiando modelos antiguos..."
rm -rf modelos/*.pkl
rm -rf modelos/*.keras
echo "Limpiando resultados antiguos..."
rm -rf resultados/*.csv
rm -rf resultados/*.png
echo "✅ Limpieza completada"
echo ""

echo "========================================================================"
echo "📥 PASO 1: DESCARGA DE DATOS (TML Database 2020-2025)"
echo "========================================================================"
echo ""
python src/data/tml_data_downloader.py
echo ""

echo "========================================================================"
echo "🧹 PASO 2: LIMPIEZA Y PROCESAMIENTO DE DATOS"
echo "========================================================================"
echo ""
python src/data/data_processor.py
echo ""

echo "========================================================================"
echo "🔧 PASO 3: FEATURE ENGINEERING COMPLETO (Fase 3)"
echo "========================================================================"
echo ""
echo "Generando 114 features avanzadas:"
echo "  - ELO Rating System"
echo "  - Estadísticas Servicio/Resto"
echo "  - Métricas de Fatiga"
echo "  - Forma Reciente"
echo "  - Head-to-Head Mejorado"
echo "  - Especialización por Superficie"
echo ""
python run_feature_engineering_fase3.py
echo ""

echo "========================================================================"
echo "🎯 PASO 4: OPTIMIZACIÓN Y ENTRENAMIENTO DE MODELOS"
echo "========================================================================"
echo ""
echo "Este paso incluye:"
echo "  1. Feature Selection (30 mejores de 114)"
echo "  2. Entrenamiento de modelos base"
echo "  3. Calibración isotónica"
echo "  4. Hyperparameter tuning"
echo ""
python run_fase3_optimization.py
echo ""

echo "========================================================================"
echo "🏆 PASO 5: WEIGHTED ENSEMBLE (Mejor Modelo)"
echo "========================================================================"
echo ""
echo "Combinando modelos con pesos optimizados..."
python src/models/weighted_ensemble.py
echo ""

# Paso 6 eliminado - validación incluida en weighted_ensemble.py

echo "========================================================================"
echo "✅ PIPELINE COMPLETADO"
echo "========================================================================"
echo ""
echo "📋 Archivos generados:"
echo "  ✅ datos/processed/atp_matches_clean.csv - Datos limpios"
echo "  ✅ datos/processed/dataset_features_fase3_completas.csv - Dataset con features"
echo "  ✅ modelos/xgboost_optimizado.pkl - Mejor modelo individual"
echo "  ✅ modelos/random_forest_calibrado.pkl - RF calibrado"
echo "  ✅ modelos/gradient_boosting_calibrado.pkl - GB calibrado"
echo "  ✅ resultados/weighted_ensemble_metrics.csv - Métricas del ensemble"
echo "  ✅ resultados/selected_features.txt - 30 features seleccionadas"
echo ""
echo "🎯 Modelo Final: Weighted Ensemble (2022-2025)"
echo "   Accuracy esperado: ~70.20%"
echo "   Brier Score esperado: ~0.1980"
echo ""
echo "🎉 ¡Listo para usar!"
