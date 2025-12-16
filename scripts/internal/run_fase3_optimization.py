"""
Pipeline completo de optimización - Fase 3
Orden correcto: Feature Selection → Model Comparison → Hyperparameter Tuning → Validation
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import warnings

# Suprimir warnings para output más limpio
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.models.comparacion_modelos import ModelComparator
from src.models.hyperparameter_tuning import tune_xgboost, tune_lightgbm
from src.models.feature_selection import FeatureSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Ejecuta el pipeline completo de optimización de Fase 3
    ORDEN CORRECTO: Feature Selection → Comparación → Tuning → Validación
    """
    
    logger.info("=" * 70)
    logger.info("🚀 FASE 3 - PIPELINE DE OPTIMIZACIÓN COMPLETO")
    logger.info("=" * 70)
    
    # ========================================================================
    # PASO 0: Cargar datos
    # ========================================================================
    logger.info("\n📂 PASO 0: Cargando dataset...")
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').reset_index(drop=True)
    
    logger.info(f"   Dataset: {len(df)} partidos")
    
    # Todas las features disponibles
    all_features = [col for col in df.columns if col not in ['resultado', 'fecha']]
    logger.info(f"   Features totales: {len(all_features)}")
    
    # Split: 60% train, 20% val, 20% test
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    # ========================================================================
    # PASO 1: Feature Selection (PRIMERO)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("🎯 PASO 1: FEATURE SELECTION")
    logger.info("=" * 70)
    
    X_train_all = df.iloc[:train_end][all_features]
    y_train = df.iloc[:train_end]['resultado']
    
    selector = FeatureSelector(X_train_all, y_train)
    
    # Método 1: Tree-based
    imp_tree = selector.feature_importance_tree_based()
    
    # Método 2: F-statistic
    imp_f = selector.feature_importance_statistical('f_classif')
    
    # Método 3: Mutual Information
    imp_mi = selector.feature_importance_statistical('mutual_info')
    
    # Comparar métodos
    features_comunes = selector.comparar_metodos()
    
    # Eliminar correlacionadas
    X_sin_corr, dropped = selector.eliminar_features_correlacionadas(threshold=0.9)
    
    # Seleccionar top 30
    X_reducido, selected_features = selector.seleccionar_mejores_k(k=30, method='tree_based')
    
    # Guardar features seleccionadas
    if selected_features:
        Path("resultados").mkdir(exist_ok=True)
        pd.Series(selected_features).to_csv('resultados/selected_features.txt', index=False, header=False)
        logger.info("\n💾 Features seleccionadas guardadas: resultados/selected_features.txt")
    
    # ========================================================================
    # PASO 2: Preparar datos con features seleccionadas
    # ========================================================================
    logger.info(f"\n📊 Usando {len(selected_features)} features seleccionadas")
    
    X_train = df.iloc[:train_end][selected_features]
    X_val = df.iloc[train_end:val_end][selected_features]
    X_test = df.iloc[val_end:][selected_features]
    y_val = df.iloc[train_end:val_end]['resultado']
    y_test = df.iloc[val_end:]['resultado']
    
    logger.info(f"   Splits: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    
    # ========================================================================
    # PASO 3: Comparación de modelos
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("📊 PASO 2: COMPARACIÓN DE MODELOS")
    logger.info("=" * 70)
    
    comparador = ModelComparator()
    comparador.inicializar_modelos_default()
    
    resultados = comparador.entrenar_y_evaluar(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )
    
    mejor_modelo, df_res = comparador.comparar_resultados()
    
    # ========================================================================
    # PASO 4: Hyperparameter tuning
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("⚙️  PASO 3: HYPERPARAMETER TUNING")
    logger.info("=" * 70)
    
    # Tune XGBoost
    try:
        logger.info("\n🔧 Optimizando XGBoost...")
        mejor_xgb = tune_xgboost(X_train, y_train)
    except Exception as e:
        logger.error(f"❌ Error en tuning XGBoost: {e}")
    
    # Tune LightGBM
    try:
        logger.info("\n🔧 Optimizando LightGBM...")
        mejor_lgbm = tune_lightgbm(X_train, y_train)
    except Exception as e:
        logger.error(f"❌ Error en tuning LightGBM: {e}")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("🎉 PIPELINE COMPLETADO")
    logger.info("=" * 70)
    
    logger.info("\n📋 Archivos generados:")
    logger.info("   ✅ resultados/selected_features.txt - 30 features seleccionadas")
    logger.info("   ✅ resultados/feature_importance_tree.png - Importancia de features")
    logger.info("   ✅ resultados/model_comparison.png - Comparación de modelos")
    logger.info("   ✅ modelos/*_calibrado.pkl - Modelos calibrados")
    logger.info("   ✅ modelos/*_optimizado.pkl - Modelos optimizados")
    
    logger.info("\n🎯 Próximo paso:")
    logger.info("   python src/models/validacion_final_fase3.py")
    logger.info("\n✅ ¡Listo para validación!")


if __name__ == "__main__":
    main()
