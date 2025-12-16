"""
Optimización de hiperparámetros para modelos de ML
Fase 3 - Optimización
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, brier_score_loss
import joblib
from pathlib import Path
import time
import logging
import warnings

# Suprimir warnings de sklearn para output más limpio
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='sklearn')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Optimización de hiperparámetros
    """
    
    def __init__(self, modelo_base, param_grid, cv=5):
        """
        Args:
            modelo_base: Modelo a optimizar
            param_grid: Diccionario con parámetros a probar
            cv: Número de folds para cross-validation
        """
        self.modelo_base = modelo_base
        self.param_grid = param_grid
        self.cv = cv
        self.mejor_modelo = None
        self.mejor_params = None
        self.cv_results = None
    
    def tune_random_search(self, X_train, y_train, n_iter=50, scoring='neg_brier_score'):
        """
        Hyperparameter tuning con Random Search (más rápido)
        
        Args:
            scoring: Métrica a optimizar
                - 'accuracy'
                - 'neg_brier_score' (recomendado para apuestas)
                - 'roc_auc'
        """
        
        logger.info("=" * 60)
        logger.info("⚙️  HYPERPARAMETER TUNING - RANDOM SEARCH")
        logger.info("=" * 60)
        
        logger.info(f"\n🎲 Iteraciones: {n_iter}")
        logger.info(f"📈 CV Folds: {self.cv}")
        logger.info(f"🎯 Métrica: {scoring}")
        
        # Crear random search
        random_search = RandomizedSearchCV(
            estimator=self.modelo_base,
            param_distributions=self.param_grid,
            n_iter=n_iter,
            cv=self.cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
        
        # Entrenar
        logger.info(f"\n🔄 Entrenando... (esto puede tardar varios minutos)")
        start_time = time.time()
        
        random_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n✅ Completado en {elapsed_time/60:.1f} minutos")
        
        # Mejores resultados
        self.mejor_modelo = random_search.best_estimator_
        self.mejor_params = random_search.best_params_
        self.cv_results = pd.DataFrame(random_search.cv_results_)
        
        logger.info(f"\n🏆 MEJORES PARÁMETROS:")
        for param, value in self.mejor_params.items():
            logger.info(f"   {param}: {value}")
        
        logger.info(f"\n📊 Mejor score (CV): {-random_search.best_score_:.4f}")
        
        return self.mejor_modelo, self.mejor_params
    
    def analizar_resultados(self):
        """
        Analiza y visualiza resultados del tuning
        """
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 ANÁLISIS DE RESULTADOS")
        logger.info("=" * 60)
        
        # Top 10 combinaciones
        top_results = self.cv_results.nsmallest(10, 'rank_test_score')
        
        logger.info("\n🏆 Top 10 combinaciones:")
        logger.info("\n" + top_results[['params', 'mean_test_score', 'std_test_score']].to_string())
        
        # Guardar resultados completos
        Path("resultados").mkdir(exist_ok=True)
        self.cv_results.to_csv('resultados/hyperparameter_tuning_results.csv', index=False)
        logger.info("\n💾 Resultados completos guardados: resultados/hyperparameter_tuning_results.csv")
        
        return top_results


def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost con parámetros recomendados
    """
    
    if not XGBOOST_AVAILABLE:
        logger.warning("⚠️  XGBoost no disponible, saltando tuning")
        return None
    
    logger.info("\n" + "=" * 60)
    logger.info("TUNING XGBOOST")
    logger.info("=" * 60)
    
    # Grid de parámetros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    modelo_base = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    
    # Usar Random Search (más rápido)
    tuner = HyperparameterTuner(
        modelo_base=modelo_base,
        param_grid=param_grid,
        cv=5
    )
    
    mejor_modelo, mejor_params = tuner.tune_random_search(
        X_train, y_train,
        n_iter=50,  # 50 combinaciones aleatorias
        scoring='neg_brier_score'
    )
    
    # Analizar
    tuner.analizar_resultados()
    
    # Guardar modelo
    Path("modelos").mkdir(exist_ok=True)
    joblib.dump(mejor_modelo, "modelos/xgboost_optimizado.pkl")
    logger.info("\n💾 Modelo optimizado guardado: modelos/xgboost_optimizado.pkl")
    
    return mejor_modelo


def tune_lightgbm(X_train, y_train):
    """
    Tune LightGBM con parámetros recomendados
    """
    
    if not LIGHTGBM_AVAILABLE:
        logger.warning("⚠️  LightGBM no disponible, saltando tuning")
        return None
    
    logger.info("\n" + "=" * 60)
    logger.info("TUNING LIGHTGBM")
    logger.info("=" * 60)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_samples': [10, 20, 30]
    }
    
    modelo_base = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    
    tuner = HyperparameterTuner(
        modelo_base=modelo_base,
        param_grid=param_grid,
        cv=5
    )
    
    mejor_modelo, mejor_params = tuner.tune_random_search(
        X_train, y_train,
        n_iter=50,
        scoring='neg_brier_score'
    )
    
    tuner.analizar_resultados()
    
    # Guardar
    joblib.dump(mejor_modelo, "modelos/lightgbm_optimizado.pkl")
    logger.info("\n💾 Modelo optimizado guardado: modelos/lightgbm_optimizado.pkl")
    
    return mejor_modelo


if __name__ == "__main__":
    # Cargar datos
    logger.info("📂 Cargando dataset...")
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').reset_index(drop=True)
    
    # Features
    feature_cols = [col for col in df.columns if col not in ['resultado', 'fecha']]
    
    # Split: usar solo train (60%)
    n = len(df)
    train_end = int(n * 0.6)
    
    X_train = df.iloc[:train_end][feature_cols]
    y_train = df.iloc[:train_end]['resultado']
    
    logger.info(f"\n📊 Datos de entrenamiento: {len(X_train)}")
    logger.info(f"📊 Features: {len(feature_cols)}")
    
    # Tune XGBoost
    if XGBOOST_AVAILABLE:
        mejor_xgb = tune_xgboost(X_train, y_train)
    
    # Tune LightGBM
    if LIGHTGBM_AVAILABLE:
        mejor_lgbm = tune_lightgbm(X_train, y_train)
    
    logger.info("\n✅ Hyperparameter tuning completado!")
