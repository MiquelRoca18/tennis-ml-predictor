"""
Comparación de múltiples algoritmos de ML para predicción de tenis
Fase 3 - Optimización
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost no disponible, se omitirá")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM no disponible, se omitirá")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compara múltiples algoritmos de ML
    """
    
    def __init__(self):
        self.modelos = {}
        self.resultados = {}
    
    def agregar_modelo(self, nombre, modelo):
        """Añade un modelo al comparador"""
        self.modelos[nombre] = modelo
    
    def inicializar_modelos_default(self):
        """
        Inicializa modelos con configuración base
        """
        
        self.modelos = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Añadir XGBoost si está disponible
        if XGBOOST_AVAILABLE:
            self.modelos['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        
        # Añadir LightGBM si está disponible
        if LIGHTGBM_AVAILABLE:
            self.modelos['LightGBM'] = LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        logger.info(f"✅ {len(self.modelos)} modelos inicializados")
    
    def entrenar_y_evaluar(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Entrena y evalúa todos los modelos
        """
        
        logger.info("\n" + "=" * 60)
        logger.info("🤖 ENTRENANDO Y EVALUANDO MODELOS")
        logger.info("=" * 60)
        
        for nombre, modelo in self.modelos.items():
            logger.info(f"\n📊 {nombre}...")
            
            try:
                # Entrenar
                modelo.fit(X_train, y_train)
                
                # Predecir
                y_pred_train = modelo.predict(X_train)
                y_pred_val = modelo.predict(X_val)
                y_pred_test = modelo.predict(X_test)
                
                y_prob_train = modelo.predict_proba(X_train)[:, 1]
                y_prob_val = modelo.predict_proba(X_val)[:, 1]
                y_prob_test = modelo.predict_proba(X_test)[:, 1]
                
                # Métricas
                resultados = {
                    'accuracy_train': accuracy_score(y_train, y_pred_train),
                    'accuracy_val': accuracy_score(y_val, y_pred_val),
                    'accuracy_test': accuracy_score(y_test, y_pred_test),
                    'brier_train': brier_score_loss(y_train, y_prob_train),
                    'brier_val': brier_score_loss(y_val, y_prob_val),
                    'brier_test': brier_score_loss(y_test, y_prob_test),
                    'auc_train': roc_auc_score(y_train, y_prob_train),
                    'auc_val': roc_auc_score(y_val, y_prob_val),
                    'auc_test': roc_auc_score(y_test, y_prob_test)
                }
                
                self.resultados[nombre] = resultados
                
                logger.info(f"   Accuracy: Train {resultados['accuracy_train']*100:.2f}% | "
                          f"Val {resultados['accuracy_val']*100:.2f}% | "
                          f"Test {resultados['accuracy_test']*100:.2f}%")
                logger.info(f"   Brier:    Train {resultados['brier_train']:.4f} | "
                          f"Val {resultados['brier_val']:.4f} | "
                          f"Test {resultados['brier_test']:.4f}")
                
                # Calibrar (usando isotonic para mejor Brier Score)
                logger.info(f"   Calibrando...")
                modelo_calibrado = CalibratedClassifierCV(modelo, method='isotonic', cv='prefit')
                modelo_calibrado.fit(X_val, y_val)
                
                y_prob_test_cal = modelo_calibrado.predict_proba(X_test)[:, 1]
                brier_cal = brier_score_loss(y_test, y_prob_test_cal)
                
                resultados['brier_test_calibrado'] = brier_cal
                logger.info(f"   Brier (calibrado): {brier_cal:.4f}")
                
                # Guardar modelo calibrado
                Path("modelos").mkdir(exist_ok=True)
                model_filename = f"modelos/{nombre.lower().replace(' ', '_')}_calibrado.pkl"
                joblib.dump(modelo_calibrado, model_filename)
                logger.info(f"   💾 Guardado: {model_filename}")
                
            except Exception as e:
                logger.error(f"   ❌ Error entrenando {nombre}: {e}")
                continue
        
        return self.resultados
    
    def comparar_resultados(self):
        """
        Compara resultados y muestra el mejor
        """
        
        logger.info("\n" + "=" * 60)
        logger.info("🏆 COMPARACIÓN DE RESULTADOS")
        logger.info("=" * 60)
        
        df_resultados = pd.DataFrame(self.resultados).T
        
        # Ordenar por Brier Score test (menor es mejor)
        df_resultados = df_resultados.sort_values('brier_test')
        
        logger.info("\n📊 Ranking por Brier Score (Test):")
        logger.info("\n" + df_resultados[['accuracy_test', 'brier_test', 'brier_test_calibrado']].to_string())
        
        # Mejor modelo
        mejor_modelo = df_resultados.index[0]
        logger.info(f"\n🥇 MEJOR MODELO: {mejor_modelo}")
        logger.info(f"   Accuracy Test: {df_resultados.loc[mejor_modelo, 'accuracy_test']*100:.2f}%")
        logger.info(f"   Brier Score Test: {df_resultados.loc[mejor_modelo, 'brier_test']:.4f}")
        logger.info(f"   Brier Calibrado: {df_resultados.loc[mejor_modelo, 'brier_test_calibrado']:.4f}")
        
        # Visualización
        self._visualizar_comparacion(df_resultados)
        
        # Guardar resultados
        df_resultados.to_csv('resultados/model_comparison_results.csv')
        logger.info("\n💾 Resultados guardados: resultados/model_comparison_results.csv")
        
        return mejor_modelo, df_resultados
    
    def _visualizar_comparacion(self, df_resultados):
        """
        Crea visualización de comparación
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy
        ax = axes[0]
        df_acc = df_resultados[['accuracy_train', 'accuracy_val', 'accuracy_test']]
        df_acc.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('')
        ax.legend(['Train', 'Validation', 'Test'])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.62, color='red', linestyle='--', label='Target (62%)', alpha=0.7)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Brier Score
        ax = axes[1]
        df_brier = df_resultados[['brier_train', 'brier_val', 'brier_test', 'brier_test_calibrado']]
        df_brier.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax.set_title('Brier Score Comparison (lower is better)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Brier Score', fontsize=12)
        ax.set_xlabel('')
        ax.legend(['Train', 'Validation', 'Test', 'Test Calibrated'])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.18, color='red', linestyle='--', label='Target (0.18)', alpha=0.7)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        Path("resultados").mkdir(exist_ok=True)
        plt.savefig('resultados/model_comparison.png', dpi=150, bbox_inches='tight')
        logger.info("\n📊 Gráfico guardado: resultados/model_comparison.png")
        plt.close()


if __name__ == "__main__":
    # Cargar datos con features completas
    logger.info("📂 Cargando dataset...")
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').reset_index(drop=True)
    
    logger.info(f"   Dataset: {len(df)} partidos")
    
    # Features (todas excepto resultado y fecha)
    feature_cols = [col for col in df.columns if col not in ['resultado', 'fecha']]
    logger.info(f"   Features: {len(feature_cols)}")
    
    # Split: 60% train, 20% val, 20% test
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    X_train = df.iloc[:train_end][feature_cols]
    y_train = df.iloc[:train_end]['resultado']
    X_val = df.iloc[train_end:val_end][feature_cols]
    y_val = df.iloc[train_end:val_end]['resultado']
    X_test = df.iloc[val_end:][feature_cols]
    y_test = df.iloc[val_end:]['resultado']
    
    logger.info(f"\n📊 Splits: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    
    # Comparar modelos
    comparador = ModelComparator()
    comparador.inicializar_modelos_default()
    
    resultados = comparador.entrenar_y_evaluar(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )
    
    mejor_modelo, df_res = comparador.comparar_resultados()
    
    logger.info("\n✅ Comparación completada!")
