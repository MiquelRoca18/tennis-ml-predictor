"""
Ensemble Stacking - Combina modelos con meta-learner
Mejora esperada: Accuracy +0.5% a +1.0%, Brier -0.01 a -0.015
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def crear_y_entrenar_stacking(X_train, y_train, X_test, y_test):
    """
    Crea y entrena un Stacking Ensemble con los 3 mejores modelos
    """
    
    logger.info("=" * 70)
    logger.info("🚀 STACKING ENSEMBLE - Meta-Learning")
    logger.info("=" * 70)
    
    # Crear modelos base desde cero (no usar calibrados pre-entrenados)
    logger.info("\n🔧 Creando modelos base...")
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    try:
        from xgboost import XGBClassifier
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
        logger.error("   ❌ XGBoost no disponible")
        return None
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    logger.info("   ✅ Random Forest inicializado")
    
    # XGBoost con mejores parámetros del tuning
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    logger.info("   ✅ XGBoost inicializado")
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    logger.info("   ✅ Gradient Boosting inicializado")
    
    # Crear stacking
    logger.info("\n🔧 Creando Stacking Classifier...")
    estimators = [
        ('rf', rf),
        ('xgb', xgb),
        ('gb', gb)
    ]
    
    # Meta-learner: Logistic Regression
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("   Estimadores base: Random Forest, XGBoost, Gradient Boosting")
    logger.info("   Meta-learner: Logistic Regression")
    logger.info("   Cross-validation: 5 folds")
    
    # Entrenar
    logger.info("\n🔄 Entrenando Stacking Ensemble...")
    logger.info("   (Esto puede tomar varios minutos...)")
    
    try:
        stacking.fit(X_train, y_train)
        logger.info("   ✅ Entrenamiento completado")
    except Exception as e:
        logger.error(f"   ❌ Error durante entrenamiento: {e}")
        return None
    
    # Evaluar
    logger.info("\n🔮 Evaluando en Test Set...")
    y_pred = stacking.predict(X_test)
    y_prob = stacking.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    logger.info("\n" + "=" * 70)
    logger.info("🎯 RESULTADOS DEL STACKING ENSEMBLE")
    logger.info("=" * 70)
    logger.info(f"\n📊 Métricas en Test Set:")
    logger.info(f"   Accuracy:     {accuracy*100:.2f}%")
    logger.info(f"   Brier Score:  {brier:.4f}")
    logger.info(f"   AUC-ROC:      {auc:.4f}")
    
    
    # Comparación con weighted ensemble si existe
    logger.info(f"\n📈 Comparación con Weighted Ensemble:")
    try:
        weighted_metrics = pd.read_csv('resultados/weighted_ensemble_metrics.csv', index_col=0)
        weighted_acc = weighted_metrics.loc['accuracy'] * 100
        weighted_brier = weighted_metrics.loc['brier_score']
        
        logger.info(f"\n   Weighted Ensemble:")
        logger.info(f"      Accuracy: {weighted_acc:.2f}% → Stacking: {accuracy*100:.2f}% ({(accuracy*100-weighted_acc):+.2f}%)")
        logger.info(f"      Brier:    {weighted_brier:.4f} → Stacking: {brier:.4f} ({brier-weighted_brier:+.4f})")
    except:
        logger.info("   (Weighted ensemble no encontrado para comparación)")
    
    
    # Verificación de criterios
    logger.info("\n" + "=" * 70)
    logger.info("✅ VERIFICACIÓN DE CRITERIOS")
    logger.info("=" * 70)
    
    objetivo_accuracy = 70.0
    objetivo_brier = 0.18
    
    cumple_accuracy = accuracy * 100 >= objetivo_accuracy
    cumple_brier = brier < objetivo_brier
    
    logger.info(f"\n🎯 Objetivos:")
    logger.info(f"   Accuracy ≥ {objetivo_accuracy}%:  {'✅ SÍ' if cumple_accuracy else '❌ NO'} ({accuracy*100:.2f}%)")
    logger.info(f"   Brier < {objetivo_brier}:      {'✅ SÍ' if cumple_brier else '❌ NO'} ({brier:.4f})")
    
    if cumple_accuracy and cumple_brier:
        logger.info("\n🎉 ¡ÉXITO COMPLETO! Ambos objetivos alcanzados")
        logger.info("   → Proceder a Fase 4")
    elif cumple_accuracy or brier < 0.19:
        logger.info("\n✅ ¡EXCELENTE! Muy cerca de los objetivos")
        logger.info("   → Resultado aceptable para producción")
    else:
        logger.info("\n⚠️  Considerar implementar features adicionales")
        logger.info("   → Ver features_momentum.py en plan de mejora")
    
    # Guardar modelo
    logger.info("\n💾 Guardando modelo...")
    Path("modelos").mkdir(exist_ok=True)
    joblib.dump(stacking, 'modelos/stacking_ensemble.pkl')
    logger.info("   ✅ Modelo guardado: modelos/stacking_ensemble.pkl")
    
    # Guardar predicciones
    resultados_df = pd.DataFrame({
        'prob_stacking': y_prob,
        'pred_stacking': y_pred,
        'y_true': y_test
    })
    
    Path("resultados").mkdir(exist_ok=True)
    resultados_df.to_csv('resultados/stacking_ensemble_predictions.csv', index=False)
    logger.info("   ✅ Predicciones guardadas: resultados/stacking_ensemble_predictions.csv")
    
    # Guardar métricas
    metricas = {
        'accuracy': accuracy,
        'brier_score': brier,
        'auc_roc': auc,
        'cumple_accuracy': cumple_accuracy,
        'cumple_brier': cumple_brier
    }
    
    pd.Series(metricas).to_csv('resultados/stacking_ensemble_metrics.csv')
    logger.info("   ✅ Métricas guardadas: resultados/stacking_ensemble_metrics.csv")
    
    logger.info("\n✅ Stacking Ensemble completado!")
    
    return stacking


if __name__ == "__main__":
    # Cargar datos
    logger.info("📂 Cargando dataset...")
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha').reset_index(drop=True)
    
    logger.info(f"   Dataset: {len(df)} partidos")
    
    # Cargar features seleccionadas
    try:
        selected_features = pd.read_csv('resultados/selected_features.txt', header=None)[0].tolist()
        logger.info(f"   ✅ {len(selected_features)} features seleccionadas cargadas")
    except Exception as e:
        logger.error(f"   ❌ Error cargando features: {e}")
        logger.error("   → Ejecutar primero: python run_fase3_optimization.py")
        exit(1)
    
    # Split: 60% train, 20% val, 20% test
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    X_train = df.iloc[:train_end][selected_features]
    y_train = df.iloc[:train_end]['resultado']
    X_test = df.iloc[val_end:][selected_features]
    y_test = df.iloc[val_end:]['resultado']
    
    logger.info(f"   Train set: {len(X_train)} partidos")
    logger.info(f"   Test set:  {len(X_test)} partidos")
    
    # Crear y entrenar stacking
    stacking_model = crear_y_entrenar_stacking(X_train, y_train, X_test, y_test)
    
    if stacking_model:
        logger.info("\n🎊 ¡Proceso completado exitosamente!")
    else:
        logger.error("\n❌ Error en el proceso de stacking")
