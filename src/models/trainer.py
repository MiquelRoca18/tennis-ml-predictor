"""
Entrenamiento de modelos de Machine Learning para predicción de tenis - Fase 3
Versión actualizada que soporta features de Fase 3 y Brier Score
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_temporal(df, train_size=0.6, val_size=0.2, test_size=0.2):
    """
    Split temporal: 60% train, 20% val, 20% test
    
    Args:
        df: DataFrame con columna 'fecha'
        train_size: Proporción de datos para entrenamiento
        val_size: Proporción de datos para validación
        test_size: Proporción de datos para test
        
    Returns:
        Tuple (df_train, df_val, df_test)
    """
    
    # Ordenar por fecha
    df = df.sort_values('fecha').reset_index(drop=True)
    
    # Calcular índices de corte
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    # Split
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]
    
    logger.info("=" * 60)
    logger.info("✂️  SPLIT TEMPORAL")
    logger.info("=" * 60)
    logger.info(f"📊 Training set: {len(df_train)} partidos")
    logger.info(f"   Fechas: {df_train['fecha'].min()} - {df_train['fecha'].max()}")
    logger.info(f"\n📊 Validation set: {len(df_val)} partidos")
    logger.info(f"   Fechas: {df_val['fecha'].min()} - {df_val['fecha'].max()}")
    logger.info(f"\n📊 Test set: {len(df_test)} partidos")
    logger.info(f"   Fechas: {df_test['fecha'].min()} - {df_test['fecha'].max()}")
    
    return df_train, df_val, df_test


def entrenar_modelo(modelo, nombre_modelo, X_train, y_train, X_val, y_val, X_test, y_test, calibrar=True):
    """
    Entrena un modelo y calcula métricas completas
    
    Args:
        modelo: Modelo de sklearn a entrenar
        nombre_modelo: Nombre descriptivo del modelo
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        X_test, y_test: Datos de test
        calibrar: Si True, calibra el modelo usando el set de validación
        
    Returns:
        Tuple (modelo_final, metricas)
    """
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🤖 ENTRENANDO {nombre_modelo.upper()}")
    logger.info("=" * 60)
    
    # Entrenar
    logger.info("🔄 Entrenando modelo...")
    modelo.fit(X_train, y_train)
    logger.info("✅ Modelo entrenado!")
    
    # Predecir
    y_pred_train = modelo.predict(X_train)
    y_pred_val = modelo.predict(X_val)
    y_pred_test = modelo.predict(X_test)
    
    y_prob_train = modelo.predict_proba(X_train)[:, 1]
    y_prob_val = modelo.predict_proba(X_val)[:, 1]
    y_prob_test = modelo.predict_proba(X_test)[:, 1]
    
    # Métricas
    metricas = {
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
    
    logger.info(f"\n📊 RESULTADOS:")
    logger.info(f"   Accuracy: Train {metricas['accuracy_train']*100:.2f}% | "
              f"Val {metricas['accuracy_val']*100:.2f}% | "
              f"Test {metricas['accuracy_test']*100:.2f}%")
    logger.info(f"   Brier:    Train {metricas['brier_train']:.4f} | "
              f"Val {metricas['brier_val']:.4f} | "
              f"Test {metricas['brier_test']:.4f}")
    logger.info(f"   AUC-ROC:  Train {metricas['auc_train']:.4f} | "
              f"Val {metricas['auc_val']:.4f} | "
              f"Test {metricas['auc_test']:.4f}")
    
    # Verificar overfitting
    if metricas['accuracy_train'] - metricas['accuracy_test'] > 0.10:
        logger.warning(f"   ⚠️  Warning: Posible overfitting (diferencia: {(metricas['accuracy_train']-metricas['accuracy_test'])*100:.1f}%)")
    else:
        logger.info(f"   ✅ Buen balance train/test")
    
    # Calibrar si se solicita
    modelo_final = modelo
    if calibrar:
        logger.info("\n🔧 Calibrando modelo...")
        modelo_calibrado = CalibratedClassifierCV(modelo, method='sigmoid', cv='prefit')
        modelo_calibrado.fit(X_val, y_val)
        
        # Evaluar modelo calibrado
        y_prob_test_cal = modelo_calibrado.predict_proba(X_test)[:, 1]
        brier_cal = brier_score_loss(y_test, y_prob_test_cal)
        
        metricas['brier_test_calibrado'] = brier_cal
        logger.info(f"   Brier (calibrado): {brier_cal:.4f}")
        
        modelo_final = modelo_calibrado
    
    # Classification report
    logger.info("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['Perdedor', 'Ganador']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {nombre_modelo}')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    
    # Guardar
    output_dir = Path("resultados")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = nombre_modelo.lower().replace(' ', '_')
    plt.savefig(output_dir / f'confusion_matrix_{filename}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 Confusion matrix guardada en: {output_dir / f'confusion_matrix_{filename}.png'}")
    
    # Feature importance (si el modelo lo soporta)
    if hasattr(modelo, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': modelo.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\n🎯 Top 10 Features:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
        plt.figure(figsize=(12, 8))
        top_20 = feature_importance.head(20)
        plt.barh(top_20['feature'], top_20['importance'])
        plt.xlabel('Importance')
        plt.title(f'Feature Importance - {nombre_modelo}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / f'feature_importance_{filename}.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 Feature importance guardado en: {output_dir / f'feature_importance_{filename}.png'}")
    
    return modelo_final, metricas


def guardar_modelo(modelo, nombre):
    """
    Guarda el modelo entrenado
    
    Args:
        modelo: Modelo entrenado
        nombre: Nombre del archivo (sin extensión)
    """
    output_dir = Path("modelos")
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{nombre}.pkl"
    joblib.dump(modelo, filepath)
    logger.info(f"💾 Modelo guardado en: {filepath}")


if __name__ == "__main__":
    # Crear carpetas
    Path("resultados").mkdir(parents=True, exist_ok=True)
    
    # Cargar datos con features de Fase 3
    logger.info("📂 Cargando dataset con features de Fase 3...")
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    logger.info(f"   Dataset: {len(df)} partidos")
    
    # Features (todas excepto resultado y fecha)
    feature_columns = [col for col in df.columns if col not in ['resultado', 'fecha']]
    logger.info(f"   Features: {len(feature_columns)}")
    
    # Split temporal (60/20/20)
    df_train, df_val, df_test = split_temporal(df, train_size=0.6, val_size=0.2, test_size=0.2)
    
    X_train = df_train[feature_columns]
    y_train = df_train['resultado']
    X_val = df_val[feature_columns]
    y_val = df_val['resultado']
    X_test = df_test[feature_columns]
    y_test = df_test['resultado']
    
    # Entrenar Random Forest
    modelo_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    modelo_rf_final, metricas_rf = entrenar_modelo(
        modelo_rf, "Random Forest",
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        calibrar=True
    )
    
    guardar_modelo(modelo_rf_final, "modelo_rf_fase3_calibrado")
    
    # Verificar criterios de éxito
    logger.info("\n" + "=" * 60)
    logger.info("🎯 VERIFICACIÓN DE CRITERIOS DE FASE 3")
    logger.info("=" * 60)
    
    criterios = {
        'Accuracy > 62%': metricas_rf['accuracy_test'] > 0.62,
        'Brier Score < 0.18': metricas_rf.get('brier_test_calibrado', metricas_rf['brier_test']) < 0.18
    }
    
    for criterio, cumple in criterios.items():
        simbolo = "✅" if cumple else "❌"
        logger.info(f"{simbolo} {criterio}")
    
    if all(criterios.values()):
        logger.info("\n🎉 ¡CRITERIOS DE FASE 3 CUMPLIDOS!")
        logger.info("✅ Listo para pasar a FASE 4")
    else:
        logger.info("\n⚠️  Algunos criterios no se cumplen")
        logger.info("💡 Ejecuta run_fase3_optimization.py para optimizar modelos")
    
    logger.info("\n✅ Entrenamiento completado!")
