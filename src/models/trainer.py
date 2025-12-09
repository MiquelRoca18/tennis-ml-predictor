"""
Entrenamiento de modelos de Machine Learning para predicción de tenis
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_temporal(df, test_size=0.2):
    """
    Split temporal: entrenamos con partidos pasados, testeamos con recientes
    
    Args:
        df: DataFrame con columna 'fecha'
        test_size: Proporción de datos para test
        
    Returns:
        Tuple (df_train, df_test)
    """
    
    # Ordenar por fecha
    df = df.sort_values('fecha').reset_index(drop=True)
    
    # Calcular índice de corte
    split_idx = int(len(df) * (1 - test_size))
    
    # Split
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    logger.info("=" * 60)
    logger.info("✂️  SPLIT TEMPORAL")
    logger.info("=" * 60)
    logger.info(f"📊 Training set: {len(df_train)} partidos")
    logger.info(f"   Fechas: {df_train['fecha'].min()} - {df_train['fecha'].max()}")
    logger.info(f"\n📊 Test set: {len(df_test)} partidos")
    logger.info(f"   Fechas: {df_test['fecha'].min()} - {df_test['fecha'].max()}")
    
    return df_train, df_test


def entrenar_random_forest(X_train, y_train, X_test, y_test):
    """
    Entrena un Random Forest
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de test
        
    Returns:
        Tuple (modelo, accuracy_test)
    """
    
    logger.info("\n" + "=" * 60)
    logger.info("🌲 ENTRENANDO RANDOM FOREST")
    logger.info("=" * 60)
    
    # Crear modelo
    modelo = RandomForestClassifier(
        n_estimators=100,        # 100 árboles
        max_depth=10,            # Profundidad máxima (evita overfitting)
        min_samples_split=20,    # Mínimo samples para split
        min_samples_leaf=10,     # Mínimo samples en hoja
        random_state=42,
        n_jobs=-1                # Usar todos los CPUs
    )
    
    # Entrenar
    logger.info("🔄 Entrenando modelo...")
    modelo.fit(X_train, y_train)
    logger.info("✅ Modelo entrenado!")
    
    # Predecir
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Probabilidades para AUC
    y_pred_proba_test = modelo.predict_proba(X_test)[:, 1]
    
    # Métricas
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    auc_test = roc_auc_score(y_test, y_pred_proba_test)
    
    logger.info(f"\n📊 RESULTADOS:")
    logger.info(f"   Accuracy TRAIN: {acc_train*100:.2f}%")
    logger.info(f"   Accuracy TEST:  {acc_test*100:.2f}%")
    logger.info(f"   AUC-ROC TEST:   {auc_test:.4f}")
    
    if acc_train - acc_test > 0.10:
        logger.warning(f"   ⚠️  Warning: Posible overfitting (diferencia: {(acc_train-acc_test)*100:.1f}%)")
    else:
        logger.info(f"   ✅ Buen balance train/test")
    
    # Classification report
    logger.info("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['Perdedor', 'Ganador']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Random Forest')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    
    # Guardar
    output_dir = Path("resultados")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'confusion_matrix_rf.png')
    plt.close()
    logger.info(f"📊 Confusion matrix guardada en: {output_dir / 'confusion_matrix_rf.png'}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': modelo.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\n🎯 Feature Importance:")
    for _, row in feature_importance.iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_rf.png')
    plt.close()
    logger.info(f"📊 Feature importance guardado en: {output_dir / 'feature_importance_rf.png'}")
    
    return modelo, acc_test


def entrenar_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Entrena una Regresión Logística (alternativa más simple)
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de test
        
    Returns:
        Tuple (modelo, accuracy_test)
    """
    
    logger.info("\n" + "=" * 60)
    logger.info("📈 ENTRENANDO LOGISTIC REGRESSION")
    logger.info("=" * 60)
    
    # Crear modelo
    modelo = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    
    # Entrenar
    logger.info("🔄 Entrenando modelo...")
    modelo.fit(X_train, y_train)
    logger.info("✅ Modelo entrenado!")
    
    # Predecir
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Probabilidades para AUC
    y_pred_proba_test = modelo.predict_proba(X_test)[:, 1]
    
    # Métricas
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    auc_test = roc_auc_score(y_test, y_pred_proba_test)
    
    logger.info(f"\n📊 RESULTADOS:")
    logger.info(f"   Accuracy TRAIN: {acc_train*100:.2f}%")
    logger.info(f"   Accuracy TEST:  {acc_test*100:.2f}%")
    logger.info(f"   AUC-ROC TEST:   {auc_test:.4f}")
    
    # Classification report
    logger.info("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['Perdedor', 'Ganador']))
    
    return modelo, acc_test


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
    
    # Cargar datos
    df = pd.read_csv("datos/processed/dataset_con_features.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Features
    feature_columns = [
        'jugador_rank', 'oponente_rank', 'rank_diff', 'rank_ratio',
        'jugador_top10', 'oponente_top10', 'jugador_top50', 'oponente_top50',
        'surface_hard', 'surface_clay', 'surface_grass'
    ]
    
    # Split temporal
    df_train, df_test = split_temporal(df, test_size=0.2)
    
    X_train = df_train[feature_columns]
    y_train = df_train['resultado']
    X_test = df_test[feature_columns]
    y_test = df_test['resultado']
    
    # Entrenar Random Forest
    modelo_rf, acc_rf = entrenar_random_forest(X_train, y_train, X_test, y_test)
    guardar_modelo(modelo_rf, "modelo_rf_v1")
    
    # Entrenar Logistic Regression
    modelo_lr, acc_lr = entrenar_logistic_regression(X_train, y_train, X_test, y_test)
    guardar_modelo(modelo_lr, "modelo_lr_v1")
    
    # Comparar
    logger.info("\n" + "=" * 60)
    logger.info("🏆 COMPARACIÓN DE MODELOS")
    logger.info("=" * 60)
    logger.info(f"Random Forest:        {acc_rf*100:.2f}%")
    logger.info(f"Logistic Regression:  {acc_lr*100:.2f}%")
    
    if acc_rf > acc_lr:
        logger.info("\n🥇 Ganador: Random Forest")
        logger.info("💡 Recomendación: Usar Random Forest para predicciones")
    else:
        logger.info("\n🥇 Ganador: Logistic Regression")
        logger.info("💡 Recomendación: Usar Logistic Regression para predicciones")
    
    logger.info("\n✅ Entrenamiento completado!")
