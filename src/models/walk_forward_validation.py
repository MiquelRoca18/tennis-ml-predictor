"""
Walk-Forward Validation para Modelo de Tenis
Valida el rendimiento del modelo simulando producción real
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Implementa Walk-Forward Validation para series temporales
    
    Simula cómo el modelo funcionaría en producción:
    - Entrena con datos pasados
    - Predice datos futuros
    - Re-entrena periódicamente con nuevos datos
    """
    
    def __init__(self, model, features):
        """
        Args:
            model: Modelo entrenado (sklearn-compatible)
            features: Lista de features a usar
        """
        self.model = model
        self.features = features
        self.results = []
    
    def validate(self, df, date_column='fecha', n_splits=5, min_train_size=0.5):
        """
        Ejecuta Walk-Forward Validation
        
        Args:
            df: DataFrame con datos
            date_column: Nombre de columna de fecha
            n_splits: Número de folds temporales
            min_train_size: Tamaño mínimo de train (% del total)
        
        Returns:
            dict con resultados
        """
        logger.info("="*60)
        logger.info("🔄 WALK-FORWARD VALIDATION")
        logger.info("="*60)
        
        # Ordenar por fecha
        df = df.sort_values(date_column).reset_index(drop=True)
        df[date_column] = pd.to_datetime(df[date_column])
        
        n = len(df)
        min_train_idx = int(n * min_train_size)
        
        # Calcular puntos de corte
        test_size = (n - min_train_idx) // n_splits
        
        logger.info(f"\n📊 Configuración:")
        logger.info(f"   Total datos: {n:,}")
        logger.info(f"   Mín train: {min_train_idx:,} ({min_train_size*100:.0f}%)")
        logger.info(f"   Splits: {n_splits}")
        logger.info(f"   Test size por fold: ~{test_size:,}")
        
        fold_results = []
        
        for fold in range(n_splits):
            # Calcular índices
            train_end = min_train_idx + (fold * test_size)
            test_start = train_end
            test_end = min(test_start + test_size, n)
            
            if test_end - test_start < 100:  # Skip si test muy pequeño
                continue
            
            # Split datos
            train_data = df.iloc[:train_end]
            test_data = df.iloc[test_start:test_end]
            
            X_train = train_data[self.features]
            y_train = train_data['resultado']
            X_test = test_data[self.features]
            y_test = test_data['resultado']
            
            train_dates = (train_data[date_column].min(), train_data[date_column].max())
            test_dates = (test_data[date_column].min(), test_data[date_column].max())
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Fold {fold + 1}/{n_splits}")
            logger.info(f"{'='*60}")
            logger.info(f"   Train: {len(X_train):,} samples ({train_dates[0].date()} a {train_dates[1].date()})")
            logger.info(f"   Test:  {len(X_test):,} samples ({test_dates[0].date()} a {test_dates[1].date()})")
            
            # Entrenar modelo
            logger.info(f"   🔄 Entrenando...")
            model_fold = self._clone_and_train(X_train, y_train)
            
            # Predecir
            y_pred_proba = model_fold.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Métricas
            acc = accuracy_score(y_test, y_pred)
            brier = brier_score_loss(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"   📊 Resultados:")
            logger.info(f"      Accuracy:    {acc*100:.2f}%")
            logger.info(f"      Brier Score: {brier:.4f}")
            logger.info(f"      AUC-ROC:     {auc:.4f}")
            
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_start': train_dates[0],
                'train_end': train_dates[1],
                'test_start': test_dates[0],
                'test_end': test_dates[1],
                'accuracy': acc,
                'brier_score': brier,
                'auc_roc': auc
            })
        
        # Resultados agregados
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 RESULTADOS AGREGADOS")
        logger.info(f"{'='*60}")
        
        accuracies = [r['accuracy'] for r in fold_results]
        briers = [r['brier_score'] for r in fold_results]
        aucs = [r['auc_roc'] for r in fold_results]
        
        logger.info(f"\n🎯 Accuracy:")
        logger.info(f"   Media:  {np.mean(accuracies)*100:.2f}%")
        logger.info(f"   Std:    {np.std(accuracies)*100:.2f}%")
        logger.info(f"   Min:    {np.min(accuracies)*100:.2f}%")
        logger.info(f"   Max:    {np.max(accuracies)*100:.2f}%")
        
        logger.info(f"\n📉 Brier Score:")
        logger.info(f"   Media:  {np.mean(briers):.4f}")
        logger.info(f"   Std:    {np.std(briers):.4f}")
        logger.info(f"   Min:    {np.min(briers):.4f}")
        logger.info(f"   Max:    {np.max(briers):.4f}")
        
        logger.info(f"\n📈 AUC-ROC:")
        logger.info(f"   Media:  {np.mean(aucs):.4f}")
        logger.info(f"   Std:    {np.std(aucs):.4f}")
        
        return {
            'fold_results': fold_results,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_brier': np.mean(briers),
            'std_brier': np.std(briers),
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs)
        }
    
    def _clone_and_train(self, X_train, y_train):
        """Clona y entrena el modelo"""
        from sklearn.base import clone
        model_clone = clone(self.model)
        model_clone.fit(X_train, y_train)
        return model_clone


def main():
    """Ejecuta Walk-Forward Validation en el modelo final"""
    
    logger.info("="*60)
    logger.info("🚀 WALK-FORWARD VALIDATION - MODELO FINAL")
    logger.info("="*60)
    
    # Cargar datos
    logger.info("\n📂 Cargando datos...")
    df = pd.read_csv('datos/processed/dataset_features_fase3_completas.csv')
    logger.info(f"   ✅ {len(df):,} filas cargadas")
    
    # Cargar features seleccionadas
    with open('resultados/selected_features.txt', 'r') as f:
        features = [line.strip() for line in f]
    logger.info(f"   ✅ {len(features)} features cargadas")
    
    # Cargar modelo
    logger.info("\n🤖 Cargando modelo...")
    model = joblib.load('modelos/xgboost_optimizado.pkl')
    logger.info(f"   ✅ Modelo cargado: {type(model).__name__}")
    
    # Ejecutar Walk-Forward Validation
    validator = WalkForwardValidator(model, features)
    results = validator.validate(df, date_column='fecha', n_splits=5, min_train_size=0.5)
    
    # Guardar resultados
    logger.info("\n💾 Guardando resultados...")
    results_df = pd.DataFrame(results['fold_results'])
    results_df.to_csv('resultados/walk_forward_validation_results.csv', index=False)
    logger.info("   ✅ Guardado: resultados/walk_forward_validation_results.csv")
    
    # Comparar con test set original
    logger.info("\n" + "="*60)
    logger.info("📊 COMPARACIÓN CON TEST SET ORIGINAL")
    logger.info("="*60)
    
    # Cargar métricas del weighted ensemble
    try:
        ensemble_metrics = pd.read_csv('resultados/weighted_ensemble_metrics.csv')
        original_acc = ensemble_metrics['accuracy'].values[0]
        original_brier = ensemble_metrics['brier_score'].values[0]
        
        logger.info(f"\n🎯 Test Set Original:")
        logger.info(f"   Accuracy:    {original_acc*100:.2f}%")
        logger.info(f"   Brier Score: {original_brier:.4f}")
        
        logger.info(f"\n🔄 Walk-Forward (Media):")
        logger.info(f"   Accuracy:    {results['mean_accuracy']*100:.2f}%")
        logger.info(f"   Brier Score: {results['mean_brier']:.4f}")
        
        diff_acc = (results['mean_accuracy'] - original_acc) * 100
        diff_brier = results['mean_brier'] - original_brier
        
        logger.info(f"\n📈 Diferencia:")
        logger.info(f"   Accuracy:    {diff_acc:+.2f}%")
        logger.info(f"   Brier Score: {diff_brier:+.4f}")
        
        # Interpretación
        logger.info(f"\n" + "="*60)
        logger.info(f"💡 INTERPRETACIÓN")
        logger.info(f"="*60)
        
        if abs(diff_acc) < 0.5:
            logger.info(f"\n✅ El modelo es CONSISTENTE")
            logger.info(f"   La diferencia es mínima ({diff_acc:+.2f}%)")
            logger.info(f"   El {original_acc*100:.2f}% es una estimación REALISTA")
        elif diff_acc < -0.5:
            logger.info(f"\n⚠️  Posible OVERFITTING")
            logger.info(f"   Walk-Forward es {abs(diff_acc):.2f}% peor")
            logger.info(f"   El rendimiento real podría ser ~{results['mean_accuracy']*100:.2f}%")
        else:
            logger.info(f"\n🎉 El modelo es MEJOR de lo esperado")
            logger.info(f"   Walk-Forward es {diff_acc:.2f}% mejor")
            logger.info(f"   El rendimiento real podría ser ~{results['mean_accuracy']*100:.2f}%")
        
    except Exception as e:
        logger.warning(f"   ⚠️  No se pudo cargar métricas originales: {e}")
    
    logger.info(f"\n✅ Walk-Forward Validation completada!")


if __name__ == "__main__":
    main()
