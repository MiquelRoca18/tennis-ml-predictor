"""
Walk-Forward Validation para Tennis ML Predictor
Valida el modelo con folds temporales para evitar data leakage
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suprimir warnings de sklearn para output más limpio
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='X does not have valid feature names')
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve

class WalkForwardValidator:
    """
    Implementa Walk-Forward Validation con folds temporales
    """
    
    def __init__(self, n_folds=5):
        """
        Args:
            n_folds: Número de folds temporales
        """
        self.n_folds = n_folds
        self.results = []
        
    def create_temporal_folds(self, df):
        """
        Crea folds temporales ordenados por fecha
        
        Args:
            df: DataFrame con columna 'fecha'
            
        Returns:
            Lista de tuplas (train_indices, test_indices)
        """
        # Asegurar que está ordenado por fecha
        df = df.sort_values('fecha').reset_index(drop=True)
        
        n_samples = len(df)
        fold_size = n_samples // (self.n_folds + 1)  # +1 porque necesitamos datos iniciales de train
        
        folds = []
        
        for i in range(self.n_folds):
            # Train: desde inicio hasta el final del fold i
            train_end = fold_size * (i + 2)
            train_indices = list(range(train_end))
            
            # Test: siguiente fold
            test_start = train_end
            test_end = min(test_start + fold_size, n_samples)
            test_indices = list(range(test_start, test_end))
            
            if len(test_indices) > 0:
                folds.append((train_indices, test_indices))
        
        return folds
    
    def validate_model(self, df, feature_columns, target_column='resultado', 
                      model_path=None, model=None):
        """
        Ejecuta Walk-Forward Validation
        
        Args:
            df: DataFrame con features y target
            feature_columns: Lista de nombres de columnas de features
            target_column: Nombre de la columna target
            model_path: Path al modelo guardado (opcional)
            model: Modelo ya cargado (opcional)
            
        Returns:
            dict con resultados agregados
        """
        
        print("=" * 80)
        print("🔄 WALK-FORWARD VALIDATION")
        print("=" * 80)
        
        # Cargar modelo si se proporciona path
        if model_path and model is None:
            model = joblib.load(model_path)
            print(f"✅ Modelo cargado: {model_path}")
        
        # Crear folds temporales
        folds = self.create_temporal_folds(df)
        print(f"\n📊 Folds temporales creados: {len(folds)}")
        
        # Validar cada fold
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"📈 FOLD {fold_idx + 1}/{len(folds)}")
            print(f"{'='*60}")
            
            # Split datos
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            
            X_train = df_train[feature_columns]
            y_train = df_train[target_column]
            X_test = df_test[feature_columns]
            y_test = df_test[target_column]
            
            # Fechas
            fecha_train_min = df_train['fecha'].min()
            fecha_train_max = df_train['fecha'].max()
            fecha_test_min = df_test['fecha'].min()
            fecha_test_max = df_test['fecha'].max()
            
            print(f"📅 Train: {fecha_train_min.date()} - {fecha_train_max.date()} ({len(df_train)} partidos)")
            print(f"📅 Test:  {fecha_test_min.date()} - {fecha_test_max.date()} ({len(df_test)} partidos)")
            
            # Re-entrenar modelo en este fold
            print(f"🔄 Entrenando modelo en fold {fold_idx + 1}...")
            
            # Clonar modelo base (manejar modelos calibrados)
            from sklearn.base import clone
            from sklearn.calibration import CalibratedClassifierCV
            
            # Si es un modelo calibrado, extraer el estimador base
            if isinstance(model, CalibratedClassifierCV):
                # Obtener el estimador base del primer calibrador
                base_estimator = model.calibrated_classifiers_[0].estimator
                fold_model_base = clone(base_estimator)
                
                # Entrenar y calibrar con cv=5
                fold_model = CalibratedClassifierCV(fold_model_base, method='isotonic', cv=5)
                fold_model.fit(X_train, y_train)
            else:
                # Modelo normal
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)
            
            # Predecir
            y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            brier = brier_score_loss(y_test, y_pred_proba)
            logloss = log_loss(y_test, y_pred_proba)
            
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = np.nan
            
            # ECE (Expected Calibration Error)
            ece = self._calculate_ece(y_test, y_pred_proba)
            
            print(f"\n📊 Métricas Fold {fold_idx + 1}:")
            print(f"   Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   Brier Score: {brier:.4f}")
            print(f"   Log Loss:    {logloss:.4f}")
            print(f"   AUC-ROC:     {auc:.4f}")
            print(f"   ECE:         {ece:.4f}")
            
            # Guardar resultados
            fold_result = {
                'fold': fold_idx + 1,
                'train_size': len(df_train),
                'test_size': len(df_test),
                'train_date_min': fecha_train_min,
                'train_date_max': fecha_train_max,
                'test_date_min': fecha_test_min,
                'test_date_max': fecha_test_max,
                'accuracy': accuracy,
                'brier_score': brier,
                'log_loss': logloss,
                'auc_roc': auc,
                'ece': ece,
                'y_true': y_test.values,
                'y_pred_proba': y_pred_proba
            }
            
            fold_results.append(fold_result)
        
        # Calcular estadísticas agregadas
        self.results = fold_results
        summary = self._calculate_summary(fold_results)
        
        # Mostrar resumen
        self._print_summary(summary)
        
        return summary, fold_results
    
    def _calculate_ece(self, y_true, y_pred_proba, n_bins=10):
        """
        Calcula Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_summary(self, fold_results):
        """
        Calcula estadísticas agregadas de todos los folds
        """
        accuracies = [r['accuracy'] for r in fold_results]
        brier_scores = [r['brier_score'] for r in fold_results]
        log_losses = [r['log_loss'] for r in fold_results]
        eces = [r['ece'] for r in fold_results]
        
        summary = {
            'n_folds': len(fold_results),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_min': np.min(accuracies),
            'accuracy_max': np.max(accuracies),
            'brier_mean': np.mean(brier_scores),
            'brier_std': np.std(brier_scores),
            'brier_min': np.min(brier_scores),
            'brier_max': np.max(brier_scores),
            'logloss_mean': np.mean(log_losses),
            'ece_mean': np.mean(eces),
            'last_fold_accuracy': accuracies[-1],
            'last_fold_brier': brier_scores[-1],
            'trend': 'improving' if accuracies[-1] > accuracies[0] else 'declining'
        }
        
        return summary
    
    def _print_summary(self, summary):
        """
        Imprime resumen de resultados
        """
        print("\n" + "=" * 80)
        print("📊 RESUMEN WALK-FORWARD VALIDATION")
        print("=" * 80)
        
        print(f"\n🎯 Accuracy:")
        print(f"   Media:  {summary['accuracy_mean']:.4f} ({summary['accuracy_mean']*100:.2f}%)")
        print(f"   Std:    {summary['accuracy_std']:.4f}")
        print(f"   Min:    {summary['accuracy_min']:.4f} ({summary['accuracy_min']*100:.2f}%)")
        print(f"   Max:    {summary['accuracy_max']:.4f} ({summary['accuracy_max']*100:.2f}%)")
        
        print(f"\n📉 Brier Score:")
        print(f"   Media:  {summary['brier_mean']:.4f}")
        print(f"   Std:    {summary['brier_std']:.4f}")
        print(f"   Min:    {summary['brier_min']:.4f}")
        print(f"   Max:    {summary['brier_max']:.4f}")
        
        print(f"\n📈 Último Fold (más reciente):")
        print(f"   Accuracy:    {summary['last_fold_accuracy']:.4f} ({summary['last_fold_accuracy']*100:.2f}%)")
        print(f"   Brier Score: {summary['last_fold_brier']:.4f}")
        
        print(f"\n📊 Tendencia: {summary['trend'].upper()}")
        
        # Evaluación
        print(f"\n{'='*60}")
        print("✅ EVALUACIÓN:")
        
        if summary['accuracy_mean'] >= 0.70:
            print("   ✅ Accuracy promedio >= 70%")
        else:
            print(f"   ⚠️  Accuracy promedio < 70% (actual: {summary['accuracy_mean']*100:.2f}%)")
        
        if summary['brier_mean'] < 0.18:
            print("   ✅ Brier Score promedio < 0.18")
        else:
            print(f"   ⚠️  Brier Score promedio >= 0.18 (actual: {summary['brier_mean']:.4f})")
        
        if summary['accuracy_std'] < 0.03:
            print("   ✅ Modelo consistente (std < 3%)")
        else:
            print(f"   ⚠️  Modelo variable (std = {summary['accuracy_std']*100:.2f}%)")
    
    def plot_results(self, output_dir='resultados/walk_forward'):
        """
        Genera visualizaciones de resultados
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            print("⚠️  No hay resultados para visualizar")
            return
        
        # 1. Accuracy y Brier por fold
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        folds = [r['fold'] for r in self.results]
        accuracies = [r['accuracy'] * 100 for r in self.results]
        brier_scores = [r['brier_score'] for r in self.results]
        
        # Accuracy
        axes[0].plot(folds, accuracies, 'o-', linewidth=2, markersize=8, color='#2ecc71')
        axes[0].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Target 70%')
        axes[0].set_xlabel('Fold', fontsize=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Accuracy por Fold Temporal', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Brier Score
        axes[1].plot(folds, brier_scores, 'o-', linewidth=2, markersize=8, color='#e74c3c')
        axes[1].axhline(y=0.18, color='g', linestyle='--', alpha=0.5, label='Target < 0.18')
        axes[1].set_xlabel('Fold', fontsize=12)
        axes[1].set_ylabel('Brier Score', fontsize=12)
        axes[1].set_title('Brier Score por Fold Temporal', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        filepath = Path(output_dir) / 'walk_forward_metrics.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\n📊 Gráfico guardado: {filepath}")
        plt.close()
        
        # 2. Reliability diagram del último fold
        last_fold = self.results[-1]
        y_true = last_fold['y_true']
        y_pred_proba = last_fold['y_pred_proba']
        
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado', linewidth=2)
        plt.plot(prob_pred, prob_true, 'o-', label=f'Fold {last_fold["fold"]}', linewidth=2, markersize=8)
        plt.xlabel('Probabilidad Predicha', fontsize=12)
        plt.ylabel('Fracción Positiva', fontsize=12)
        plt.title(f'Reliability Diagram - Último Fold (más reciente)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        brier = last_fold['brier_score']
        plt.text(0.05, 0.95, f'Brier Score: {brier:.4f}', 
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=11)
        
        filepath = Path(output_dir) / 'reliability_diagram_last_fold.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"📊 Gráfico guardado: {filepath}")
        plt.close()
        
        # 3. Tabla de resultados
        df_results = pd.DataFrame([
            {
                'Fold': r['fold'],
                'Test Inicio': r['test_date_min'].strftime('%Y-%m-%d'),
                'Test Fin': r['test_date_max'].strftime('%Y-%m-%d'),
                'N Partidos': r['test_size'],
                'Accuracy': f"{r['accuracy']:.4f}",
                'Brier Score': f"{r['brier_score']:.4f}",
                'ECE': f"{r['ece']:.4f}"
            }
            for r in self.results
        ])
        
        filepath = Path(output_dir) / 'walk_forward_results.csv'
        df_results.to_csv(filepath, index=False)
        print(f"📊 Resultados guardados: {filepath}")
        
        return df_results


def main():
    """
    Script principal
    """
    print("🎾 Walk-Forward Validation - Tennis ML Predictor")
    print("=" * 80)
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    df = pd.read_csv('datos/processed/dataset_features_fase3_completas.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    print(f"✅ Dataset cargado: {len(df)} partidos")
    print(f"📅 Rango: {df['fecha'].min().date()} - {df['fecha'].max().date()}")
    
    # Cargar features seleccionadas
    with open('resultados/selected_features.txt', 'r') as f:
        feature_columns = [line.strip() for line in f if line.strip()]
    
    print(f"✅ Features cargadas: {len(feature_columns)}")
    
    # Cargar modelo
    model_path = 'modelos/random_forest_calibrado.pkl'
    print(f"✅ Modelo: {model_path}")
    
    # Crear validador
    validator = WalkForwardValidator(n_folds=5)
    
    # Ejecutar validación
    summary, fold_results = validator.validate_model(
        df=df,
        feature_columns=feature_columns,
        target_column='resultado',
        model_path=model_path
    )
    
    # Generar visualizaciones
    print("\n" + "=" * 80)
    print("📊 Generando visualizaciones...")
    df_results = validator.plot_results()
    
    print("\n" + "=" * 80)
    print("✅ WALK-FORWARD VALIDATION COMPLETADA")
    print("=" * 80)
    
    return summary, fold_results


if __name__ == "__main__":
    summary, fold_results = main()
