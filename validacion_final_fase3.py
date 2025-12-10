"""
Validación Final Fase 3 - Tennis ML Predictor
Consolida todas las validaciones y verifica criterios de éxito
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Importar walk-forward validator
import sys
sys.path.append('.')
from walk_forward_validation import WalkForwardValidator


class FinalValidator:
    """
    Validación final completa de Fase 3
    """
    
    def __init__(self):
        self.results = {}
        
    def validar_todo(self):
        """
        Ejecuta todas las validaciones
        """
        print("=" * 80)
        print("🎯 VALIDACIÓN FINAL FASE 3")
        print("=" * 80)
        
        # 1. Walk-Forward Validation
        print("\n" + "=" * 80)
        print("1️⃣  WALK-FORWARD VALIDATION")
        print("=" * 80)
        wf_summary, wf_folds = self._run_walk_forward()
        self.results['walk_forward'] = {
            'summary': wf_summary,
            'folds': wf_folds
        }
        
        # 2. Comparación Temporal
        print("\n" + "=" * 80)
        print("2️⃣  COMPARACIÓN TEMPORAL")
        print("=" * 80)
        temporal_results = self._comparacion_temporal()
        self.results['temporal'] = temporal_results
        
        # 3. Validación de Ensemble
        print("\n" + "=" * 80)
        print("3️⃣  VALIDACIÓN DE ENSEMBLE")
        print("=" * 80)
        ensemble_results = self._validar_ensemble()
        self.results['ensemble'] = ensemble_results
        
        # 4. Verificación de Criterios
        print("\n" + "=" * 80)
        print("4️⃣  VERIFICACIÓN DE CRITERIOS")
        print("=" * 80)
        criterios = self._verificar_criterios()
        self.results['criterios'] = criterios
        
        # 5. Generar Reporte
        print("\n" + "=" * 80)
        print("5️⃣  GENERANDO REPORTE")
        print("=" * 80)
        self._generar_reporte()
        
        return self.results
    
    def _run_walk_forward(self):
        """
        Ejecuta Walk-Forward Validation
        """
        # Cargar datos
        df = pd.read_csv('datos/processed/dataset_features_fase3_completas.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Cargar features
        with open('resultados/selected_features.txt', 'r') as f:
            feature_columns = [line.strip() for line in f if line.strip()]
        
        # Crear validador
        validator = WalkForwardValidator(n_folds=5)
        
        # Ejecutar
        summary, fold_results = validator.validate_model(
            df=df,
            feature_columns=feature_columns,
            target_column='resultado',
            model_path='modelos/random_forest_calibrado.pkl'
        )
        
        # Generar visualizaciones
        validator.plot_results()
        
        return summary, fold_results
    
    def _comparacion_temporal(self):
        """
        Compara rendimiento en diferentes ventanas temporales
        """
        print("\n📊 Comparando ventanas temporales...")
        
        # Cargar datos
        df = pd.read_csv('datos/processed/dataset_features_fase3_completas.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Cargar features
        with open('resultados/selected_features.txt', 'r') as f:
            feature_columns = [line.strip() for line in f if line.strip()]
        
        # Cargar modelo
        model = joblib.load('modelos/random_forest_calibrado.pkl')
        
        # Definir ventanas
        ventanas = [
            ('2022-2025', '2022-01-01', '2025-12-31'),
            ('2023-2025', '2023-01-01', '2025-12-31'),
            ('2024-2025', '2024-01-01', '2025-12-31'),
            ('2025', '2025-01-01', '2025-12-31')
        ]
        
        resultados = []
        
        for nombre, fecha_inicio, fecha_fin in ventanas:
            # Filtrar datos
            df_ventana = df[
                (df['fecha'] >= fecha_inicio) & 
                (df['fecha'] <= fecha_fin)
            ]
            
            if len(df_ventana) == 0:
                continue
            
            # Split temporal 80/20
            split_idx = int(len(df_ventana) * 0.8)
            df_test = df_ventana.iloc[split_idx:]
            
            if len(df_test) == 0:
                continue
            
            X_test = df_test[feature_columns]
            y_test = df_test['resultado']
            
            # Predecir
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            brier = brier_score_loss(y_test, y_pred_proba)
            
            print(f"\n📅 {nombre}:")
            print(f"   Partidos: {len(df_test)}")
            print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   Brier:    {brier:.4f}")
            
            resultados.append({
                'ventana': nombre,
                'n_partidos': len(df_test),
                'accuracy': accuracy,
                'brier_score': brier
            })
        
        # Crear visualización
        if resultados:
            df_resultados = pd.DataFrame(resultados)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Accuracy
            axes[0].bar(df_resultados['ventana'], df_resultados['accuracy'] * 100, color='#3498db')
            axes[0].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Target 70%')
            axes[0].set_ylabel('Accuracy (%)', fontsize=12)
            axes[0].set_title('Accuracy por Ventana Temporal', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
            
            # Brier Score
            axes[1].bar(df_resultados['ventana'], df_resultados['brier_score'], color='#e74c3c')
            axes[1].axhline(y=0.18, color='g', linestyle='--', alpha=0.5, label='Target < 0.18')
            axes[1].set_ylabel('Brier Score', fontsize=12)
            axes[1].set_title('Brier Score por Ventana Temporal', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            filepath = 'resultados/walk_forward/comparacion_temporal.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\n📊 Gráfico guardado: {filepath}")
            plt.close()
            
            # Guardar CSV
            filepath_csv = 'resultados/walk_forward/comparacion_temporal.csv'
            df_resultados.to_csv(filepath_csv, index=False)
            print(f"📊 Resultados guardados: {filepath_csv}")
        
        return resultados
    
    def _validar_ensemble(self):
        """
        Valida el weighted ensemble
        """
        print("\n📊 Validando weighted ensemble...")
        
        # Verificar si existe el archivo de predicciones del ensemble
        ensemble_file = 'resultados/weighted_ensemble_predictions.csv'
        
        if not Path(ensemble_file).exists():
            print("⚠️  No se encontró archivo de predicciones del ensemble")
            return None
        
        # Cargar predicciones
        df_ensemble = pd.read_csv(ensemble_file)
        
        # Calcular métricas
        y_true = df_ensemble['y_true']
        y_pred_proba = df_ensemble['prob_ensemble']
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_pred_proba)
        
        print(f"\n📊 Métricas Ensemble:")
        print(f"   Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Brier Score: {brier:.4f}")
        
        # Comparar con modelos individuales
        modelos_individuales = []
        for col in df_ensemble.columns:
            if col.startswith('prob_') and col != 'prob_ensemble':
                modelo_name = col.replace('prob_', '')
                y_pred_modelo = (df_ensemble[col] >= 0.5).astype(int)
                acc = accuracy_score(y_true, y_pred_modelo)
                brier_modelo = brier_score_loss(y_true, df_ensemble[col])
                
                modelos_individuales.append({
                    'modelo': modelo_name,
                    'accuracy': acc,
                    'brier_score': brier_modelo
                })
        
        if modelos_individuales:
            df_comp = pd.DataFrame(modelos_individuales)
            df_comp = pd.concat([
                df_comp,
                pd.DataFrame([{
                    'modelo': 'ENSEMBLE',
                    'accuracy': accuracy,
                    'brier_score': brier
                }])
            ])
            
            print("\n📊 Comparación Ensemble vs Individuales:")
            print(df_comp.to_string(index=False))
            
            # Visualización
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Accuracy
            colors = ['#3498db'] * len(modelos_individuales) + ['#2ecc71']
            axes[0].bar(df_comp['modelo'], df_comp['accuracy'] * 100, color=colors)
            axes[0].set_ylabel('Accuracy (%)', fontsize=12)
            axes[0].set_title('Comparación de Accuracy', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='y')
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
            
            # Brier Score
            axes[1].bar(df_comp['modelo'], df_comp['brier_score'], color=colors)
            axes[1].set_ylabel('Brier Score', fontsize=12)
            axes[1].set_title('Comparación de Brier Score', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            filepath = 'resultados/walk_forward/ensemble_comparison.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\n📊 Gráfico guardado: {filepath}")
            plt.close()
        
        return {
            'accuracy': accuracy,
            'brier_score': brier,
            'modelos_individuales': modelos_individuales
        }
    
    def _verificar_criterios(self):
        """
        Verifica si se cumplen los criterios de éxito
        """
        print("\n📋 Verificando criterios de éxito...")
        
        criterios = {
            'accuracy_70': False,
            'brier_018': False,
            'consistencia': False,
            'mejora_fase2': False
        }
        
        # Obtener métricas de Walk-Forward
        wf_summary = self.results['walk_forward']['summary']
        
        # 1. Accuracy >= 70%
        if wf_summary['accuracy_mean'] >= 0.70:
            criterios['accuracy_70'] = True
            print(f"✅ Accuracy >= 70%: {wf_summary['accuracy_mean']*100:.2f}%")
        else:
            print(f"⚠️  Accuracy < 70%: {wf_summary['accuracy_mean']*100:.2f}%")
        
        # 2. Brier Score < 0.18
        if wf_summary['brier_mean'] < 0.18:
            criterios['brier_018'] = True
            print(f"✅ Brier Score < 0.18: {wf_summary['brier_mean']:.4f}")
        else:
            print(f"⚠️  Brier Score >= 0.18: {wf_summary['brier_mean']:.4f}")
        
        # 3. Consistencia (std < 3%)
        if wf_summary['accuracy_std'] < 0.03:
            criterios['consistencia'] = True
            print(f"✅ Modelo consistente: std = {wf_summary['accuracy_std']*100:.2f}%")
        else:
            print(f"⚠️  Modelo variable: std = {wf_summary['accuracy_std']*100:.2f}%")
        
        # 4. Mejora sobre Fase 2 (baseline: 69.82%)
        baseline_fase2 = 0.6982
        if wf_summary['accuracy_mean'] > baseline_fase2:
            criterios['mejora_fase2'] = True
            mejora = (wf_summary['accuracy_mean'] - baseline_fase2) * 100
            print(f"✅ Mejora sobre Fase 2: +{mejora:.2f}%")
        else:
            diferencia = (wf_summary['accuracy_mean'] - baseline_fase2) * 100
            print(f"⚠️  Sin mejora sobre Fase 2: {diferencia:+.2f}%")
        
        # Resumen
        criterios_cumplidos = sum(criterios.values())
        total_criterios = len(criterios)
        
        print(f"\n{'='*60}")
        print(f"📊 CRITERIOS CUMPLIDOS: {criterios_cumplidos}/{total_criterios}")
        print(f"{'='*60}")
        
        if criterios_cumplidos == total_criterios:
            print("🎉 ¡TODOS LOS CRITERIOS CUMPLIDOS!")
        elif criterios_cumplidos >= total_criterios - 1:
            print("✅ Casi todos los criterios cumplidos")
        else:
            print("⚠️  Varios criterios pendientes")
        
        return criterios
    
    def _generar_reporte(self):
        """
        Genera reporte consolidado
        """
        print("\n📄 Generando reporte consolidado...")
        
        # Crear directorio
        Path('resultados/walk_forward').mkdir(parents=True, exist_ok=True)
        
        # Generar reporte en texto
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("REPORTE FINAL - FASE 3")
        report_lines.append("Tennis ML Predictor")
        report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        
        # Walk-Forward Results
        wf_summary = self.results['walk_forward']['summary']
        report_lines.append("\n" + "=" * 80)
        report_lines.append("1. WALK-FORWARD VALIDATION")
        report_lines.append("=" * 80)
        report_lines.append(f"\nNúmero de folds: {wf_summary['n_folds']}")
        report_lines.append(f"\nAccuracy:")
        report_lines.append(f"  Media:  {wf_summary['accuracy_mean']:.4f} ({wf_summary['accuracy_mean']*100:.2f}%)")
        report_lines.append(f"  Std:    {wf_summary['accuracy_std']:.4f}")
        report_lines.append(f"  Min:    {wf_summary['accuracy_min']:.4f}")
        report_lines.append(f"  Max:    {wf_summary['accuracy_max']:.4f}")
        report_lines.append(f"\nBrier Score:")
        report_lines.append(f"  Media:  {wf_summary['brier_mean']:.4f}")
        report_lines.append(f"  Std:    {wf_summary['brier_std']:.4f}")
        report_lines.append(f"  Min:    {wf_summary['brier_min']:.4f}")
        report_lines.append(f"  Max:    {wf_summary['brier_max']:.4f}")
        report_lines.append(f"\nÚltimo Fold (más reciente):")
        report_lines.append(f"  Accuracy:    {wf_summary['last_fold_accuracy']:.4f}")
        report_lines.append(f"  Brier Score: {wf_summary['last_fold_brier']:.4f}")
        
        # Temporal Comparison
        if 'temporal' in self.results and self.results['temporal']:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("2. COMPARACIÓN TEMPORAL")
            report_lines.append("=" * 80)
            for r in self.results['temporal']:
                report_lines.append(f"\n{r['ventana']}:")
                report_lines.append(f"  Partidos: {r['n_partidos']}")
                report_lines.append(f"  Accuracy: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")
                report_lines.append(f"  Brier:    {r['brier_score']:.4f}")
        
        # Ensemble
        if 'ensemble' in self.results and self.results['ensemble']:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("3. WEIGHTED ENSEMBLE")
            report_lines.append("=" * 80)
            ens = self.results['ensemble']
            report_lines.append(f"\nAccuracy:    {ens['accuracy']:.4f} ({ens['accuracy']*100:.2f}%)")
            report_lines.append(f"Brier Score: {ens['brier_score']:.4f}")
        
        # Criterios
        if 'criterios' in self.results:
            report_lines.append("\n" + "=" * 80)
            report_lines.append("4. VERIFICACIÓN DE CRITERIOS")
            report_lines.append("=" * 80)
            crit = self.results['criterios']
            report_lines.append(f"\nAccuracy >= 70%:        {'✅' if crit['accuracy_70'] else '❌'}")
            report_lines.append(f"Brier Score < 0.18:     {'✅' if crit['brier_018'] else '❌'}")
            report_lines.append(f"Consistencia (std<3%):  {'✅' if crit['consistencia'] else '❌'}")
            report_lines.append(f"Mejora sobre Fase 2:    {'✅' if crit['mejora_fase2'] else '❌'}")
            
            cumplidos = sum(crit.values())
            total = len(crit)
            report_lines.append(f"\nCRITERIOS CUMPLIDOS: {cumplidos}/{total}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("FIN DEL REPORTE")
        report_lines.append("=" * 80)
        
        # Guardar reporte
        report_text = "\n".join(report_lines)
        filepath = 'resultados/walk_forward/REPORTE_VALIDACION_FINAL.txt'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✅ Reporte guardado: {filepath}")
        
        # También imprimir en consola
        print("\n" + report_text)


def main():
    """
    Script principal
    """
    print("🎾 Validación Final Fase 3 - Tennis ML Predictor")
    print("=" * 80)
    
    # Crear validador
    validator = FinalValidator()
    
    # Ejecutar todas las validaciones
    results = validator.validar_todo()
    
    print("\n" + "=" * 80)
    print("✅ VALIDACIÓN FINAL COMPLETADA")
    print("=" * 80)
    print("\nArchivos generados:")
    print("  - resultados/walk_forward/walk_forward_metrics.png")
    print("  - resultados/walk_forward/reliability_diagram_last_fold.png")
    print("  - resultados/walk_forward/walk_forward_results.csv")
    print("  - resultados/walk_forward/comparacion_temporal.png")
    print("  - resultados/walk_forward/comparacion_temporal.csv")
    print("  - resultados/walk_forward/ensemble_comparison.png")
    print("  - resultados/walk_forward/REPORTE_VALIDACION_FINAL.txt")
    
    return results


if __name__ == "__main__":
    results = main()
