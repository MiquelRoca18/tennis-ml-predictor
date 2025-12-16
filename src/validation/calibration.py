"""
Módulo de Validación de Calibración
===================================

Lógica refactorizada para validar calibración de modelos.
Extraído y optimizado desde validacion_calibracion.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from pathlib import Path
import json
import logging

from src.utils import print_header, print_metric, create_directory, load_model

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

logger = logging.getLogger(__name__)


class CalibrationValidator:
    """Validador de calibración de modelos - Versión optimizada"""
    
    def __init__(self, modelos_dir="modelos", resultados_dir="resultados/calibracion"):
        self.modelos_dir = Path(modelos_dir)
        self.resultados_dir = create_directory(resultados_dir)
        create_directory(self.resultados_dir / "reliability_diagrams")
        
        self.modelos = {}
        self.metricas = {}
    
    def cargar_modelos(self):
        """Carga todos los modelos calibrados"""
        print_header("CARGANDO MODELOS CALIBRADOS", "📂")
        
        archivos_modelos = list(self.modelos_dir.glob("*_calibrado.pkl"))
        
        if not archivos_modelos:
            logger.error("❌ No se encontraron modelos calibrados")
            return False
        
        for archivo in archivos_modelos:
            nombre = archivo.stem.replace("_calibrado", "").replace("_", " ").title()
            try:
                self.modelos[nombre] = load_model(archivo)
                print(f"✅ {nombre}: {archivo.name}")
            except Exception as e:
                logger.error(f"❌ Error cargando {archivo.name}: {e}")
        
        print(f"\n📊 Total modelos cargados: {len(self.modelos)}")
        return True
    
    def calcular_ece(self, y_true, y_prob, n_bins=10):
        """
        Calcula Expected Calibration Error (ECE)
        
        ECE < 0.05 = Excelente
        ECE < 0.10 = Bueno
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def evaluar_modelo(self, nombre, modelo, X_test, y_test):
        """Evalúa un modelo y calcula métricas de calibración"""
        print_header(f"EVALUANDO: {nombre}")
        
        # Predecir
        y_prob = modelo.predict_proba(X_test)[:, 1]
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        accuracy = np.mean(y_pred == y_test)
        brier = brier_score_loss(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)
        ece = self.calcular_ece(y_test, y_prob, n_bins=10)
        
        # Curva de calibración
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        
        metricas = {
            'accuracy': accuracy,
            'brier_score': brier,
            'log_loss': logloss,
            'ece': ece,
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
        
        # Mostrar métricas
        print_metric("Accuracy", accuracy * 100, "%", 2)
        print_metric("Brier Score", brier, " ✅" if brier < 0.20 else " ⚠️", 4)
        print_metric("Log Loss", logloss, "", 4)
        print_metric("ECE", ece, " ✅" if ece < 0.05 else " ⚠️" if ece < 0.10 else " ❌", 4)
        
        # Interpretación
        if ece < 0.05:
            print("  📈 Calibración: EXCELENTE")
        elif ece < 0.10:
            print("  📈 Calibración: BUENA")
        else:
            print("  📈 Calibración: MEJORABLE")
        
        return metricas, y_prob
    
    def crear_reliability_diagram(self, nombre, y_true, y_prob, metricas):
        """Crea Reliability Diagram (optimizado)"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Reliability Curve
        ax = axes[0]
        prob_true = np.array(metricas['prob_true'])
        prob_pred = np.array(metricas['prob_pred'])
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectamente Calibrado', alpha=0.7)
        ax.plot(prob_pred, prob_true, 'o-', linewidth=3, markersize=10, 
                color='#3498db', label=nombre, markeredgecolor='white', markeredgewidth=2)
        
        ax.set_xlabel('Probabilidad Predicha', fontsize=13, fontweight='bold')
        ax.set_ylabel('Fracción Real de Positivos', fontsize=13, fontweight='bold')
        ax.set_title(f'Reliability Diagram - {nombre}', fontsize=15, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Métricas como texto
        metrics_text = f"""Métricas:
Brier: {metricas['brier_score']:.4f}
ECE: {metricas['ece']:.4f}
Accuracy: {metricas['accuracy']*100:.2f}%"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
        
        # Subplot 2: Distribución de Probabilidades
        ax = axes[1]
        ax.hist(y_prob, bins=20, alpha=0.7, color='#3498db', edgecolor='black', linewidth=1.2)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Umbral 0.5')
        
        ax.set_xlabel('Probabilidad Predicha', fontsize=13, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=13, fontweight='bold')
        ax.set_title(f'Distribución - {nombre}', fontsize=15, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        
        filename = self.resultados_dir / "reliability_diagrams" / f"{nombre.lower().replace(' ', '_')}_reliability.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  💾 Gráfico guardado: {filename.name}")
        plt.close()
    
    def generar_resumen(self, metricas_todos):
        """Genera resumen de métricas (optimizado)"""
        print_header("GUARDANDO RESUMEN", "💾")
        
        # DataFrame
        df_metricas = pd.DataFrame({
            'Modelo': list(metricas_todos.keys()),
            'Accuracy': [m['accuracy'] for m in metricas_todos.values()],
            'Brier_Score': [m['brier_score'] for m in metricas_todos.values()],
            'ECE': [m['ece'] for m in metricas_todos.values()]
        }).sort_values('Brier_Score')
        
        # Guardar
        df_metricas.to_csv(self.resultados_dir / "calibration_metrics.csv", index=False)
        
        with open(self.resultados_dir / "calibration_analysis.json", 'w') as f:
            json.dump(metricas_todos, f, indent=2)
        
        print("\n📊 RESUMEN DE MÉTRICAS:")
        print(df_metricas.to_string(index=False))
        
        # Mejor modelo
        mejor = df_metricas.iloc[0]
        print(f"\n🏆 MEJOR MODELO: {mejor['Modelo']}")
        print_metric("Accuracy", mejor['Accuracy'] * 100, "%", 2)
        print_metric("Brier Score", mejor['Brier_Score'], "", 4)
        print_metric("ECE", mejor['ECE'], "", 4)
        
        return df_metricas
    
    def validar_todos(self, X_test, y_test):
        """Ejecuta validación completa"""
        print_header("VALIDACIÓN DE CALIBRACIÓN", "🎯")
        
        metricas_todos = {}
        
        for nombre, modelo in self.modelos.items():
            metricas, y_prob = self.evaluar_modelo(nombre, modelo, X_test, y_test)
            metricas_todos[nombre] = metricas
            self.crear_reliability_diagram(nombre, y_test, y_prob, metricas)
        
        df_resumen = self.generar_resumen(metricas_todos)
        
        print(f"\n📁 Resultados en: {self.resultados_dir}")
        
        return metricas_todos, df_resumen


def validar_calibracion():
    """Función principal de validación de calibración"""
    from src.config import Config
    
    validator = CalibrationValidator()
    
    if not validator.cargar_modelos():
        return False
    
    # Cargar datos
    print_header("CARGANDO DATOS DE TEST", "📂")
    
    # Usar el dataset con todas las features generadas en Fase 3
    df = pd.read_csv("datos/processed/dataset_features_fase3_completas.csv")
    
    # Cargar features seleccionadas desde el archivo de texto
    with open("resultados/selected_features.txt", "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Total partidos: {len(df)}")
    
    # Split test (últimos 20%)
    n = len(df)
    test_start = int(n * 0.8)
    
    X_test = df.iloc[test_start:][feature_cols]
    y_test = df.iloc[test_start:]['resultado']
    
    print(f"  Test set: {len(X_test)} partidos ({len(X_test)/n*100:.1f}%)")
    
    # Validar
    metricas, df_resumen = validator.validar_todos(X_test, y_test)
    
    print_header("VALIDACIÓN COMPLETADA", "✅")
    
    return True
